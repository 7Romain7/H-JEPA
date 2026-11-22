"""Diagnostic offline d'un cout de planning (etape 2 du protocole).

Sur des trajectoires held-out, mesure pour chaque cout c(z, z_goal) :
  - la correlation de Spearman entre le cout et le nombre de steps restants ;
  - la monotonicite, proportion de pas consecutifs ou le cout baisse.
Les deux mesures sont faites sur les latents encodes et sur les latents predits
par le predicteur en rollout sous les actions reellement jouees.

A lancer depuis h_jepa/ :
  python ../psc/diagnostic.py --ckpt $RUN/pusht_lewm_object.ckpt \
      --h5 $HJEPA_HOME/pusht_expert_val.h5 --offsets 15 --out diag_pusht_seed42.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (filtres de compression des fichiers Cube)
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- couts
def cost_l2(z, zg):
    """Cout de la baseline : somme des ecarts au carre (meme rang que la MSE)."""
    return ((z - zg) ** 2).sum(-1)


COSTS = {"l2": cost_l2}


# ---------------------------------------------------------------- modele
def load_model(ckpt, device):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not hasattr(obj, "get_cost"):
        obj = next(m for m in obj.modules() if hasattr(m, "get_cost"))
    return obj.to(device).eval()


def norm_fn(stats, key, device):
    mean = torch.as_tensor(stats[key]["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(stats[key]["std"], dtype=torch.float32, device=device)

    def fn(x):
        return torch.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    return fn


@torch.no_grad()
def encode_episode(model, f, start, length, frameskip, stats, device, batch=64):
    """Encode un episode toutes les `frameskip` frames.

    Rend z (T, D) et les embeddings d'action (T, A_emb), l'action k couvrant les
    frames [k*frameskip, (k+1)*frameskip).
    """
    n_frames = length // frameskip  # nombre de blocs d'actions complets
    idx = start + np.arange(n_frames) * frameskip
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    need_proprio = hasattr(model.encoder, "proprio_encoder")
    proprio_key = getattr(model.encoder, "proprio_key", "proprio")

    act = torch.from_numpy(f["action"][start : start + n_frames * frameskip]).to(device)
    act = norm_fn(stats, "action", device)(act.float())
    act = act.reshape(n_frames, -1)  # (T, frameskip * action_dim)

    zs, aes = [], []
    for i in range(0, n_frames, batch):
        sel = idx[i : i + batch]
        px = torch.from_numpy(f["pixels"][sel[0] : sel[-1] + 1 : frameskip]).to(device)
        px = px.permute(0, 3, 1, 2).float().div(255.0).sub(mean).div(std)
        info = {"pixels": px.unsqueeze(1), "action": act[i : i + batch].unsqueeze(1)}
        if need_proprio:
            pr = torch.from_numpy(f[proprio_key][sel[0] : sel[-1] + 1 : frameskip]).to(device)
            info[proprio_key] = norm_fn(stats, proprio_key, device)(pr.float()).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            out = model.encode(info)
        zs.append(out["embed_0"][:, 0].float())
        aes.append(out["action_0"][:, 0].float())
    return torch.cat(zs), torch.cat(aes)


@torch.no_grad()
def rollout(model, z0, act_emb, history):
    """z0 (B, D), act_emb (B, G, A_emb) -> latents predits (B, G, D) aux pas 1..G."""
    emb = z0.unsqueeze(1)
    for t in range(act_emb.size(1)):
        lo = max(0, t + 1 - history)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            nxt = model.predict(emb[:, lo:], act_emb[:, lo : t + 1])[:, -1:]
        emb = torch.cat([emb, nxt.float()], dim=1)
    return emb[:, 1:]


# ---------------------------------------------------------------- mesures
def measures(cost, remaining):
    """cost, remaining : (N, K), K positions le long de chaque fenetre."""
    c = cost.cpu().numpy()
    r = remaining.cpu().numpy()
    pooled = spearmanr(c.ravel(), r.ravel()).statistic
    per_win = [spearmanr(ci, ri).statistic for ci, ri in zip(c, r)]
    per_win = np.array([x for x in per_win if np.isfinite(x)])
    mono = (c[:, 1:] < c[:, :-1]).mean(axis=1)
    return {
        "spearman_pooled": float(pooled),
        "spearman_par_fenetre": float(per_win.mean()),
        "monotonicite": float(mono.mean()),
        "fenetres_parfaitement_monotones": float((mono == 1.0).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--offsets", type=int, nargs="+", required=True,
                   help="distances start->goal, en pas latents")
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=200)
    p.add_argument("--stride", type=int, default=1, help="pas entre deux fenetres")
    p.add_argument("--save-cache", default=None)
    p.add_argument("--only-cache", action="store_true")
    p.add_argument("--heads", nargs="*", default=[], help="fichiers .pt de projectors")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    stats = torch.load(Path(args.ckpt).parent / "normalizer.pt", weights_only=False)["stats"]

    episodes = []
    with h5py.File(args.h5, "r") as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        for e in range(min(len(ep_len), args.max_episodes)):
            z, ae = encode_episode(model, f, int(ep_off[e]), int(ep_len[e]),
                                   args.frameskip, stats, device)
            episodes.append({"z": z, "act_emb": ae})
    lens = [len(e["z"]) for e in episodes]
    print(f"{len(episodes)} episodes, {sum(lens)} latents, dim {episodes[0]['z'].shape[-1]}, "
          f"longueurs {min(lens)}-{max(lens)}")
    if args.save_cache:
        torch.save([{k: v.cpu() for k, v in e.items()} for e in episodes], args.save_cache)

    if args.only_cache:
        print("CACHE_DONE")
        return
    sys.path.insert(0, str(Path(__file__).parent))
    from heads import load_head
    for path in args.heads:
        head = load_head(path, device)
        COSTS[Path(path).stem] = head.cost
    results = {"ckpt": args.ckpt, "h5": args.h5, "episodes": len(episodes), "offsets": {}}
    for G in args.offsets:
        Z, A = [], []
        for e in episodes:
            T = len(e["z"])
            for s in range(0, T - G, args.stride):
                Z.append(e["z"][s : s + G + 1])
                A.append(e["act_emb"][s : s + G])
        if not Z:
            continue
        Z, A = torch.stack(Z), torch.stack(A)  # (N, G+1, D), (N, G, A_emb)
        pred = torch.cat([rollout(model, Z[i : i + 512, 0], A[i : i + 512], args.history)
                          for i in range(0, len(Z), 512)])  # (N, G, D), pas 1..G
        goal = Z[:, -1:]
        # positions k = 1..G-1 : il reste G-k pas, de G-1 a 1
        remaining = torch.arange(G - 1, 0, -1).expand(len(Z), -1)
        enc_pos, pred_pos = Z[:, 1:G], pred[:, : G - 1]
        gap = ((pred_pos - enc_pos) ** 2).sum(-1).mean().item()
        scale = ((enc_pos - goal) ** 2).sum(-1).mean().item()
        res = {"fenetres": len(Z), "ecart_predit_encode_relatif": gap / scale, "couts": {}}
        for name, fn in COSTS.items():
            with torch.no_grad():
              res["couts"][name] = {
                "encode": measures(fn(enc_pos, goal), remaining),
                "predit": measures(fn(pred_pos, goal), remaining),
              }
        results["offsets"][str(G)] = res
        print(f"--- offset {G} ({len(Z)} fenetres), ecart predit/encode {gap / scale:.3f}")
        for name, r in res["couts"].items():
            for kind in ("encode", "predit"):
                m = r[kind]
                print(f"{name:>22} {kind:>7}  spearman {m['spearman_pooled']:.3f} "
                      f"(par fenetre {m['spearman_par_fenetre']:.3f})  "
                      f"monotonicite {m['monotonicite']:.3f}")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
