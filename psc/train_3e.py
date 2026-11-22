"""Tete 3e : cout configurable par une query, d'apres la description de ConfigWM
donnee par le tuteur. h(z, q) conditionnee par FiLM. Positifs : etats dont les
facteurs pertinents pour q sont proches dans l'etat simulateur. Negatifs : reste
du batch et meme etat sous une autre query. InfoNCE bidirectionnelle + SIGReg.

A lancer depuis h_jepa/ :
  python ../psc/train_3e.py --env cube --cache cache_cube_train_seed42.pt \
      --h5 $HJEPA_HOME/cube_single_expert_train.h5 --out-dir heads_cube_seed42
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).parent))

from factors import QUERY_NAMES, load_factors, standardize  # noqa: E402
from heads import QueryHead, effective_rank, save_head  # noqa: E402
from loss import SIGReg  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-query", action="store_true", help="controle : toujours la query tout")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--pool", type=int, default=8192)
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--w-sigreg", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = [e["z"].float() for e in torch.load(args.cache, weights_only=False)]
    Z = torch.cat(episodes).to(device)
    fac, _ = standardize(load_factors(args.h5, args.env, [len(z) for z in episodes]))
    fac = {k: v.to(device) for k, v in fac.items()}
    F_q = [fac["objet"], fac["agent"], torch.cat([fac["objet"], fac["agent"]], dim=1)]
    names = QUERY_NAMES[args.env]
    print(f"{len(Z)} latents, queries {names}")

    head = QueryHead(Z.shape[-1], n_queries=3, mean=Z.mean(0), std=Z.std(0) + 1e-6).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sigreg = SIGReg().to(device)
    probe = Z[torch.randperm(len(Z), device=device)[:10000]]
    labels = torch.arange(args.batch, device=device)

    hist = []
    for step in range(args.steps + 1):
        if step % 500 == 0:
            with torch.no_grad():
                rec = {"step": step}
                for k, n in enumerate(names):
                    rec[f"erank_{n}"] = effective_rank(head(probe, k))
            hist.append(rec)
            print(json.dumps(rec))
        if step == args.steps:
            break
        i = torch.randint(len(Z), (args.batch,), device=device)
        pool = torch.randint(len(Z), (args.pool,), device=device)
        if args.no_query:
            q = torch.full((args.batch,), 2, device=device)
        else:
            q = torch.randint(3, (args.batch,), device=device)
        pos = torch.empty_like(i)
        for k in range(3):
            m = q == k
            if m.any():
                d = torch.cdist(F_q[k][i[m]], F_q[k][pool])
                d[i[m][:, None] == pool[None]] = float("inf")
                pos[m] = pool[d.argmin(1)]
        u, v = head(Z[i], q), head(Z[pos], q)
        un, vn = F.normalize(u, dim=-1), F.normalize(v, dim=-1)
        logits = un @ vn.T / args.temp
        if args.no_query:
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        else:
            q_other = (q + torch.randint(1, 3, (args.batch,), device=device)) % 3
            wn = F.normalize(head(Z[i], q_other), dim=-1)
            extra = (un * wn).sum(-1, keepdim=True) / args.temp
            loss = F.cross_entropy(torch.cat([logits, extra], 1), labels)
            loss = loss + F.cross_entropy(logits.T, labels)
        loss = loss + args.w_sigreg * sigreg(u.unsqueeze(0))
        opt.zero_grad()
        loss.backward()
        opt.step()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra = {"args": vars(args), "hist": hist}
    if args.no_query:
        head.cfg["query"] = head.query = 2
        save_head(head, out / "configwm_sansq.pt", extra=extra)
        (out / "configwm_sansq.json").write_text(json.dumps(extra, indent=1))
    else:
        for k, n in enumerate(names):
            h = copy.deepcopy(head)
            h.cfg["query"] = h.query = k
            save_head(h, out / f"configwm_q_{n}.pt", extra=extra)
        (out / "configwm.json").write_text(json.dumps(extra, indent=1))
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
