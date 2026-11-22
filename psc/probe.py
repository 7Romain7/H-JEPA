"""Linear probe : la position de l'objet se decode-t-elle des latents encodes
et des latents predits en rollout ? Prealable a la tete 3e.

A lancer depuis h_jepa/ :
  python ../psc/probe.py --env cube --ckpt $CK --cache cache_cube_train_seed42.pt \
      --train-h5 $HJEPA_HOME/cube_single_expert_train.h5 \
      --val-h5 $HJEPA_HOME/cube_single_expert_val.h5 --out probe.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).parent))

from diagnostic import encode_episode, load_model, rollout  # noqa: E402
from factors import load_factors  # noqa: E402


def r2(pred, target):
    res = ((pred - target) ** 2).sum(0)
    tot = ((target - target.mean(0)) ** 2).sum(0)
    return float((1 - res / tot).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--train-h5", required=True)
    p.add_argument("--val-h5", required=True)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 15])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = [e["z"].float() for e in torch.load(args.cache, weights_only=False)]
    Ztr = torch.cat(train).to(device).double()
    ftr = load_factors(args.train_h5, args.env, [len(z) for z in train])
    mean, std = Ztr.mean(0), Ztr.std(0) + 1e-6
    ones = torch.ones(len(Ztr), 1, device=device, dtype=torch.double)
    X = torch.cat([(Ztr - mean) / std, ones], 1)
    A = X.T @ X + 1e-3 * len(X) * torch.eye(X.shape[1], device=device, dtype=torch.double)
    W, tstats = {}, {}
    for k, v in ftr.items():
        v = v.to(device).double()
        tstats[k] = (v.mean(0), v.std(0) + 1e-6)
        W[k] = torch.linalg.solve(A, X.T @ ((v - tstats[k][0]) / tstats[k][1]))

    def predict(z, k):
        o = torch.ones(len(z), 1, device=device, dtype=torch.double)
        x = torch.cat([(z.double() - mean) / std, o], 1)
        return x @ W[k] * tstats[k][1] + tstats[k][0]

    model = load_model(args.ckpt, device)
    stats = torch.load(Path(args.ckpt).parent / "normalizer.pt", weights_only=False)["stats"]
    eps = []
    with h5py.File(args.val_h5, "r") as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        for e in range(min(len(ep_len), 200)):
            z, ae = encode_episode(model, f, int(ep_off[e]), int(ep_len[e]), 5, stats, device)
            eps.append((z, ae))
    lens = [len(z) for z, _ in eps]
    fva = load_factors(args.val_h5, args.env, lens)
    starts = torch.tensor([0] + lens[:-1]).cumsum(0).tolist()

    res = {"encode": {}, "predit": {}}
    Zva = torch.cat([z for z, _ in eps])
    for k in fva:
        res["encode"][k] = r2(predict(Zva, k), fva[k].to(device).double())
    for d in args.depths:
        P, idx = [], []
        for (z, ae), s0 in zip(eps, starts):
            T = len(z)
            if T - d <= 0:
                continue
            st = torch.arange(0, T - d, device=device)
            win = torch.stack([ae[s : s + d] for s in range(T - d)])
            P.append(rollout(model, z[st], win, 1)[:, -1])
            idx.append(s0 + st.cpu() + d)
        if not P:
            continue
        P, idx = torch.cat(P), torch.cat(idx)
        res["predit"][str(d)] = {
            k: r2(predict(P, k), fva[k][idx].to(device).double()) for k in fva
        }
    print(json.dumps(res, indent=1))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
