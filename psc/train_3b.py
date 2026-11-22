"""Tete 3b : distance temporelle quasimetrique apprise par l'objectif de QRL
(Wang et al. 2023) sur latents geles. On maximise la distance entre etats tires
au hasard sous la contrainte d(z_t, z_{t+1}) <= 1 sur les transitions observees.

A lancer depuis h_jepa/ :
  python ../psc/train_3b.py --cache cache_pusht_train_seed42.pt --name qrl --out-dir heads_pusht_seed42
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).parent))

from heads import QuasiMetric, save_head  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eps", type=float, default=0.25)
    p.add_argument("--offset", type=float, default=15.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = [e["z"].float() for e in torch.load(args.cache, weights_only=False)]
    cur = torch.cat([z[:-1] for z in episodes]).to(device)
    nxt = torch.cat([z[1:] for z in episodes]).to(device)
    print(f"{len(episodes)} episodes, {len(cur)} transitions")

    head = QuasiMetric(cur.shape[-1], mean=cur.mean(0), std=cur.std(0) + 1e-6).to(device)
    raw_lam = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_lam = torch.optim.Adam([raw_lam], lr=1e-2)

    hist = []
    for step in range(args.steps + 1):
        i = torch.randint(len(cur), (args.batch,), device=device)
        j = torch.randint(len(cur), (args.batch,), device=device)
        d_rand = head.cost(cur[i], cur[j])
        d_step = head.cost(cur[i], nxt[i])
        neg = F.softplus(args.offset - d_rand, beta=args.beta).mean()
        viol = F.relu(d_step - 1.0).square().mean() - args.eps**2
        lam = F.softplus(raw_lam)
        loss = neg + lam.detach() * viol
        opt.zero_grad()
        loss.backward()
        opt.step()
        opt_lam.zero_grad()
        (-F.softplus(raw_lam) * viol.detach()).backward()
        opt_lam.step()
        if step % 2000 == 0:
            rec = {"step": step, "d_aleatoire": float(d_rand.mean()), "d_transition": float(d_step.mean()),
                   "d_transition_max": float(d_step.max()), "lambda": float(lam)}
            hist.append(rec)
            print(json.dumps(rec))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_head(head, out / f"{args.name}.pt", extra={"args": vars(args), "hist": hist})
    (out / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "hist": hist}, indent=1))
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
