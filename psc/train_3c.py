"""Tete 3c : value supervisee GCIVL (IQL sur V, Kostrikov et al. 2021, variante
goal-conditioned d'OGBench) sur latents geles. Reward sparse : 0 quand l'etat
est le goal, -1 sinon. Les goals sont relabellises dans les trajectoires.

A lancer depuis h_jepa/ :
  python ../psc/train_3c.py --cache cache_pusht_train_seed42.pt --name gcivl --out-dir heads_pusht_seed42
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).parent))

from heads import ValueHead, save_head  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--p-cur", type=float, default=0.2)
    p.add_argument("--p-traj", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = [e["z"].float() for e in torch.load(args.cache, weights_only=False)]
    Z = torch.cat(episodes).to(device)
    lens = torch.tensor([len(z) for z in episodes])
    ends = torch.repeat_interleave(torch.cumsum(lens, 0) - 1, lens).to(device)  # dernier indice de l'episode
    idx_all = torch.arange(len(Z), device=device)
    valid = idx_all[idx_all < ends]  # etats qui ont un successeur
    print(f"{len(episodes)} episodes, {len(valid)} transitions")

    head = ValueHead(Z.shape[-1], mean=Z.mean(0), std=Z.std(0) + 1e-6).to(device)
    target = copy.deepcopy(head).requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    hist = []
    for step in range(args.steps + 1):
        i = valid[torch.randint(len(valid), (args.batch,), device=device)]
        u = torch.rand(args.batch, device=device)
        fut = i + 1 + (torch.rand(args.batch, device=device) * (ends[i] - i).float()).long()
        fut = torch.minimum(fut, ends[i])
        rnd = torch.randint(len(Z), (args.batch,), device=device)
        g = torch.where(u < args.p_cur, i, torch.where(u < args.p_cur + args.p_traj, fut, rnd))
        success = (g == i).float()
        reward, mask = success - 1.0, 1.0 - success
        s, s2, zg = Z[i], Z[i + 1], Z[g]
        with torch.no_grad():
            n1, n2 = target.values(s2, zg)
            q = reward + args.gamma * mask * torch.minimum(n1, n2)
            t1, t2 = target.values(s, zg)
            adv = q - (t1 + t2) / 2
        w = torch.where(adv >= 0, args.expectile, 1 - args.expectile)
        v1, v2 = head.values(s, zg)
        q1 = reward + args.gamma * mask * n1
        q2 = reward + args.gamma * mask * n2
        loss = (w * (q1 - v1) ** 2).mean() + (w * (q2 - v2) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pt, ph in zip(target.parameters(), head.parameters()):
                pt.lerp_(ph, args.tau)
        if step % 3000 == 0:
            rec = {"step": step, "loss": float(loss), "v_moyen": float(v1.mean()),
                   "v_goal_egal_etat": float(v1[success > 0].mean()) if success.sum() > 0 else None}
            hist.append(rec)
            print(json.dumps(rec))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_head(head, out / f"{args.name}.pt", extra={"args": vars(args), "hist": hist})
    (out / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "hist": hist}, indent=1))
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
