"""Tete 3a : projector de straightening sur latents geles, avec SIGReg et suivi
de l'effective rank.

A lancer depuis h_jepa/ :
  python ../psc/train_3a.py --cache cache_pusht_train_seed42.pt --name consecutif_sig0.1 \
      --straight consecutif --w-straight 1 --w-sigreg 0.1 --out-dir heads_pusht_seed42
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).parent))

from heads import STRAIGHT_LOSSES, Projector, effective_rank, save_head  # noqa: E402
from loss import SIGReg  # noqa: E402  (code du depot H-JEPA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--straight", choices=list(STRAIGHT_LOSSES), default="consecutif")
    p.add_argument("--w-straight", type=float, default=1.0)
    p.add_argument("--w-sigreg", type=float, default=0.0)
    p.add_argument("--seg-len", type=int, default=8)
    p.add_argument("--out-dim", type=int, default=192)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = [e["z"].float() for e in torch.load(args.cache, weights_only=False)]
    L = args.seg_len
    segs = torch.stack([z[s : s + L] for z in episodes for s in range(len(z) - L + 1)]).to(device)
    allz = torch.cat(episodes).to(device)
    probe = allz[torch.randperm(len(allz), device=device)[:10000]]
    print(f"{len(episodes)} episodes, {len(allz)} latents, {len(segs)} segments de longueur {L}")

    head = Projector(allz.shape[-1], args.out_dim, mean=allz.mean(0), std=allz.std(0) + 1e-6).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sigreg = SIGReg().to(device)
    straight = STRAIGHT_LOSSES[args.straight]

    hist = []
    for step in range(args.steps + 1):
        if step % 250 == 0:
            with torch.no_grad():
                head.eval()
                y = head(segs[torch.randint(len(segs), (2048,), device=device)])
                rec = {
                    "step": step,
                    "erank": effective_rank(head(probe)),
                    "consecutif": float(STRAIGHT_LOSSES["consecutif"](y)),
                    "corde": float(STRAIGHT_LOSSES["corde"](y)),
                    "sigreg": float(sigreg(y.transpose(0, 1))),
                }
                head.train()
            hist.append(rec)
            print(json.dumps(rec))
        if step == args.steps:
            break
        y = head(segs[torch.randint(len(segs), (args.batch,), device=device)])
        loss = y.new_zeros(())
        if args.w_straight > 0:
            loss = loss + args.w_straight * straight(y)
        if args.w_sigreg > 0:
            loss = loss + args.w_sigreg * sigreg(y.transpose(0, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_head(head, out / f"{args.name}.pt", extra={"args": vars(args), "hist": hist})
    (out / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "hist": hist}, indent=1))
    print("erank_latents_bruts", effective_rank(probe))
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
