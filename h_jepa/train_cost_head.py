"""Train one cost head on the frozen level-1 latents of one H-JEPA seed.

  python train_cost_head.py --latents $HJEPA_HOME/latents/pusht_train --seed 42 \
      --head straight --curv 1 --sigreg 0.01 --out $HJEPA_HOME/heads/straight_c1_s0.01/seed42

Heads:
  straight  MLP h, loss = curv * (1 - cos(h(z_{t+1}) - h(z_t), h(z_{t+2}) - h(z_{t+1})))
                        + sigreg * SIGReg(h(z))          (curv 0 = "SIGReg seul" control)
  qrl       IQE quasimetric, QRL objective (Wang et al. 2023)
  gcivl     goal-conditioned value, sparse reward "goal reached" (OGBench GCIVL)
One step t -> t+1 is one model step (5 env steps). Logs the effective rank (RankMe) of the
head's output on 10k fixed train states every --log-every steps (metrics.jsonl).
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cost_heads import GCIVLHead, ProjHead, QRLHead, effective_rank
from cost_heads_data import FRAMESKIP, Latents
from loss import SIGReg

S = FRAMESKIP


def straight_loss(head, lat, i, a, sigreg):
    z = torch.stack([lat.z[i], lat.z[i + S], lat.z[i + 2 * S]])  # (3, B, D)
    h = head.net(z)
    v0, v1 = h[1] - h[0], h[2] - h[1]
    out = {}
    loss = h.new_zeros(())
    if a.curv > 0:
        ok = (v0.norm(dim=-1) > 1e-6) & (v1.norm(dim=-1) > 1e-6)
        cos = F.cosine_similarity(v0[ok], v1[ok], dim=-1, eps=1e-8)
        out['curv_loss'] = (1 - cos).mean()
        out['cos'] = cos.mean()
        loss = loss + a.curv * out['curv_loss']
    if a.sigreg > 0:
        out['sigreg_loss'] = sigreg(h)
        loss = loss + a.sigreg * out['sigreg_loss']
    return loss, out


def qrl_loss(head, lat, i, a, gen, lam_raw):
    j = torch.randint(0, lat.z.shape[0], i.shape, generator=gen, device=i.device)
    es, es1, eg = head.phi(lat.z[i]), head.phi(lat.z[i + S]), head.phi(lat.z[j])
    d_rand = head.dist_emb(es, eg)
    d_pos = head.dist_emb(es, es1)
    phi = -F.softplus(a.qrl_offset - d_rand, beta=a.qrl_beta)
    viol = F.relu(d_pos - 1).pow(2).mean()
    lam = F.softplus(lam_raw)
    loss = -phi.mean() + lam.detach() * viol
    lam_loss = -lam * (viol.detach() - a.qrl_eps ** 2)
    return loss + lam_loss, {'d_rand': d_rand.mean(), 'd_pos': d_pos.mean(), 'viol': viol, 'lambda': lam}


def sample_goals(lat, i, gen, a):
    """OGBench value-goal sampling: current state, future state of the trajectory, random state."""
    n = i.shape[0]
    u = torch.rand(n, generator=gen, device=i.device)
    # future state: k >= 1 model steps ahead, k ~ Geometric(1 - gamma), clipped to the episode
    geo = torch.distributions.Geometric(probs=torch.full((n,), 1 - a.gamma, device=i.device))
    k = geo.sample().long() + 1
    k_max = (lat.frame_end_t[i] - 1 - i) // S
    traj = i + S * torch.minimum(k, k_max)
    rnd = torch.randint(0, lat.z.shape[0], (n,), generator=gen, device=i.device)
    g = torch.where(u < a.p_cur, i, torch.where(u < a.p_cur + a.p_traj, traj, rnd))
    return g


def expectile(adv, diff, tau):
    w = torch.where(adv >= 0, tau, 1 - tau)
    return (w * diff.pow(2)).mean()


def gcivl_loss(head, target, lat, i, a, gen):
    g = sample_goals(lat, i, gen, a)
    success = (g == i).float()
    r, m = success - 1.0, 1.0 - success
    zs, zs1, zg = lat.z[i], lat.z[i + S], lat.z[g]
    with torch.no_grad():
        nv1, nv2 = target.values(zs1, zg)
        nv = torch.minimum(nv1, nv2)
        q = r + a.gamma * m * nv
        tv1, tv2 = target.values(zs, zg)
        adv = q - (tv1 + tv2) / 2
        q1, q2 = r + a.gamma * m * nv1, r + a.gamma * m * nv2
    v1, v2 = head.values(zs, zg)
    loss = expectile(adv, q1 - v1, a.expectile) + expectile(adv, q2 - v2, a.expectile)
    return loss, {'v': ((v1 + v2) / 2).mean(), 'success_frac': success.mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latents', required=True)
    p.add_argument('--seed', type=int, required=True, help='model seed (selects the latents)')
    p.add_argument('--head', choices=['straight', 'qrl', 'gcivl'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--log-every', type=int, default=1000)
    p.add_argument('--train-seed', type=int, default=0)
    p.add_argument('--out-dim', type=int, default=128)
    p.add_argument('--curv', type=float, default=1.0)
    p.add_argument('--sigreg', type=float, default=0.0)
    p.add_argument('--qrl-offset', type=float, default=50.0)
    p.add_argument('--qrl-beta', type=float, default=0.1)
    p.add_argument('--qrl-eps', type=float, default=0.25)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--expectile', type=float, default=0.9)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--p-cur', type=float, default=0.2)
    p.add_argument('--p-traj', type=float, default=0.5)
    a = p.parse_args()

    dev = 'cuda'
    torch.manual_seed(a.train_seed)
    gen = torch.Generator(device=dev).manual_seed(a.train_seed)
    lat = Latents(a.latents, a.seed).to(dev)
    D = lat.z.shape[1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=1))

    target, lam_raw, sigreg = None, None, None
    if a.head == 'straight':
        head = ProjHead(D, 512, a.out_dim)
        sigreg = SIGReg().to(dev)
    elif a.head == 'qrl':
        head = QRLHead(D)
        lam_raw = torch.nn.Parameter(torch.zeros((), device=dev))
    else:
        head = GCIVLHead(D)
    head = head.to(dev)
    params = list(head.parameters()) + ([lam_raw] if lam_raw is not None else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    if a.head == 'gcivl':
        target = copy.deepcopy(head).requires_grad_(False)

    rank_idx = torch.randperm(lat.z.shape[0], generator=torch.Generator().manual_seed(0))[:10000].to(dev)
    min_future = 2 if a.head == 'straight' else 1
    log = open(out / 'metrics.jsonl', 'w')
    t0 = time.time()
    for step in range(1, a.steps + 1):
        i = lat.sample_starts(a.batch, min_future, gen, dev)
        if a.head == 'straight':
            loss, info = straight_loss(head, lat, i, a, sigreg)
        elif a.head == 'qrl':
            loss, info = qrl_loss(head, lat, i, a, gen, lam_raw)
        else:
            loss, info = gcivl_loss(head, target, lat, i, a, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if target is not None:
            with torch.no_grad():
                for pt, ph in zip(target.parameters(), head.parameters()):
                    pt.lerp_(ph, a.tau)
        if step % a.log_every == 0 or step == a.steps:
            rec = {'step': step, 'loss': float(loss), **{k: float(v) for k, v in info.items()},
                   'time_min': (time.time() - t0) / 60}
            head.eval()
            with torch.no_grad():
                e = head.embed(lat.z[rank_idx])
                if e is not None:
                    rec['effective_rank'] = effective_rank(e)
                    rec['embed_dim'] = int(e.shape[-1])
            head.train()
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(rec, flush=True)
    head.eval()
    torch.save(head.cpu(), out / 'head.pt')
    print('saved', out / 'head.pt')


if __name__ == '__main__':
    main()
