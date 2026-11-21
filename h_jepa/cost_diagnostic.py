"""Offline diagnostic of planning costs (step 2), no planning.

On held-out expert trajectories (pusht_expert_val.h5), for every segment of
--goal-steps model steps (goal = last frame of the segment, 75 env steps = the planning
tasks' goal offset), and for every cost c(z, z_goal):
  - Spearman correlation between c and the true number of model steps left to the goal,
    per segment (mean, std) and pooled over all segments;
  - monotonicity: fraction of consecutive steps where c decreases along the segment;
on encoded latents z_t = E(o_t) (t = 0 .. N-1) and on latents predicted by the level-1
predictor rolled out open-loop from z_0 with the expert actions (t = 1 .. N), history 1
as in planning. The goal is always encoded. Also the effective rank (RankMe) of the space
the cost is measured in, on 10k train states.

  python cost_diagnostic.py --ckpt-dir $HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2 --seed 42 \
      --val-h5 $HJEPA_HOME/pusht_expert_val.h5 --rank-latents $HJEPA_HOME/latents/pusht_train \
      --head L2=l2 --head level2=level2 --head straight=/path/head.pt --out diag_seed42.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from cost_heads import build_head, effective_rank
from cost_heads_data import FRAMESKIP, action_stats, encode_frames, load_hjepa, read_episodes


def segments(ep_off, ep_len, n_steps, stride):
    starts = []
    for o, n in zip(ep_off, ep_len):
        last = o + n - 1 - FRAMESKIP * n_steps
        starts += list(range(o, last + 1, stride))
    return np.asarray(starts, dtype=np.int64)


@torch.no_grad()
def rollout(jepa1, z0, acts):
    """z0 (n, D), acts (n, T, A) -> predicted (n, T, D) for steps 1..T, history 1."""
    z, preds = z0[:, None], []
    for k in range(acts.shape[1]):
        a_emb = jepa1.action_embed(acts[:, k:k + 1])
        z = jepa1.predict(z, a_emb)[:, -1:]
        preds.append(z[:, 0])
    return torch.stack(preds, 1)


def metrics(cost, remaining):
    """cost (n, T), remaining (T,) decreasing -> Spearman (mean, std, pooled), monotonicity."""
    rhos = []
    for c in cost:
        if np.ptp(c) == 0:
            continue
        rhos.append(spearmanr(c, remaining).statistic)
    pooled = spearmanr(cost.reshape(-1), np.tile(remaining, len(cost))).statistic
    mono = float((np.diff(cost, axis=1) < 0).mean())
    return {'spearman_mean': float(np.nanmean(rhos)), 'spearman_std': float(np.nanstd(rhos)),
            'spearman_pooled': float(pooled), 'monotonicity': mono, 'n_segments': len(rhos)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--val-h5', required=True)
    p.add_argument('--rank-latents', default=None)
    p.add_argument('--head', action='append', required=True, help='name=spec (l2, level2, path)')
    p.add_argument('--goal-steps', type=int, default=15)
    p.add_argument('--start-stride', type=int, default=1)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = 'cuda'

    ckpt = Path(a.ckpt_dir) / f'seed{a.seed}' / f'{Path(a.ckpt_dir).name}_object.ckpt'
    hjepa = load_hjepa(ckpt, dev)
    jepa1 = hjepa.jepas[0]
    f, ep_off, ep_len = read_episodes(a.val_h5)
    z_all = encode_frames(jepa1, f['pixels'][:], dev).to(dev)
    mean, std = action_stats(ckpt)
    act_all = (f['action'][:].astype(np.float32) - mean) / std
    act_all = np.nan_to_num(act_all)

    T = a.goal_steps
    starts = segments(ep_off, ep_len, T, a.start_stride)
    frame_idx = starts[:, None] + FRAMESKIP * np.arange(T + 1)[None]          # (n, T+1)
    act_idx = starts[:, None] + np.arange(T * FRAMESKIP)[None]                 # (n, 5T)
    z_seg = z_all[torch.as_tensor(frame_idx, device=dev)]                      # (n, T+1, D)
    acts = torch.as_tensor(act_all[act_idx].reshape(len(starts), T, -1), device=dev)
    z_pred = rollout(jepa1, z_seg[:, 0], acts)                                 # steps 1..T
    goal = z_seg[:, T]
    pred_err = (z_pred - z_seg[:, 1:]).pow(2).sum(-1).mean(0).cpu().tolist()

    rank_z = None
    if a.rank_latents:
        zt = np.load(Path(a.rank_latents) / f'latents_seed{a.seed}.npy', mmap_mode='r')
        idx = np.sort(np.random.default_rng(0).choice(len(zt), 10000, replace=False))
        rank_z = torch.as_tensor(np.asarray(zt[idx], dtype=np.float32), device=dev)

    results = {'seed': a.seed, 'goal_steps': T, 'n_segments': len(starts),
               'rollout_sq_error_by_step': pred_err, 'heads': {}}
    rem_enc = np.arange(T, 0, -1)        # z_0 .. z_{T-1}: T .. 1 steps left
    rem_pred = np.arange(T - 1, -1, -1)  # zhat_1 .. zhat_T: T-1 .. 0 steps left
    for spec in a.head:
        name, path = spec.split('=', 1)
        head = build_head(path, hjepa, dev).to(dev).eval()
        with torch.no_grad():
            c_enc = head.pair_cost(z_seg[:, :T], goal[:, None]).float().cpu().numpy()
            c_pred = head.pair_cost(z_pred, goal[:, None]).float().cpu().numpy()
            r = {'encoded': metrics(c_enc, rem_enc), 'predicted': metrics(c_pred, rem_pred)}
            if rank_z is not None:
                e = head.embed(rank_z)
                r['effective_rank'] = None if e is None else effective_rank(e)
                r['embed_dim'] = None if e is None else int(e.shape[-1])
        results['heads'][name] = r
        print(name, json.dumps(r), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
