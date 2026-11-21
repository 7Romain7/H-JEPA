"""Compare the planning cost reached by CEM, GD and the expert actions on val segments.

Same cost (JEPA.get_cost, level-1 L2, cost_last_n 2), same horizon (15 model steps), start
and goal taken from pusht_expert_val. If CEM reaches a lower predicted cost than GD but plans
worse in the env, it exploits the model; if its cost is higher, it optimizes worse.

  python debug_solvers.py --ckpt $HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt \
      --val-h5 $HJEPA_HOME/pusht_expert_val.h5
"""
import argparse
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cost_diagnostic import segments
from cost_heads_data import FRAMESKIP, action_stats, preprocess, read_episodes
from stable_worldmodel.solver import CEMSolver, GradientSolver


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--val-h5', required=True)
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--horizon', type=int, default=15)
    a = p.parse_args()
    dev = 'cuda'
    hj = torch.load(a.ckpt, map_location='cpu', weights_only=False).to(dev).eval()
    jepa = hj.jepas[0]
    for q in jepa.parameters():
        q.requires_grad_(False)
    f, off, ln = read_episodes(a.val_h5)
    T = a.horizon
    starts = segments(off, ln, T, 7)[:a.n]
    mean, std = action_stats(a.ckpt)
    act = np.nan_to_num((f['action'][:] - mean) / std).astype(np.float32)
    cfg = SimpleNamespace(horizon=T, action_block=FRAMESKIP, receding_horizon=2, cost_last_n=2,
                          intermediate_cost_weight=1.0)
    space = gym.spaces.Box(-np.inf, np.inf, shape=(1, 2))
    solvers = {
        'gd': GradientSolver(jepa, n_steps=90, num_samples=800, var_scale=1.0, device=dev, seed=0,
                             optimizer_cls=torch.optim.AdamW, optimizer_kwargs={'lr': 0.05}),
        'cem300': CEMSolver(jepa, num_samples=300, n_steps=30, topk=30, device=dev, seed=0),
        'cem1000': CEMSolver(jepa, num_samples=1000, n_steps=30, topk=100, device=dev, seed=0),
        'cem1000_m.5_k10': CEMSolver(jepa, num_samples=1000, n_steps=30, topk=100, momentum=0.5,
                                     keep_elites=10, device=dev, seed=0),
        'cem1000_it90_m.5_k10': CEMSolver(jepa, num_samples=1000, n_steps=90, topk=50, momentum=0.5,
                                          keep_elites=10, device=dev, seed=0),
        'cem3000_m.3_k30': CEMSolver(jepa, num_samples=3000, n_steps=30, topk=300, momentum=0.3,
                                     keep_elites=30, device=dev, seed=0),
        'cem3000_it60_m.5_k30': CEMSolver(jepa, num_samples=3000, n_steps=60, topk=100, momentum=0.5,
                                          keep_elites=30, device=dev, seed=0),
    }
    for s in solvers.values():
        s.configure(action_space=space, n_envs=1, config=cfg)
    res = {k: [] for k in ['expert', 'zero', *solvers]}
    norms = {k: [] for k in res}

    def cost_of(info_base, actions):
        info = {k: (v.unsqueeze(1) if torch.is_tensor(v) else v) for k, v in info_base.items()}
        info['cost_last_n'] = 2
        with torch.no_grad():
            return float(jepa.get_cost(info, actions.to(dev)[:, None])[0, 0])

    for st in starts:
        pix = preprocess(f['pixels'][[st]], dev)[None]                  # (1, 1, C, H, W)
        goal = preprocess(f['pixels'][[st + FRAMESKIP * T]], dev)[None]
        info = {'pixels': pix, 'goal': goal, 'action': torch.zeros(1, 1, 2 * FRAMESKIP, device=dev)}
        expert = torch.as_tensor(act[st:st + FRAMESKIP * T].reshape(1, T, -1))
        for name, sol in solvers.items():
            with torch.enable_grad():
                out = sol.solve(dict(info), planning_horizon=T)
            res[name].append(cost_of(info, out['actions']))
            norms[name].append(float(out['actions'].abs().mean()))
        res['expert'].append(cost_of(info, expert))
        norms['expert'].append(float(expert.abs().mean()))
        res['zero'].append(cost_of(info, torch.zeros_like(expert)))
        norms['zero'].append(0.0)
    for k in res:
        print(f'{k:14s} cost median {np.median(res[k]):8.2f} mean {np.mean(res[k]):8.2f} '
              f'| mean |action| {np.mean(norms[k]):.3f}')


if __name__ == '__main__':
    main()
