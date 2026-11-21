"""Cross-Entropy Method solver, drop-in replacement for GradientSolver.

Same interface as GradientSolver (configure(level=...), solve(planning_horizon=...,
steps_taken=..., eval_budget=...), solve_records, torch_gen), so planning_eval's chunked
evaluation and planner reseeding work unchanged. Ported from stable-worldmodel's
CEMSolver (planning/solver/cem.py), with the decreasing horizon schedule of this repo.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .solver import Costable, get_planning_horizon


class CEMSolver(torch.nn.Module):
    """Cross-Entropy Method over action sequences.

    Args:
        model: World model implementing the Costable protocol (get_cost, rollout).
        num_samples: Candidate sequences sampled per iteration.
        n_steps: CEM iterations.
        topk: Elite sequences used to refit the Gaussian.
        var_scale: Initial std of the sampling distribution (actions are z-scored).
        batch_size: Environments solved together (None: all).
        device, seed: Device and seed of the sampling generator.
    """

    def __init__(
        self,
        model: Costable,
        num_samples: int = 300,
        n_steps: int = 30,
        topk: int = 30,
        var_scale: float = 1.0,
        batch_size: int | None = None,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        momentum: float = 0.0,
        keep_elites: int = 0,
        min_std: float = 0.0,
        **unused: Any,
    ) -> None:
        super().__init__()
        self.model = model
        # momentum: mean/std <- momentum * old + (1 - momentum) * elite statistics
        # keep_elites: the best keep_elites sequences of the previous iteration are re-evaluated
        # among the candidates (iCEM, Pinneri et al. 2020)
        self.momentum = float(momentum)
        self.keep_elites = int(keep_elites)
        self.min_std = float(min_std)
        self.num_samples = int(num_samples)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.var_scale = float(var_scale)
        self.batch_size = batch_size
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self._n_envs = None
        self._action_dim = None
        self._config = None
        self.solve_records = []

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any, level: int = 1) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self.level = int(level)
        if level == 1:
            self._action_dim = int(np.prod(action_space.shape[1:]))
        else:
            self._action_dim = int(config.action_dim)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _expand(self, info_dict: dict, start: int, end: int, n: int) -> dict:
        out = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                vb = v[start:end].to(self.device)
                out[k] = vb.unsqueeze(1).expand(end - start, n, *vb.shape[1:])
            elif isinstance(v, np.ndarray):
                out[k] = np.repeat(v[start:end][:, None, ...], n, axis=1)
            else:
                out[k] = v
        return out

    @torch.inference_mode()
    def _rollout_final_actions(self, info_dict: dict, actions: torch.Tensor) -> dict:
        info = self._expand(info_dict, 0, self.n_envs, 1)
        rollout_info = self.model.rollout(info, actions.to(self.device).unsqueeze(1))
        preds = {}
        for k, v in rollout_info.items():
            if k.startswith('predicted') and torch.is_tensor(v):
                v = v.detach().cpu()
                preds[k] = v[:, 0] if v.ndim > 1 and v.shape[1] == 1 else v
        return preds

    @torch.no_grad()
    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        planning_horizon: int | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        t0 = time.time()
        cost_last_n = max(1, int(self._config.cost_last_n))
        intermediate_cost_weight = float(getattr(self._config, 'intermediate_cost_weight', 1.0))
        if planning_horizon is None:
            planning_horizon = get_planning_horizon(
                configured_horizon=self.horizon,
                replanning_interval=int(self._config.receding_horizon) * int(self._config.action_block),
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )
        H, A = int(planning_horizon), self.action_dim
        bs = self.batch_size or self.n_envs
        best_actions = []
        for start in range(0, self.n_envs, bs):
            end = min(start + bs, self.n_envs)
            cur = end - start
            mean = torch.zeros(cur, H, A, device=self.device)
            if init_action is not None:
                k = min(H, init_action.shape[1])
                mean[:, :k] = init_action[start:end, :k].to(self.device)
            std = torch.full((cur, H, A), self.var_scale, device=self.device)
            infos = self._expand(info_dict, start, end, self.num_samples)
            prev_elites = None
            for _ in range(self.n_steps):
                cand = torch.randn(cur, self.num_samples, H, A, generator=self.torch_gen,
                                   device=self.device) * std.unsqueeze(1) + mean.unsqueeze(1)
                cand[:, 0] = mean
                if prev_elites is not None and self.keep_elites > 0:
                    k = min(self.keep_elites, prev_elites.shape[1], self.num_samples - 1)
                    cand[:, 1:1 + k] = prev_elites[:, :k]
                info = dict(infos)
                info['cost_last_n'] = cost_last_n
                info['intermediate_cost_weight'] = intermediate_cost_weight
                costs = self.model.get_cost(info, cand)
                for key in ('embed_0', 'goal_embed_0'):  # encode observation and goal once
                    if key in info:
                        infos.setdefault(key, info[key])
                assert costs.shape == (cur, self.num_samples), costs.shape
                idx = torch.topk(costs, k=self.topk, dim=1, largest=False).indices
                elites = cand[torch.arange(cur, device=self.device)[:, None], idx]  # sorted by cost
                best = elites[:, 0]
                prev_elites = elites
                m = self.momentum
                mean = m * mean + (1 - m) * elites.mean(dim=1)
                std = (m * std + (1 - m) * elites.std(dim=1, correction=0)).clamp_min(self.min_std)
            # best sequence seen in the last iteration (the mean is among the candidates)
            best_actions.append(best.cpu())
        outputs = {'actions': torch.cat(best_actions, dim=0)}
        outputs['predictions'] = self._rollout_final_actions(info_dict, outputs['actions'])
        wall = time.time() - t0
        self.solve_records.append({
            'level': int(getattr(self, 'level', 1)), 'horizon': H,
            'num_samples': self.num_samples, 'n_iters': [self.n_steps] * self.n_envs,
            'wall_clock': wall, 'wall_clock_per_env': [wall / max(1, self.n_envs)] * self.n_envs,
        })
        return outputs
