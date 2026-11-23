"""Planning cost heads on top of a frozen level-1 H-JEPA latent.

Every head defines pair_cost(z, g) -> cost, broadcast over leading dims, where z is a
(predicted or encoded) level-1 embedding and g the encoded goal embedding. The planner
uses it in place of JEPA.criterion (see install_cost_head), so all heads share the same
rollout, solver and protocol; only the cost changes.

Heads:
  L2Head        ||z - g||^2 in the level-1 latent (the native H-JEPA cost).
  ProjHead      ||h(z) - h(g)||^2, h an MLP trained with 1 - cos (temporal straightening,
                Wang et al. 2026) and/or SIGReg on its output.
  Level2Head    ||E2(z) - E2(g)||^2, E2 the frozen level-2 encoder of the same H-JEPA
                (pointwise on Push-T: window_size 1, no proprio): the l2_project baseline.
  QRLHead       IQE quasimetric d(phi(z), phi(g)) trained with QRL (Wang et al. 2023).
  GCIVLHead     -V(z, g), V trained by goal-conditioned implicit value learning with the
                sparse reward "goal reached" (OGBench GCIVL).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, hidden, out_dim, depth=2, ln=True):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden)] + ([nn.LayerNorm(hidden)] if ln else []) + [nn.GELU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def effective_rank(z: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe (Garrido et al. 2023): exp of the entropy of the normalized singular values."""
    s = torch.linalg.svdvals(z.float())
    p = s / s.sum() + eps
    return float(torch.exp(-(p * p.log()).sum()))


class CostHead(nn.Module):
    name = 'head'

    def embed(self, z):
        """Space in which the cost is measured (for the effective rank); None if no such space."""
        return z

    def pair_cost(self, z, g):
        raise NotImplementedError


class L2Head(CostHead):
    name = 'l2'

    def pair_cost(self, z, g):
        return (z - g).pow(2).sum(-1)


class ProjHead(CostHead):
    name = 'proj'

    def __init__(self, in_dim=192, hidden=512, out_dim=128):
        super().__init__()
        self.net = mlp(in_dim, hidden, out_dim)

    def embed(self, z):
        return self.net(z)

    def pair_cost(self, z, g):
        return (self.net(z) - self.net(g)).pow(2).sum(-1)


class Level2Head(CostHead):
    name = 'level2'

    def __init__(self, jepa2):
        super().__init__()
        assert jepa2.temporal_window_size == 1, 'pointwise level-2 projection needs window_size 1'
        self.jepa2 = jepa2
        for p in self.jepa2.parameters():
            p.requires_grad_(False)

    def embed(self, z):
        shape = z.shape
        out = self.jepa2.encode({'embed_1': z.reshape(-1, 1, shape[-1])}, key='embed_1',
                                chunk_temporal_inputs=True, temporal_stride=1)['embed_0']
        return out[:, 0].reshape(*shape[:-1], out.shape[-1])

    def pair_cost(self, z, g):
        return (self.embed(z) - self.embed(g)).pow(2).sum(-1)


def iqe(x, y, n_comp):
    """Interval Quasimetric Embedding distance d(x -> y) (Wang & Isola 2022).

    x, y: (..., n_comp * m). For each component, the length of the union of the intervals
    [x_j, max(x_j, y_j)], j = 1..m; returns (..., n_comp) component distances.
    """
    x = x.unflatten(-1, (n_comp, -1))
    y = y.unflatten(-1, (n_comp, -1))
    a, b = torch.broadcast_tensors(x, torch.maximum(x, y))
    a_sorted, order = a.sort(dim=-1)
    b_sorted = b.gather(-1, order)
    run_max = b_sorted.cummax(dim=-1).values
    prev_max = torch.cat([torch.full_like(run_max[..., :1], -math.inf), run_max[..., :-1]], dim=-1)
    return F.relu(b_sorted - torch.maximum(a_sorted, prev_max)).sum(-1)


class QRLHead(CostHead):
    name = 'qrl'

    def __init__(self, in_dim=192, hidden=512, n_comp=64, comp_dim=8):
        super().__init__()
        self.n_comp = n_comp
        self.phi = mlp(in_dim, hidden, n_comp * comp_dim)
        self.alpha_raw = nn.Parameter(torch.zeros(()))  # maxmean mixing weight

    def embed(self, z):
        return self.phi(z)

    def dist_emb(self, ez, eg):
        d = iqe(ez, eg, self.n_comp)
        alpha = torch.sigmoid(self.alpha_raw)
        return alpha * d.max(-1).values + (1 - alpha) * d.mean(-1)

    def pair_cost(self, z, g):
        return self.dist_emb(self.phi(z), self.phi(g))


class GCIVLHead(CostHead):
    name = 'gcivl'

    def __init__(self, in_dim=192, hidden=512):
        super().__init__()
        self.v1 = mlp(2 * in_dim, hidden, 1, depth=3)
        self.v2 = mlp(2 * in_dim, hidden, 1, depth=3)

    def values(self, z, g):
        z, g = torch.broadcast_tensors(z, g)
        x = torch.cat([z, g], -1)
        return self.v1(x)[..., 0], self.v2(x)[..., 0]

    def embed(self, z):
        return None

    def pair_cost(self, z, g):
        v1, v2 = self.values(z, g)
        return -(v1 + v2) / 2


def install_cost_head(jepa, head: CostHead):
    """Replace jepa.criterion by the head's cost, with the same window / weighting rules."""
    jepa.cost_head = head

    def criterion(info_dict):
        pred = info_dict['predicted_embed_0'][:, :, 1:, :]
        window = min(max(1, int(info_dict.get('cost_last_n', 1))), pred.shape[-2])
        pred = pred[:, :, -window:, :]
        goal = info_dict['goal_embed_0'][..., -1:, :].detach().expand_as(pred)
        cost = jepa.cost_head.pair_cost(pred, goal)  # (B, S, W)
        w_int = float(info_dict.get('intermediate_cost_weight', 1.0))
        if w_int != 1.0 and cost.shape[-1] > 1:
            weights = cost.new_ones(cost.shape[-1])
            weights[:-1] = w_int
            cost = cost * weights
        return cost.sum(-1)

    jepa.criterion = criterion
    return jepa


class OnLevel2(CostHead):
    """A head trained on level-2 latents: cost = inner(E2(z), E2(g)), E2 the frozen level-2 encoder."""
    name = 'on_level2'

    def __init__(self, level2: Level2Head, inner: CostHead):
        super().__init__()
        self.level2 = level2
        self.inner = inner

    def embed(self, z):
        return self.inner.embed(self.level2.embed(z))

    def pair_cost(self, z, g):
        return self.inner.pair_cost(self.level2.embed(z), self.level2.embed(g))


def build_head(spec: str, hjepa=None, device='cpu'):
    """spec: 'l2', 'level2' (needs the full HJEPA), or a path to a saved head (torch.save).
    A saved head with on_level2=True was trained on level-2 latents and is wrapped in OnLevel2
    (needs the full HJEPA)."""
    if spec == 'l2':
        return L2Head()
    if spec == 'level2':
        return Level2Head(hjepa.jepas[1])
    if spec == 'curv_head':  # projection trained jointly with the world model (level1.curv.head)
        return hjepa.jepas[0].curv_head
    head = torch.load(spec, map_location=device, weights_only=False)
    if getattr(head, 'on_level2', False):
        assert hjepa is not None, 'a level-2 head needs the full HJEPA'
        head = OnLevel2(Level2Head(hjepa.jepas[1]), head)
    return head
