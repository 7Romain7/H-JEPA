"""Tetes de cout entrainees sur les latents geles (etape 3 du protocole)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Petit MLP h applique aux latents geles. Cout : ||h(z) - h(z_goal)||^2."""

    def __init__(self, in_dim, out_dim=192, hidden=512, mean=None, std=None):
        super().__init__()
        self.cfg = {"in_dim": in_dim, "out_dim": out_dim, "hidden": hidden}
        self.register_buffer("mean", torch.zeros(in_dim) if mean is None else mean)
        self.register_buffer("std", torch.ones(in_dim) if std is None else std)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net((z - self.mean) / self.std)

    def cost(self, z, zg):
        return ((self(z) - self(zg)) ** 2).sum(-1)


class QuasiMetric(nn.Module):
    """Distance temporelle quasimetrique d(z, z_goal), de forme MRN :
    une partie asymetrique max_k relu(a_k(zg) - a_k(z)) et une partie symetrique
    ||s(z) - s(zg)||. Elle verifie l'inegalite triangulaire par construction."""

    def __init__(self, in_dim, out_dim=128, hidden=512, mean=None, std=None):
        super().__init__()
        self.cfg = {"in_dim": in_dim, "out_dim": out_dim, "hidden": hidden}
        self.register_buffer("mean", torch.zeros(in_dim) if mean is None else mean)
        self.register_buffer("std", torch.ones(in_dim) if std is None else std)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * out_dim),
        )

    def forward(self, z):
        return self.net((z - self.mean) / self.std)

    def cost(self, z, zg):
        a, s = self(z).chunk(2, dim=-1)
        ag, sg = self(zg).chunk(2, dim=-1)
        asym = F.relu(ag - a).amax(-1)
        sym = ((s - sg) ** 2).sum(-1).add(1e-8).sqrt()
        return asym + sym


class ValueHead(nn.Module):
    """Value conditionnee par le goal V(z, z_goal), deux reseaux. Cout : -V moyen."""

    def __init__(self, in_dim, hidden=512, mean=None, std=None):
        super().__init__()
        self.cfg = {"in_dim": in_dim, "hidden": hidden}
        self.register_buffer("mean", torch.zeros(in_dim) if mean is None else mean)
        self.register_buffer("std", torch.ones(in_dim) if std is None else std)

        def mlp():
            return nn.Sequential(
                nn.Linear(2 * in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )

        self.v1, self.v2 = mlp(), mlp()

    def values(self, z, zg):
        zg = zg.expand_as(z)
        x = torch.cat([(z - self.mean) / self.std, (zg - self.mean) / self.std], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)

    def cost(self, z, zg):
        v1, v2 = self.values(z, zg)
        return -(v1 + v2) / 2


class QueryHead(nn.Module):
    """Tete h(z, q) conditionnee par une query q via FiLM. Cout pour la query
    par defaut : ||h(z, q) - h(z_goal, q)||^2."""

    def __init__(self, in_dim, n_queries=3, out_dim=64, hidden=512, query=0, mean=None, std=None):
        super().__init__()
        self.cfg = {"in_dim": in_dim, "n_queries": n_queries, "out_dim": out_dim,
                    "hidden": hidden, "query": query}
        self.query = query
        self.register_buffer("mean", torch.zeros(in_dim) if mean is None else mean)
        self.register_buffer("std", torch.ones(in_dim) if std is None else std)
        self.l1, self.l2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.film = nn.Embedding(n_queries, 4 * hidden)
        nn.init.zeros_(self.film.weight)

    def forward(self, z, q):
        """z (..., D), q entier ou tenseur d'entiers de forme z.shape[:-1]."""
        if not torch.is_tensor(q):
            q = torch.full(z.shape[:-1], int(q), device=z.device, dtype=torch.long)
        g1, b1, g2, b2 = self.film(q).chunk(4, dim=-1)
        x = F.gelu(self.l1((z - self.mean) / self.std) * (1 + g1) + b1)
        x = F.gelu(self.l2(x) * (1 + g2) + b2)
        return self.out(x)

    def cost(self, z, zg):
        return ((self(z, self.query) - self(zg, self.query)) ** 2).sum(-1)


KINDS = {"Projector": Projector, "QuasiMetric": QuasiMetric,
         "ValueHead": ValueHead, "QueryHead": QueryHead}


def save_head(head, path, extra=None):
    torch.save({"kind": type(head).__name__, "cfg": head.cfg, "state": head.state_dict(), "extra": extra or {}}, path)


def load_head(path, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    head = KINDS[blob.get("kind", "Projector")](**blob["cfg"])
    head.load_state_dict(blob["state"])
    return head.to(device).eval()


def effective_rank(Z, eps=1e-7):
    """exp de l'entropie du spectre normalise de Z (N, D), comme dans RankMe."""
    s = torch.linalg.svdvals(Z.float())
    p = s / s.sum() + eps
    return float(torch.exp(-(p * p.log()).sum()))


def loss_consecutive(y):
    """1 - cos entre deux deplacements consecutifs. y : (B, L, D)."""
    v = y[:, 1:] - y[:, :-1]
    return (1 - F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1)).mean()


def loss_chord(y):
    """1 - cos entre chaque deplacement et la corde vers le dernier etat du segment."""
    v = y[:, 1:-1] - y[:, :-2]
    chord = y[:, -1:] - y[:, :-2]
    return (1 - F.cosine_similarity(v, chord, dim=-1)).mean()


STRAIGHT_LOSSES = {"consecutif": loss_consecutive, "corde": loss_chord}
