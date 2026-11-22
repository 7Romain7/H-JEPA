"""Facteurs lus dans l'etat simulateur, alignes sur les latents du cache."""
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

QUERY_NAMES = {"pusht": ["bloc", "agent", "tout"], "cube": ["cube", "bras", "tout"]}


def _rows(f, lens, frameskip):
    off = f["ep_offset"][:]
    return np.concatenate([off[e] + np.arange(T) * frameskip for e, T in enumerate(lens)])


def load_factors(h5_path, env, lens, frameskip=5):
    """Rend {"objet": (N, d1), "agent": (N, d2)} non normalises, N = somme des lens."""
    with h5py.File(h5_path, "r") as f:
        rows = _rows(f, lens, frameskip)

        def col(name):
            return torch.from_numpy(f[name][:][rows]).float()

        if env == "pusht":
            st = col("state")  # agent xy, bloc xy, angle du bloc, vitesse agent
            objet = torch.cat([st[:, 2:4], torch.cos(st[:, 4:5]), torch.sin(st[:, 4:5])], dim=1)
            agent = st[:, 0:2]
        elif env == "cube":
            yaw = col("privileged_block_0_yaw")
            objet = torch.cat([col("privileged_block_0_pos"), torch.cos(yaw), torch.sin(yaw)], dim=1)
            agent = torch.cat([col("proprio_effector_pos"), col("proprio_gripper_opening")], dim=1)
        else:
            raise ValueError(env)
    return {"objet": objet, "agent": agent}


def standardize(fac, stats=None):
    """Normalise les colonnes de position ; les cos et sin de l'objet restent tels quels."""
    if stats is None:
        stats = {k: (v.mean(0), v.std(0) + 1e-6) for k, v in fac.items()}
    out = {}
    for k, v in fac.items():
        m, s = stats[k]
        w = (v - m) / s
        if k == "objet":
            w = torch.cat([w[:, :-2], v[:, -2:]], dim=1)
        out[k] = w
    return out, stats
