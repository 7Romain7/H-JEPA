"""Shared I/O for the cost-head experiments: checkpoints, frames, actions, level-1 latents."""

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (pixels are stored with an HDF5 compression plugin)
import numpy as np
import torch
import torch.nn.functional as F

from data import IMAGENET_STATS, NORMALIZER_ARTIFACT_FILENAME, load_normalizer_artifact

FRAMESKIP = 5  # Push-T level-1 frameskip: one model step = 5 env steps


def load_hjepa(ckpt_path, device='cuda'):
    hjepa = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hjepa = hjepa.to(device).eval()
    for p in hjepa.parameters():
        p.requires_grad_(False)
    return hjepa


def action_stats(ckpt_path):
    art = load_normalizer_artifact(Path(ckpt_path).parent / NORMALIZER_ARTIFACT_FILENAME)
    st = art['stats']['action']
    return (np.asarray(st['mean'], np.float32).reshape(-1),
            np.asarray(st['std'], np.float32).reshape(-1))


_MEAN = torch.tensor(IMAGENET_STATS['mean']).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STATS['std']).view(1, 3, 1, 1)


def preprocess(pixels_uint8, device, img_size=224):
    """(N, H, W, 3) uint8 -> (N, 3, img_size, img_size) float, ImageNet-normalized (as planning_eval)."""
    x = torch.as_tensor(pixels_uint8).to(device).permute(0, 3, 1, 2).float().div_(255)
    x = (x - _MEAN.to(device)) / _STD.to(device)
    if x.shape[-1] != img_size or x.shape[-2] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', antialias=True)
    return x


@torch.no_grad()
def encode_frames(jepa1, pixels_uint8, device, batch=512):
    """Level-1 embeddings of individual frames: (N, D) float32 on CPU."""
    out = []
    for i in range(0, len(pixels_uint8), batch):
        x = preprocess(pixels_uint8[i:i + batch], device)
        info = jepa1.encode({'pixels': x.unsqueeze(1)})
        out.append(info['embed_0'][:, 0].float().cpu())
    return torch.cat(out)


def read_episodes(h5_path, max_episodes=None):
    f = h5py.File(h5_path, 'r')
    ep_len = f['ep_len'][:].astype(np.int64)
    ep_off = f['ep_offset'][:].astype(np.int64)
    if max_episodes is not None:
        ep_len, ep_off = ep_len[:max_episodes], ep_off[:max_episodes]
    return f, ep_off, ep_len


def stacked_actions(actions_norm, ep_off, ep_len, i, n_steps):
    """Level-1 actions for model steps starting at raw index i (inside one episode):
    step k concatenates raw actions i+5k .. i+5k+4, as the dataset/planner stack them."""
    idx = i + np.arange(n_steps * FRAMESKIP)
    return actions_norm[idx].reshape(n_steps, FRAMESKIP * actions_norm.shape[-1])


class Latents:
    """Encoded train set of one model seed: z (N, D) at every raw frame, episode bounds."""

    def __init__(self, directory, seed):
        d = Path(directory)
        self.z = torch.from_numpy(np.load(d / f'latents_seed{seed}.npy').astype(np.float32))
        meta = np.load(d / 'meta.npz')
        self.ep_off, self.ep_len = meta['ep_offset'], meta['ep_len']
        ep_end = self.ep_off + self.ep_len
        # episode end (exclusive) of every frame
        self.frame_end = np.repeat(ep_end, self.ep_len)
        self.frame_ep_start = np.repeat(self.ep_off, self.ep_len)

    def to(self, device):
        self.z = self.z.to(device)
        self.frame_end_t = torch.as_tensor(self.frame_end, device=device)
        return self

    def sample_starts(self, n, min_future_steps, gen, device):
        """Raw indices i such that i + FRAMESKIP * min_future_steps is in the same episode."""
        need = FRAMESKIP * min_future_steps
        N = self.z.shape[0]
        out = torch.empty(0, dtype=torch.long, device=device)
        while out.numel() < n:
            i = torch.randint(0, N, (2 * n,), generator=gen, device=device)
            ok = i + need < self.frame_end_t[i]
            out = torch.cat([out, i[ok]])
        return out[:n]
