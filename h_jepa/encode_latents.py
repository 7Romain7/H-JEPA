"""Encode a Push-T HDF5 set into the frozen level-1 latent of several H-JEPA checkpoints.

Reads the pixels once and runs every checkpoint's level-1 encoder on each chunk.
Writes <out>/latents_seed<s>.npy (float16, one row per raw frame) and <out>/meta.npz
(episode offsets/lengths relative to the kept episodes, raw actions).

  python encode_latents.py --h5 /tmp/pusht_expert_train.h5 --out $HJEPA_HOME/latents/pusht_train \
      --ckpt-dir $HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2 --seeds 42 43 44 --max-episodes 6000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cost_heads_data import encode_frames, load_hjepa, read_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--max-episodes', type=int, default=None)
    p.add_argument('--chunk', type=int, default=8192)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f, ep_off, ep_len = read_episodes(a.h5, a.max_episodes)
    start, stop = int(ep_off[0]), int(ep_off[-1] + ep_len[-1])
    assert np.all(ep_off[1:] == ep_off[:-1] + ep_len[:-1]), 'episodes are not contiguous'
    n = stop - start
    print(f'{len(ep_len)} episodes, {n} frames', flush=True)

    models = {}
    for s in a.seeds:
        ckpt = Path(a.ckpt_dir) / f'seed{s}' / f'{Path(a.ckpt_dir).name}_object.ckpt'
        models[s] = load_hjepa(ckpt).jepas[0]
    D = None
    bufs = {}
    t0 = time.time()
    for i in range(start, stop, a.chunk):
        pix = f['pixels'][i:min(i + a.chunk, stop)]
        for s, m in models.items():
            z = encode_frames(m, pix, 'cuda').numpy().astype(np.float16)
            if D is None:
                D = z.shape[1]
            if s not in bufs:
                bufs[s] = np.lib.format.open_memmap(out / f'latents_seed{s}.npy.tmp', mode='w+',
                                                    dtype=np.float16, shape=(n, D))
            bufs[s][i - start:i - start + len(z)] = z
        done = min(i + a.chunk, stop) - start
        el = time.time() - t0
        print(f'{done}/{n} frames, {el / 60:.1f} min, ETA {el / done * (n - done) / 60:.1f} min', flush=True)
    for s, b in bufs.items():
        b.flush()
        del b
        (out / f'latents_seed{s}.npy.tmp').rename(out / f'latents_seed{s}.npy')
    np.savez(out / 'meta.npz', ep_offset=ep_off - start, ep_len=ep_len,
             action=f['action'][start:stop].astype(np.float32))
    print('done', flush=True)


if __name__ == '__main__':
    main()
