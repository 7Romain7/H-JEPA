#!/usr/bin/env python3
"""Figures 2 et 4 (Temporal Straightening) côte à côte pour quelques modèles c9 :
même problème départ/but et même épisode pour tous (landscape.py, --seed 0).
  python plot_landscapes_c9.py   -> resultats/c9/landscapes_<env>.png"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / 'resultats' / 'c9'
MODELS = [('base_a0', 'LeWM, α=0'), ('base_a1', 'LeWM, α=1 (méthode du papier)'),
          ('hist_a0_l0p1', 'Idée 9, α=0, λ=0,1'), ('hist_a1_l0p1', 'Idée 9, α=1, λ=0,1')]
for env, name in (('c9p', 'Pendulum'), ('c9m', 'MountainCar')):
    found = []
    for tag, label in MODELS:
        js = sorted(R.glob(f'{env}_{tag}_s*/landscape/landscape.json'))
        if js:
            found.append((label, json.loads(js[0].read_text()), js[0].parts[-3]))
    if not found:
        continue
    fig, axs = plt.subplots(3, len(found), figsize=(4.6 * len(found), 12))
    axs = np.atleast_2d(axs).reshape(3, len(found))
    for k, (label, L, job) in enumerate(found):
        c = np.array(L['landscape'])
        v = L['grid_values']
        ax = axs[0, k]
        ax.imshow((c / c.max()).T, origin='lower', cmap='hot', extent=[v[0], v[-1], v[0], v[-1]], aspect='auto')
        ax.set_title(f'{label}\n{job}', fontsize=10)
        ax.set_xlabel('action du bloc 1')
        ax.set_ylabel('action du bloc 2')
        pc = np.array(L['pca'])
        axs[1, k].plot(pc[:, 0], pc[:, 1], '-o', ms=3)
        axs[1, k].plot(pc[-1, 0], pc[-1, 1], '*', ms=16, color='gold', mec='k')
        axs[1, k].set_title(f"ACP de la trajectoire (cos moyen {L['mean_cos_consecutive']:.2f})", fontsize=10)
        axs[2, k].plot(L['mse_to_last'])
        axs[2, k].set_title('MSE à la dernière image', fontsize=10)
        axs[2, k].set_xlabel('temps (blocs)')
    fig.suptitle(f'{name} : paysage de coût (haut, sombre = bas, normalisé) et trajectoire latente', fontsize=13)
    fig.tight_layout()
    fig.savefig(R / f'landscapes_{env}.png', dpi=100)
    print(R / f'landscapes_{env}.png', [f[2] for f in found])
