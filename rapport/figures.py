"""Figures du rapport (données : x-cluster/resultats). python figures.py -> fig_*.pdf"""
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE.parent / 'x-cluster' / 'resultats'
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})
C = {'base': '#D9822B', 'hist': '#2F6FB0', 'hist_u0p1': '#8FB3D9', 'phase': '#9AA3B2'}
LAB = {'base': 'LeWM', 'hist': 'LeWM-Phase (mémoire 3)', 'hist_u0p1': 'LeWM-Phase + $L_{scal}$ ($\\lambda=0{,}1$)',
       'phase': 'LeWM-Phase (mémoire 2)'}


def sem(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


# ---------------------------------------------------------------- Fig. 1 : i9 / r9
def summary_success(name):
    rows = []
    txt = (R / name / 'summary.md').read_text(encoding='utf-8').split('## Taux de succès')[1]
    for line in txt.splitlines():
        m = re.match(r'\| (\w+) \| (\d+) \| ([\d.]+) ± ([\d.]+) \|', line)
        if m:
            rows.append((m.group(1), float(m.group(3)), float(m.group(4))))
    return rows


fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
for ax, (name, title, rnd) in zip(axs, [('i9', 'PushT (1 époque)', 3.0), ('r9', 'Reacher (5 époques)', None)]):
    rows = summary_success(name)
    x = np.arange(len(rows))
    col = [C['base'] if r[0] == 'base' else C['hist'] if r[0] == 'phase' else C['hist_u0p1'] for r in rows]
    ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], color=col, capsize=2)
    lab = {'base': 'LeWM', 'phase': 'LeWM-Phase'}
    ax.set_xticks(x, [lab.get(r[0], r[0].replace('0p', '0,').replace('pdf', 'pdf ').replace('inv', 'inv '))
                      for r in rows], rotation=45, ha='right')
    if rnd:
        ax.axhline(rnd, ls='--', c='k', lw=0.8)
        ax.text(len(rows) - 0.45, rnd, 'hasard', ha='left', va='center', fontsize=8)
    ax.set_title(title)
axs[0].set_ylabel('succès (%)')
fig.tight_layout()
fig.savefig(HERE / 'fig_i9_r9.pdf')

# ---------------------------------------------------------------- Fig. 2 : dérive (g9 vs m9)
ro = pd.read_csv(R / 'rollout_g9' / 'rollout.csv')
m9 = defaultdict(list)
for d in glob.glob(str(R / 'm9' / 'm9*')):
    m = re.search(r'm9([pm])_(.*)_s(\d+)$', d)
    f = Path(d) / 'rollout.json'
    if m and f.exists():
        m9[(m.group(1), m.group(2))].append(json.loads(f.read_text())['nmse_model'])

fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2))
for ax, (e, title) in zip(axs, [('p', 'Pendulum'), ('m', 'MountainCar')]):
    for arm in ('base', 'hist', 'hist_u0p1'):
        d = ro[(ro.env == f'g9{e}') & (ro.arm == arm)].groupby('k').nmse.mean()
        ax.plot(d.index, d.values, '-o', ms=3, c=C[arm], label=LAB[arm] + ' — loss 1 bloc')
        if m9[(e, arm)]:
            v = np.mean(m9[(e, arm)], 0)
            ax.plot(range(1, 6), v, '--s', ms=3, c=C[arm], alpha=0.9, label=LAB[arm] + ' — loss multi-blocs')
    ax.set_title(title)
    ax.set_xlabel('nombre de blocs prédits à la suite')
    ax.set_xticks(range(1, 6))
axs[0].set_ylabel('erreur normalisée')
h, l = axs[1].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.2, 1, 1))
fig.legend(h, l, loc='lower center', ncol=3, fontsize=7, frameon=False)
fig.savefig(HERE / 'fig_derive.pdf')
print('ok', sorted((k, len(v)) for k, v in m9.items()))
