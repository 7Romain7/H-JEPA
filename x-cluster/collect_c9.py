#!/usr/bin/env python3
"""Agrège la campagne c9 (grille alpha = poids de L_curv x lambda = poids de L_scal) et
trace les heatmaps alpha x lambda. Données : resultats/c9/<job>/ (rapatriées du cluster).

  python collect_c9.py      -> resultats/c9/c9.csv, c9_summary.md, heatmap_<env>.png"""
import glob, json, math, os, re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / 'resultats' / 'c9'
rows = []
for d in sorted(list(R.glob('c9*')) + list(R.glob('d9*'))):   # c9 + d9 (meme protocole)
    m = re.fullmatch(r'(?:c9|d9)([pm])_(base|hist)_a([0-9p]+)(?:_l([0-9p]+))?_s(\d+)', d.name)
    if not m:
        continue
    f = lambda x: float(x.replace('p', '.')) if x else np.nan
    r = dict(env='c9' + m.group(1), model=m.group(2), alpha=f(m.group(3)), lam=f(m.group(4)), seed=int(m.group(5)))
    if (d / 'metrics.jsonl').exists():
        L = [json.loads(l) for l in open(d / 'metrics.jsonl')]
        va = [x for x in L if x['stage'] != 'fit' and x['n'] > 5]
        if va:
            r.update(nmse1=va[-1].get('pred_last_nmse'), curv=va[-1].get('curv_loss'), ep=va[-1]['epoch'])
    for t in ('gd_h2', 'cem_h2'):
        if (d / f'eval_{t}.json').exists():
            r[t] = json.loads((d / f'eval_{t}.json').read_text())['success_rate']
    if (d / 'rollout.json').exists():
        r['k5'] = json.loads((d / 'rollout.json').read_text())['nmse_model'][4]
    lj = d / 'landscape' / 'landscape.json'
    if lj.exists():
        L = json.loads(lj.read_text())
        c = np.array(L['landscape'])
        r.update(cos_traj=L['mean_cos_consecutive'], land_min=c.min(), land_med=float(np.median(c)))
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(R / 'c9.csv', index=False)
# seuls les modeles qui ont fini leurs epoques (les autres ont ete coupes en cours de route)
n_all = len(df)
df = df[df.ep == df.ep.max()].copy()
print(f'{len(df)} modeles complets sur {n_all}')

ENV = {'c9p': 'Pendulum', 'c9m': 'MountainCar'}
cols = ['gd_h2', 'cem_h2', 'nmse1', 'k5', 'curv', 'cos_traj']
lines = ['# Campagne c9 : alpha (L_curv) x lambda (L_scal), loss multi-blocs, 4 époques', '',
         f"Époque de la dernière validation : {df.ep.value_counts().to_dict()}", '']
for env in ENV:
    e = df[df.env == env]
    g = e.groupby(['model', 'alpha', 'lam'], dropna=False)
    t = g[cols].mean()
    t['n'] = g['gd_h2'].count()
    lines += [f'## {ENV[env]}', '', t.round(4).to_markdown(), '']
(R / 'c9_summary.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A = sorted(df.alpha.unique())
Lm = sorted(df.lam.dropna().unique())
for env in ENV:
    e = df[df.env == env]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (col, title, cmap) in zip(axs, [('gd_h2', 'succès gradient (%)', 'viridis'),
                                            ('cem_h2', 'succès CEM (%)', 'viridis'),
                                            ('k5', 'erreur à 5 blocs', 'viridis_r')]):
        # lignes = alpha ; colonnes = LeWM puis Idée 9 lambda = 0, 0.1, 1
        M = np.full((len(A), 1 + len(Lm)), np.nan)
        N = np.zeros_like(M, dtype=int)
        for i, a in enumerate(A):
            v = e[(e.model == 'base') & (e.alpha == a)][col].dropna()
            M[i, 0], N[i, 0] = (v.mean() if len(v) else np.nan), len(v)
            for k, l in enumerate(Lm):
                v = e[(e.model == 'hist') & (e.alpha == a) & (e.lam == l)][col].dropna()
                M[i, 1 + k], N[i, 1 + k] = (v.mean() if len(v) else np.nan), len(v)
        im = ax.imshow(M, cmap=cmap, aspect='auto')
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                if not np.isnan(M[i, k]):
                    txt = f'{M[i, k]:.0f}' if col != 'k5' else f'{M[i, k]:.3f}'
                    ax.text(k, i, f'{txt}\n(n={N[i, k]})', ha='center', va='center', fontsize=10,
                            color='white' if im.norm(M[i, k]) < 0.5 else 'black')
        ax.set_xticks(range(M.shape[1]), ['LeWM'] + [f'Idée 9\nλ={l:g}' for l in Lm])
        ax.set_yticks(range(len(A)), [f'α={a:g}' for a in A])
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f'{ENV[env]} — α = poids de L_curv, λ = poids de L_scal (loss multi-blocs, 4 époques)')
    fig.tight_layout()
    fig.savefig(R / f'heatmap_{env}.png', dpi=110)
print('figures :', [str(R / f'heatmap_{e}.png') for e in ENV])
