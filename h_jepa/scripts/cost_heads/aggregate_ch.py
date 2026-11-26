"""Tableau Push-T des tetes de cout, a partir des resultats rapatries du cluster.

  scp -r login@machine:~/psc/hjepa_home/results resultats_ch/
  python aggregate_ch.py resultats_ch/results  > resultats_ch/tableau_pusht.md

Colonnes : Spearman (encode), Spearman (predit), monotonicite (encode / predit), effective rank,
success rate CEM (et GD si disponible). Moyenne +- erreur standard sur les seeds de modele 42/43/44.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

BASE = ['straight_c1_s0', 'straight_c1_s0.001', 'straight_c1_s0.01', 'straight_c1_s0.1',
        'straight_c1_s1', 'sigreg_only', 'qrl', 'gcivl']
WM = ['wm_base', 'wm_h_c0.1_h', 'wm_h_c1_h', 'wm_z_c0.1', 'wm_z_c1', 'wm_h_c0.1', 'wm_h_c1', 'wm_h_c1_e2_h', 'wm_h_c1_e2']
ORDER = ['L2', 'level2'] + BASE + [f'lv2_{n}' for n in BASE] + WM
LABEL = {'L2': 'L2 (latent niveau 1)', 'level2': 'latent niveau 2 H-JEPA (d)',
         'sigreg_only': 'SIGReg seul (contrôle)', 'qrl': 'QRL (b)', 'gcivl': 'GCIVL (c)'}
LABEL.update({f'lv2_{n}': f'niveau 2 + {LABEL.get(n, n)}' for n in BASE})
LABEL.update({'wm_base': 'WM réentraîné, L2 (λ_curv 0)',
              'wm_z_c0.1': 'WM + L_curv 0,1 sur z, L2', 'wm_z_c1': 'WM + L_curv 1 sur z, L2',
              'wm_h_c0.1': 'WM + L_curv 0,1 sur h, L2 dans z', 'wm_h_c1': 'WM + L_curv 1 sur h, L2 dans z',
              'wm_h_c0.1_h': 'WM + L_curv 0,1 sur h, coût dans h', 'wm_h_c1_h': 'WM + L_curv 1 sur h, coût dans h',
              'wm_h_c1_e2_h': 'idem, 2 époques (test postwm), coût dans h',
              'wm_h_c1_e2': 'idem, 2 époques (test postwm), L2 dans z'})
PLAN_NAME = {'L2': 'l2', 'level2': 'level2'}
PLAN_NAME.update({'wm_h_c0.1_h': 'wm_h_c0.1', 'wm_h_c1_h': 'wm_h_c1', 'wm_h_c0.1': None, 'wm_h_c1': None,
                  'wm_h_c1_e2_h': 'wm_h_c1_e2', 'wm_h_c1_e2': None})


def fmt(xs, pct=False, nd=2):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return '–'
    m = np.mean(xs)
    se = np.std(xs, ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return f'{m:.1f} ± {se:.1f}' if pct else f'{m:.{nd}f} ± {se:.{nd}f}'


def main(root):
    root = Path(root)
    # one dict per model seed, merging seed<s>.json and seed<s>_<tag>.json
    by_seed = {}
    for p in sorted((root / 'diag').glob('seed*.json')):
        d = json.loads(p.read_text())
        if d['seed'] in by_seed:
            by_seed[d['seed']]['heads'].update(d['heads'])
        else:
            by_seed[d['seed']] = d
    diags = [by_seed[s] for s in sorted(by_seed)]
    names = [n for n in ORDER if any(n in d['heads'] for d in diags)]
    names += sorted({n for d in diags for n in d['heads']} - set(names))
    print(f'Push-T, {len(diags)} seeds de modèle, {diags[0]["n_segments"] if diags else 0} segments '
          f'de {diags[0]["goal_steps"] if diags else "?"} pas (but à 75 pas d\'env).\n')
    print('| coût | Spearman (encodé) | Spearman (prédit) | monotonicité (encodé / prédit) '
          '| effective rank | success rate CEM-300 | success rate iCEM-3000 | success rate GD |')
    print('|---|---|---|---|---|---|---|---|')
    for n in names:
        rs = [d['heads'][n] for d in diags if n in d['heads']]
        sp_e = fmt([r['encoded']['spearman_mean'] for r in rs])
        sp_p = fmt([r['predicted']['spearman_mean'] for r in rs])
        mo = f"{fmt([r['encoded']['monotonicity'] for r in rs])} / {fmt([r['predicted']['monotonicity'] for r in rs])}"
        er = [r.get('effective_rank') for r in rs]
        dim = rs[0].get('embed_dim') if rs else None
        er_s = fmt(er, nd=1) + (f' (/{dim})' if dim else '')
        sr = {}
        for solver in ('cem', 'icem', 'gd'):
            vals = []
            pn = PLAN_NAME.get(n, n)
            for p in ([] if pn is None else sorted((root / 'plan' / f'{solver}_{pn}').glob('seed*/metrics.yaml'))):
                vals.append(yaml.safe_load(p.read_text())['success_rate'])
            sr[solver] = fmt(vals, pct=True) + (f' ({len(vals)} seeds)' if vals and len(vals) < 3 else '')
        print(f'| {LABEL.get(n, n)} | {sp_e} | {sp_p} | {mo} | {er_s} | {sr["cem"]} | {sr["icem"]} | {sr["gd"]} |')


if __name__ == '__main__':
    main(sys.argv[1])
