"""Jobs des tetes de cout (H-JEPA, Push-T), par phase.

  A : encodage du train set (latents des 3 seeds) + baselines de planification L2 (CEM et GD)
  B : entrainement des tetes (apres A/encode)
  C : diagnostic offline de toutes les tetes (apres B)
  D : planification CEM (+ GD) des tetes retenues : python make_jobs_ch.py D --heads straight_c1_s0.01 level2

Ecrit jobs_ch_<phase>.txt. Lancement :
  $env:PSC_REMOTE_CODE="~/psc/hjepa"; $env:PSC_REMOTE_VENV="~/psc/hjepa_env"
  $env:PSC_EXTRA_ENV="export HJEPA_HOME=~/psc/hjepa_home MUJOCO_GL=egl"
  python cluster.py run jobs_ch_A.txt --deadline "2026-11-23 07:30" --max-machines 60
"""
import argparse
from pathlib import Path

SEEDS = [42, 43, 44]
J = 'bash h_jepa/scripts/cost_heads/job.sh'
HEADS = {  # nom -> arguments de train_cost_head.py
    'straight_c1_s0': '--head straight --curv 1 --sigreg 0 --steps 30000',
    'straight_c1_s0.001': '--head straight --curv 1 --sigreg 0.001 --steps 30000',
    'straight_c1_s0.01': '--head straight --curv 1 --sigreg 0.01 --steps 30000',
    'straight_c1_s0.1': '--head straight --curv 1 --sigreg 0.1 --steps 30000',
    'straight_c1_s1': '--head straight --curv 1 --sigreg 1 --steps 30000',
    'sigreg_only': '--head straight --curv 0 --sigreg 1 --steps 30000',
    'qrl': '--head qrl --steps 100000',
    'gcivl': '--head gcivl --steps 100000',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=list('ABCD'))
    p.add_argument('--heads', nargs='*', default=[])
    p.add_argument('--gd', action='store_true', help='phase D : aussi le planificateur GD (AdamW)')
    a = p.parse_args()
    lines = []
    if a.phase == 'A':
        lines.append(f'ch_encode | {J} encode')
        for s in SEEDS:
            lines.append(f'ch_cem_l2_s{s} | {J} plan pusht_flat_cem {s} l2 cem_l2')
            lines.append(f'ch_gd_l2_s{s} | {J} plan pusht_flat {s} l2 gd_l2')
    elif a.phase == 'B':
        for name, args in HEADS.items():
            for s in SEEDS:
                lines.append(f'ch_tr_{name}_s{s} | {J} train {s} {name} {args}')
    elif a.phase == 'C':
        specs = ['L2=l2', 'level2=level2'] + [
            f'{n}=$HJEPA_HOME/heads/{n}/seed{{s}}/head.pt' for n in HEADS]
        for s in SEEDS:
            lines.append(f'ch_diag_s{s} | {J} diag {s} ' + ' '.join(x.format(s=s) for x in specs))
    else:
        for h in a.heads:
            for s in SEEDS:
                lines.append(f'ch_cem_{h}_s{s} | {J} plan pusht_flat_cem {s} {h} cem_{h}')
                if a.gd:
                    lines.append(f'ch_gd_{h}_s{s} | {J} plan pusht_flat {s} {h} gd_{h}')
    out = Path(__file__).with_name(f'jobs_ch_{a.phase}.txt')
    out.write_text('\n'.join(lines) + '\n')
    print(f'{len(lines)} jobs -> {out.name}')


if __name__ == '__main__':
    main()
