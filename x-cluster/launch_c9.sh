#!/bin/bash
# Nuit du 14 au 15/10/2026 : campagne c9, grille alpha (L_curv) x lambda (L_scal),
# loss multi-blocs, 4 époques, Pendulum (c9p) et MountainCar (c9m).
set -e
cd "$(dirname "$0")"
N=$(python -c "import argparse,cluster;print(len(cluster.free_machines(argparse.Namespace(rooms=None,hosts=None,min_vram=0,allow_users=False))))")
python make_jobs_c9.py --machines "$N" --out jobs_c9.txt
python -u cluster.py run jobs_c9.txt --deadline "2026-10-15 07:30" --launch-until "2026-10-14 19:30" --poll 60
