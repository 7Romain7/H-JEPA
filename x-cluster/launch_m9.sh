#!/bin/bash
# Nuit du 13 au 14/10/2026 : campagne m9 (loss sur plusieurs blocs, k = 5) sur Pendulum
# (m9p) et MountainCar (m9m) ; chaque job entraîne (5 époques), évalue en planification
# (gradient et CEM, contexte 2 et 3) puis mesure l'erreur de rollout.
set -e
cd "$(dirname "$0")"
N=$(python -c "import argparse,cluster;print(len(cluster.free_machines(argparse.Namespace(rooms=None,hosts=None,min_vram=0,allow_users=False))))")
python make_jobs_m9.py --machines "$N" --epochs 5 --out jobs_m9.txt
python -u cluster.py run jobs_m9.txt --deadline "2026-10-14 07:30" --launch-until "2026-10-13 19:30" --poll 60
