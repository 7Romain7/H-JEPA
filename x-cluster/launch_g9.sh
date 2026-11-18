#!/bin/bash
# Nuit du 7 au 8/10/2026 : campagne g9 (Idée 9 avec historique, sonde le long de u_t)
# sur Pendulum (g9p) et MountainCar (g9m) ; chaque job entraîne puis évalue.
#   bash launch_g9.sh <epochs> <num_eval>
set -e
cd "$(dirname "$0")"
EP=$1; NE=$2
N=$(python -c "import argparse,cluster;print(len(cluster.free_machines(argparse.Namespace(rooms=None,hosts=None,min_vram=0,allow_users=False))))")
python make_jobs_g9.py --machines "$((N - 2))" --epochs "$EP" --num-eval "$NE" --out jobs_g9.txt
{ echo "ev_g9p_random | bash scripts/cluster/run_eval.sh pendulum random eval.num_eval=$NE"
  echo "ev_g9m_random | bash scripts/cluster/run_eval.sh mountaincar random eval.num_eval=$NE"
  tail -n +2 jobs_g9.txt; } > night_g9.txt
echo "$N GPU libres -> $(grep -c '|' night_g9.txt) jobs"
python -u cluster.py run night_g9.txt --deadline "2026-10-08 07:30" --launch-until "2026-10-07 20:00" --poll 60
