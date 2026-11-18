#!/bin/bash
# Nuit du 29 au 30/09/2026 : (A) évaluation en planification des checkpoints PushT i9,
# (B) campagne Idée 9 sur Reacher (entraînement + évaluation en fin de job).
#   bash launch_night2.sh <num_eval_pusht> <num_eval_reacher> <epochs_reacher>
# Les évaluations passent en premier (épinglées à la machine qui a le checkpoint) ;
# chaque machine libérée prend ensuite un entraînement Reacher. Plus aucun lancement
# après --launch-until (un job lancé trop tard ne finirait pas avant la deadline).
set -e
cd "$(dirname "$0")"
NE=$1; NER=$2; EP=$3
python make_eval_jobs.py --prefix i9 --env pusht --num-eval "$NE" --out eval_i9.txt
N=$(python -c "import argparse,cluster;print(len(cluster.free_machines(argparse.Namespace(rooms=None,hosts=None,min_vram=0,allow_users=False))))")
python make_jobs.py --machines "$N" --epochs "$EP" --data reacher_sub --eval "reacher,eval.num_eval=$NER" \
    --prefix r9 --keep-seeds 1 --extra wm.phase_include_first=true --out jobs_r9_all.txt
{ cat eval_i9.txt; tail -n +2 jobs_r9_all.txt | head -n "$N"; } > night2.txt
echo "$N GPU libres -> $(grep -c '^ev_' night2.txt) évaluations PushT + $(grep -c '^r9_' night2.txt) entraînements Reacher"
python cluster.py run night2.txt --deadline "2026-09-30 07:30" --launch-until "2026-09-29 20:00" --poll 60
