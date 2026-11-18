#!/bin/bash
# Lancement campagne Idée 9 (28/09/2026 19h) : 1 job par GPU libre, pas de file d'attente
# (un job lancé tard ne finirait pas son époque avant la deadline).
cd "$(dirname "$0")"
python make_jobs.py --auto --epochs 1 --prefix i9 | grep -av "quantum\|store now\|upgraded"
N=$(python -c "import argparse,cluster;print(len(cluster.free_machines(argparse.Namespace(rooms=None,hosts=None,min_vram=0,allow_users=False))))")
{ head -1 jobs.txt; tail -n +2 jobs.txt | head -n "$N"; } > jobs_i9.txt
echo "$N GPU libres -> $(($(wc -l < jobs_i9.txt)-1)) jobs"
python cluster.py run jobs_i9.txt --deadline "2026-09-29 07:30" --poll 60
