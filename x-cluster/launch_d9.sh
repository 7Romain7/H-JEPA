#!/bin/bash
# Nuit du 15 au 16/10/2026 : campagne d9, alpha (L_curv) in {1e-3, 1e-2} x (LeWM ; Idée 9
# lambda in {0, 0.1}), loss multi-blocs, 4 époques ; 60 machines max (GPU 24 Go d'abord).
set -e
cd "$(dirname "$0")"
python make_jobs_c9.py --machines 60 --alphas 0.001 0.01 --lambdas 0 0.1 --prefix d9 --out jobs_d9.txt
python -u cluster.py run jobs_d9.txt --deadline "2026-10-16 07:30" --launch-until "2026-10-15 22:00" --poll 60
