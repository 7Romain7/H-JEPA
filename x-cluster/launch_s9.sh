#!/bin/bash
# Dimanche 18/10/2026 -> lundi 7h30 : campagne s9, latent SPATIAL (64 patchs, branche `patch`,
# code dans ~/psc/code_patch), loss multi-blocs, 4 époques, 42 machines (max 60).
set -e
cd "$(dirname "$0")"
python make_jobs_s9.py --machines 42 --out jobs_s9.txt
PSC_REMOTE_CODE='~/psc/code_patch' python -u cluster.py run jobs_s9.txt --deadline "2026-10-19 07:30" --launch-until "2026-10-18 22:00" --poll 60 --retry-failed
