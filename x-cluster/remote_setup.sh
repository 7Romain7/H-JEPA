#!/bin/bash
# À lancer UNE fois, sur une machine des salles info :  ssh ablette.polytechnique.fr 'bash -s' < remote_setup.sh
# Installe le venv + stable-worldmodel dans ~/psc (home partagé → valable sur toutes les machines).
set -e
REPO_URL="${REPO_URL:-https://github.com/Reddimolk/psc-jepa-straightening.git}"   # remplace par ton fork/repo PSC
ROOT=~/psc
mkdir -p $ROOT/runs $ROOT/datasets $ROOT/swm_home
# mêmes emplacements que ceux utilisés par cluster.py pour chaque job
export STABLEWM_HOME=$ROOT/swm_home LOCAL_DATASET_DIR=$ROOT/datasets

echo "== Quota / place disque (torch+CUDA ~ 5-6 Go) =="
quota -s 2>/dev/null || true
df -h ~ | tail -1

echo "== GPU / CUDA =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "python3 systeme (ignore, on utilise conda ci-dessous) : $(python3 --version)"

[ -d $ROOT/code ] || git clone "$REPO_URL" $ROOT/code

# stable-worldmodel exige Python >= 3.10 ; le python3 systeme de ces machines est 3.9.
# On cree donc l'environnement avec conda (disponible sur ces machines) plutot qu'avec
# python3 -m venv, en gardant le meme repertoire $ROOT/venv pour ne rien changer au reste.
if [ ! -d $ROOT/venv ]; then
    conda create -y -p $ROOT/venv --override-channels -c conda-forge python=3.11
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ROOT/venv
echo "python de l'environnement conda : $(python --version)"

pip install --upgrade pip
# PIP_CACHE_DIR sur /tmp pour ne pas bouffer le quota du home avec le cache
export PIP_CACHE_DIR=/tmp/pipcache-$USER
cd $ROOT/code && git pull --ff-only
if [ -f requirements.txt ]; then
    pip install -r requirements.txt          # ton repo PSC (pas un package Python)
elif [ -f pyproject.toml ] || [ -f setup.py ]; then
    pip install -e ".[all]"
fi

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda dispo:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import stable_worldmodel; print("stable_worldmodel OK")
EOF
# TODO : pré-télécharger ici les datasets utilisés (commande à trouver dans la doc stable-worldmodel),
# pour éviter que 100 jobs les téléchargent en même temps.
du -sh $ROOT/venv $ROOT/datasets 2>/dev/null
echo "Setup termine."
