#!/bin/bash
# Installation de H-JEPA (fork 7Romain7/H-JEPA, branche cost-heads) sur le home partagé de l'X.
# Usage (depuis le PC) :
#   Get-Content hjepa_setup.sh -Raw | ssh login@machine.polytechnique.fr "tr -d '\r' | bash -s"
# Crée ~/psc/hjepa (code), ~/psc/hjepa_env (conda, Python 3.11), ~/psc/hjepa_home (HJEPA_HOME :
# checkpoints Push-T, tâches d'éval, val set). Le gros train set ne va PAS dans le home (quota) :
# il est téléchargé dans /tmp par le job d'encodage des latents.
set -e
ROOT=~/psc
CODE=$ROOT/hjepa
ENV=$ROOT/hjepa_env
HOME_H=$ROOT/hjepa_home
BRANCH=${BRANCH:-cost-heads}
HF=https://huggingface.co

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
quota -s | tail -1

[ -d $CODE ] || git clone -b $BRANCH https://github.com/7Romain7/H-JEPA.git $CODE
cd $CODE && git fetch -q && git checkout -q $BRANCH && git pull -q --ff-only

source "$(conda info --base)/etc/profile.d/conda.sh"
[ -d $ENV ] || conda create -y -q -p $ENV --override-channels -c conda-forge python=3.11
conda activate $ENV
export PIP_CACHE_DIR=/tmp/pipcache-$USER
pip install -q --upgrade pip
pip install -q -e ".[train,env]" --extra-index-url https://download.pytorch.org/whl/cu128
rm -rf /tmp/pipcache-$USER

mkdir -p $HOME_H/eval_trajs/pusht
for s in 42 43 44; do
  d=$HOME_H/ckpts/pusht/pusht_hjepa_l2/seed$s; mkdir -p $d
  for f in pusht_hjepa_l2_object.ckpt normalizer.pt; do
    [ -s $d/$f ] || curl -sfL -o $d/$f $HF/jepa-world-models/h-jepa/resolve/main/pusht/pusht_hjepa_l2/seed$s/$f
  done
done
[ -s $HOME_H/pusht_expert_val.h5 ] || curl -sfL -o $HOME_H/pusht_expert_val.h5 \
  $HF/datasets/jepa-world-models/h-jepa/resolve/main/pusht_expert_val.h5
[ -s $HOME_H/eval_trajs/pusht/goal_offset_75_val.pt ] || curl -sfL -o $HOME_H/eval_trajs/pusht/goal_offset_75_val.pt \
  $HF/datasets/jepa-world-models/h-jepa/resolve/main/eval_trajs/pusht/goal_offset_75_val.pt
# les configs d'éval lisent assets/eval_trajs/... relativement à h_jepa/
mkdir -p $CODE/h_jepa/assets && ln -sfn $HOME_H/eval_trajs $CODE/h_jepa/assets/eval_trajs

python - <<'EOF'
import torch, stable_worldmodel
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
EOF
du -sh $ENV $HOME_H
quota -s | tail -1
echo "SETUP OK"
