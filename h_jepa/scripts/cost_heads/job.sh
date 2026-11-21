#!/bin/bash
# Entry point of the cost-head jobs on the X cluster (one GPU per job).
#   job.sh encode                                  train set -> level-1 latents of seeds 42 43 44
#   job.sh plan <config> <seed> <cost_head> <name> planning eval (50 tasks), cost_head: l2|level2|path
#   job.sh train <seed> <name> <train_cost_head args...>
#   job.sh diag <seed> <name=spec>...              offline diagnostic (step 2)
# Needs HJEPA_HOME (checkpoints, val set, eval tasks, latents, heads, results).
set -euo pipefail
cd "$(dirname "$0")/../.."   # h_jepa/
H=${HJEPA_HOME:?HJEPA_HOME not set}
CKPT_DIR=$H/ckpts/pusht/pusht_hjepa_l2
ckpt() { echo "$CKPT_DIR/seed$1/pusht_hjepa_l2_object.ckpt"; }
export MUJOCO_GL=${MUJOCO_GL:-egl}

kind=$1; shift
case $kind in
  encode)
    TMP=/tmp/$USER/pusht_train; mkdir -p "$TMP"
    if [ ! -s "$TMP/pusht_expert_train.h5" ]; then
      curl -sfL -o "$TMP/train.h5.zst" \
        https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst
      unzstd -q -f --rm "$TMP/train.h5.zst" -o "$TMP/pusht_expert_train.h5"
    fi
    python encode_latents.py --h5 "$TMP/pusht_expert_train.h5" --out "$H/latents/pusht_train" \
      --ckpt-dir "$CKPT_DIR" --seeds 42 43 44 --max-episodes "${MAX_EPISODES:-4000}"
    rm -rf "$TMP"
    ;;
  plan)
    config=$1; seed=$2; head=$3; name=$4
    # a trained head name (heads/<name>/seed<seed>/head.pt) instead of l2 / level2 / a path
    if [ -f "$H/heads/$head/seed$seed/head.pt" ]; then head="$H/heads/$head/seed$seed/head.pt"; fi
    python eval.py --config-name "$config" seed="$seed" policy="$(ckpt "$seed")" \
      ++cost_head="$head" output.dir="$H/results/plan/$name/seed$seed"
    ;;
  train)
    seed=$1; name=$2; shift 2
    python train_cost_head.py --latents "$H/latents/pusht_train" --seed "$seed" \
      --out "$H/heads/$name/seed$seed" "$@"
    ;;
  diag)
    seed=$1; shift
    args=()
    for spec in "$@"; do args+=(--head "$spec"); done
    python cost_diagnostic.py --ckpt-dir "$CKPT_DIR" --seed "$seed" --val-h5 "$H/pusht_expert_val.h5" \
      --rank-latents "$H/latents/pusht_train" --out "$H/results/diag/seed$seed.json" "${args[@]}"
    ;;
  *) echo "unknown job kind $kind"; exit 2 ;;
esac
