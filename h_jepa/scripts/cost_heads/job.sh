#!/bin/bash
# Entry point of the cost-head jobs on the X cluster (one GPU per job).
#   job.sh encode                                  train set -> level-1 latents of seeds 42 43 44
#   job.sh plan <config> <seed> <cost_head> <name> planning eval (50 tasks), cost_head: l2|level2|path
#   job.sh train <seed> <name> <train_cost_head args...>
#   job.sh diag <seed> <name=spec>...              offline diagnostic (step 2)
#   job.sh wm / postwm ...                          joint training of a world model (see below)
# Needs HJEPA_HOME (checkpoints, val set, eval tasks, latents, heads, results).
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
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
    config=$1; seed=$2; head=$3; name=$4; shift 4   # remaining args: extra Hydra overrides
    # a trained head name (heads/<name>/seed<seed>/head.pt) instead of l2 / level2 / a path
    if [ -f "$H/heads/$head/seed$seed/head.pt" ]; then head="$H/heads/$head/seed$seed/head.pt"; fi
    python eval.py --config-name "$config" seed="$seed" policy="$(ckpt "$seed")" \
      ++cost_head="$head" output.dir="$H/results/plan/$name/seed$seed" "$@"
    ;;
  train)
    seed=$1; name=$2; shift 2
    python train_cost_head.py --latents "$H/latents/pusht_train" --seed "$seed" \
      --ckpt "$(ckpt "$seed")" --out "$H/heads/$name/seed$seed" "$@"
    ;;
  diag)
    # diag <seed> [tag=<tag>] <name=spec>... -> results/diag/seed<seed>[_<tag>].json
    seed=$1; shift
    tag=""
    if [[ "${1:-}" == tag=* ]]; then tag="_${1#tag=}"; shift; fi
    args=()
    for spec in "$@"; do args+=(--head "$spec"); done
    python cost_diagnostic.py --ckpt-dir "$CKPT_DIR" --seed "$seed" --val-h5 "$H/pusht_expert_val.h5" \
      --rank-latents "$H/latents/pusht_train" --out "$H/results/diag/seed$seed$tag.json" "${args[@]}"
    ;;
  pipe)
    # pipe <seed> <name> <train args...>: train the head, diagnose it, plan with GD (pusht_flat)
    seed=$1; name=$2; shift 2
    bash "$SELF" train "$seed" "$name" "$@"
    bash "$SELF" diag "$seed" "tag=$name" "$name=$H/heads/$name/seed$seed/head.pt"
    bash "$SELF" plan pusht_flat "$seed" "$name" "gd_$name"
    ;;
  wm)
    # wm <seed> <name> <cost_head l2|curv_head> <hydra overrides...>: train an H-JEPA (pusht_hjepa_l2)
    # on the machine-local disk, then diagnose it and plan with GD (postwm); only results go to $H.
    # WM_RESUME=1: resume the run from its lightning_resume/last.ckpt (local, else one stashed in
    # $H/wm_resume/<name>/seed<seed>/); otherwise start from an empty run dir.
    seed=$1; name=$2; head=$3; shift 3
    T=/tmp/$USER/hjepa_wm; mkdir -p "$T"
    if [ ! -s "$T/pusht_expert_train.h5" ] || [ -s "$T/train.h5.zst" ]; then
      curl -sfL --retry 8 --retry-all-errors --retry-delay 20 -C - -o "$T/train.h5.zst" \
        https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst
      unzstd -q -f --rm "$T/train.h5.zst" -o "$T/pusht_expert_train.h5"
    fi
    [ -s "$T/pusht_expert_val.h5" ] || cp "$H/pusht_expert_val.h5" "$T/"
    python ../scripts/data/add_pusht_block_ori.py "$T/pusht_expert_train.h5"  # README step (idempotent)
    R=$T/ckpts/pusht/$name/seed$seed
    if [ -n "${WM_RESUME:-}" ]; then
      S=$H/wm_resume/$name/seed$seed/last.ckpt
      if [ ! -s "$R/lightning_resume/last.ckpt" ] && [ -s "$S" ]; then
        mkdir -p "$R/lightning_resume"; cp "$S" "$R/lightning_resume/"
      fi
      ls -la "$R/lightning_resume/last.ckpt"
    else
      rm -rf "$R"   # no resume from an earlier job of the same name
    fi
    HJEPA_HOME=$T python main_hjepa.py --config-name pusht_hjepa_l2 seed="$seed" \
      output_model_name="$name" planning_eval.enabled=false final_probing_decoding_eval.enabled=false \
      num_workers="${NUM_WORKERS:-8}" "$@"
    # small artifacts home: training logs of the run (not the 93 MB checkpoints)
    mkdir -p "$H/wm_runs/$name/seed$seed"
    find "$R" -maxdepth 3 \( -name "*.yaml" -o -name "*.csv" -o -name "*.json" -o -name "*.log" \) \
      -size -5M -exec cp {} "$H/wm_runs/$name/seed$seed/" \; || true
    bash "$SELF" postwm "$seed" "$name" "$head"
    ;;
  postwm)
    # postwm <seed> <name> <cost_head> [epoch]: offline diagnostic + GD planning of a world model
    # trained by `wm` on this machine: its final checkpoint, or the one of epoch <epoch>
    # (results then named <name>_e<epoch>).
    seed=$1; name=$2; head=$3; ep=${4:-}
    T=/tmp/$USER/hjepa_wm; R=$T/ckpts/pusht/$name/seed$seed
    out=$name; D=$T/ckpts/pusht/$name
    if [ -n "$ep" ]; then   # run-dir view whose <out>_object.ckpt is the epoch checkpoint
      out=${name}_e$ep; D=$T/post/$out; mkdir -p "$D/seed$seed"
      for f in "$R"/*; do if [ -f "$f" ]; then ln -sf "$f" "$D/seed$seed/"; fi; done
      ln -sf "$R/${name}_epoch_${ep}_object.ckpt" "$D/seed$seed/${out}_object.ckpt"
    fi
    CK=$D/seed$seed/${out}_object.ckpt
    ls -laL "$CK"
    specs=("$out=l2")
    [ "$head" = curv_head ] && specs+=("${out}_h=curv_head")
    args=(); for spec in "${specs[@]}"; do args+=(--head "$spec"); done
    python cost_diagnostic.py --ckpt-dir "$D" --seed "$seed" --val-h5 "$T/pusht_expert_val.h5" \
      --out "$H/results/diag/seed${seed}_$out.json" "${args[@]}"
    python eval.py --config-name pusht_flat seed="$seed" policy="$CK" \
      ++cost_head="$head" output.dir="$H/results/plan/gd_$out/seed$seed"
    ;;
  *) echo "unknown job kind $kind"; exit 2 ;;
esac
