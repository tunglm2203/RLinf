#!/bin/bash
# Robometer-aligned evaluation of the progress value head on the 3 standard eval groups
# (RBM-EVAL-ID / RBM-EVAL-OOD / LIBERO-mixed, sourced from robometer's dataset_category.py).
#
# Matches robometer's eval params (bash_scripts/eval/eval_{reward_alignment,policy_ranking}.sh):
#   - frames: linspace subsample to MAX_FRAMES=8 per trajectory
#   - episodes: ALL (no cap)
#   - metrics: reward_alignment (rho / VOC / tau_a, per-trajectory, SUCCESS-only)
#              + Suc-Fail (pairwise preference accuracy on success-vs-failure terminal progress)
# The ONLY intentional difference: our head is SINGLE-FRAME (theirs is an 8-frame video). The metric
# computation is identical; the prediction is not (temporal context is deferred — see M8).
#
# Stages (feature cache is reusable — re-eval a different head in seconds):
#   1. build_labels_rbm  (CPU)  -> $CACHE/{pairs.parquet, frames/}
#   2. extract_features  (GPU)  -> $CACHE/features/        (backbone = $MODEL, MUST match $HEAD)
#   3. eval_rbm_groups   (CPU)  -> per-group fmt7 report + logs
#
#   bash bashscripts/pretrain_value_head/eval_rbm.sh                         # Cosmos head, all stages
#   STAGE=eval bash bashscripts/pretrain_value_head/eval_rbm.sh              # re-eval off the cache
#   # base Qwen3-VL head needs its OWN feature cache (backbone must match the head):
#   MODEL=/mnt/workspace/HUGGINGFACE_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/<hash> \
#     HEAD=$PWD/vr_value_head/runs/vh_ego_qwenbase/head.pt CACHE=$PWD/vr_value_head/cache_rbm_eval_qwenbase \
#     LABEL=qwen3vl_ego bash bashscripts/pretrain_value_head/eval_rbm.sh
set -euo pipefail

VENV="${VENV:-/opt/venv/egoverse}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="$REPO/vr_value_head"

# ---- knobs (defaults = robometer-aligned Cosmos frozen-ego eval) ------------------------------
MODEL="${MODEL:-/mnt/workspace/HUGGINGFACE_HOME/hub/Cosmos-Reason2-2B}"   # backbone — MUST match HEAD
HEAD="${HEAD:-$PKG/runs/vh_baseline_500_ce10/head.pt}"                    # value head to score
CACHE="${CACHE:-$PKG/cache_rbm_eval}"                                     # per-backbone feature cache
LABEL="${LABEL:-cosmos_ego}"
ROOT="${ROOT:-/mnt/workspace2/all_dataset/robometer_dataset/processed_datasets}"
MAX_FRAMES="${MAX_FRAMES:-8}"     # robometer eval subsamples to 8
NEG="${NEG:-0}"                   # 0 = pure robometer parity (no language metric); 1 also computes lang
CAP="${CAP:-0}"                   # 0 = all episodes (robometer uses all)
POOL="${POOL:-last_last}"
N_BINS="${N_BINS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-1}"                   # extract on GPU 1 (keep GPU 0 free to avoid overlap)
PER_DATASET="${PER_DATASET:-1}"
STAGE="${STAGE:-all}"

CACHE="$(realpath -m "$CACHE")"
LOG_DIR="$REPO/logs/eval_rbm/$LABEL"; mkdir -p "$LOG_DIR"

# ---- guards ----------------------------------------------------------------------------------
[ -x "$VENV/bin/python" ] || { echo "ERROR: venv missing: $VENV" >&2; exit 1; }
case "$STAGE" in all|build|extract|eval) ;; *) echo "ERROR: bad STAGE=$STAGE" >&2; exit 1 ;; esac

echo "=== eval_rbm | STAGE=$STAGE | backbone=$(basename "$MODEL") | head=$HEAD ==="
echo "  cache=$CACHE | frames=$MAX_FRAMES(linspace) neg=$NEG cap=$CAP | GPU=$GPU"
echo "  log=$LOG_DIR"
cd "$PKG"

# ---- stage 1: build labels (all 18 eval datasets, linspace-8) ---------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = build ]; then
  echo "--- [1/3] build_labels_rbm ---"
  "$VENV/bin/python" build_labels_rbm.py --groups --cap "$CAP" --linspace --max-frames "$MAX_FRAMES" \
      --neg "$NEG" --root "$ROOT" --out "$CACHE" 2>&1 | tee "$LOG_DIR/1_build.log"
fi

# ---- stage 2: extract features (GPU, backbone MUST match the head) ----------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = extract ]; then
  echo "--- [2/3] extract_features (GPU $GPU) ---"
  [ -f "$CACHE/pairs.parquet" ] || { echo "ERROR: no cache at $CACHE — run STAGE=build first" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="$GPU" "$VENV/bin/python" extract_features.py \
      --cache "$CACHE" --model "$MODEL" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
      2>&1 | tee "$LOG_DIR/2_extract.log"
fi

# ---- stage 3: eval on the 3 groups -----------------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo "--- [3/3] eval_rbm_groups ---"
  [ -n "$(ls "$CACHE"/features/*.npz 2>/dev/null)" ] || { echo "ERROR: no features in $CACHE — run STAGE=extract first" >&2; exit 1; }
  PD=""; [ "$PER_DATASET" = 1 ] && PD="--per-dataset"
  "$VENV/bin/python" eval_rbm_groups.py --cache "$CACHE" --head "$HEAD" --pool "$POOL" \
      --n-bins "$N_BINS" --label "$LABEL" $PD 2>&1 | tee "$LOG_DIR/3_eval.log"
fi

echo "=== done -> $LOG_DIR ==="
