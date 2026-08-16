#!/bin/bash
# Pretrain the progress value head (vr_value_head/) on ego-centric egoverse + frozen GR00T N1.7.
#
# All three stages run in ONE venv: /opt/venv/egoverse already carries the full gr00t_n1d7 torch
# stack (torch 2.7.1+cu128, transformers 4.57.3, sm_120) AND zarr, so it reads egoverse frames
# AND runs the frozen Cosmos-Reason2-2B backbone. Verified end-to-end. (gr00t_n1d7 also works for
# stages 2-3 but lacks zarr for stage 1 — set VENV=/opt/venv/gr00t_n1d7 to use it there.)
#   1. build_labels     (reads zarr frames, no backbone)         -> cache/
#   2. extract_features  (frozen Cosmos-Reason2-2B, needs a GPU)  -> cache/features/
#   3. train            (MLP on the cache, seconds, CPU ok)       -> runs/
#
# Stages 1-2 are the expensive, cache-producing steps; stage 3 is cheap and re-runnable. Set
# STAGE to skip ahead (e.g. re-train with a different pooling without re-extracting):
#   STAGE=all      full pipeline (default)
#   STAGE=labels   stage 1 only
#   STAGE=extract  stages 1-2
#   STAGE=train    stage 3 only  (requires an existing cache)
#
# Usage:
#   bash bashscripts/pretrain_value_head/run.sh
#   EPISODES=200 ARM=subgoal POOL=mean_last TIME_WARP=1 bash bashscripts/pretrain_value_head/run.sh
#   STAGE=train POOL=last_last bash bashscripts/pretrain_value_head/run.sh   # re-train off the cache
set -euo pipefail

# ---- knobs (defaults = the LOCKED baseline; all overridable from the environment) ------------
# Locked baseline (2026-08): last_last pooling · robometer head · AdamW(cosine,warmup,wd) ·
# discrete 10-bin CE · negatives-excluded · 500 episodes · 150 epochs.
# Held-out result: test ρ 0.32, language-sensitivity 0.31 (clears M2 GO bar). See vr_value_head/README.
EPISODES="${EPISODES:-500}"        # egoverse episodes to draw from (of 500)
STRIDE="${STRIDE:-10}"             # frame subsampling within a segment
ARM="${ARM:-subgoal}"              # {subgoal | full}  text-slot composition (full at handover)
NEG="${NEG:-1}"                    # wrong-sub-goal negatives per positive frame (for the metric)
POOL="${POOL:-last_last}"          # {mean_last | last_last | mean_mid} — last_last wins
EPOCHS="${EPOCHS:-150}"
LR="${LR:-1e-3}"                   # for the frozen-feature MLP (robometer's 2e-5 is for VLM FT)
SEGS_PER_BATCH="${SEGS_PER_BATCH:-16}"
TIME_WARP="${TIME_WARP:-0}"        # off: sequence-level anti-clock doesn't apply to a per-frame head
BATCH_SIZE="${BATCH_SIZE:-16}"     # backbone extraction batch
DEVICE="${DEVICE:-cuda}"
N_BINS="${N_BINS:-10}"             # 10=discrete CE (robometer, +0.07 ρ over huber); 1=continuous huber
STAGE="${STAGE:-all}"

# single venv for all stages — egoverse has zarr + the full torch stack
VENV="${VENV:-/opt/venv/egoverse}"

# ---- paths -----------------------------------------------------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="$REPO/vr_value_head"
CACHE="${CACHE:-$PKG/cache}"                       # cache/ (pairs.parquet + frames/ + features/)
RUN_NAME="${RUN_NAME:-vh_${ARM}_${POOL}$([ "$TIME_WARP" = 1 ] && echo _tw)}"
OUT="${OUT:-$PKG/runs/$RUN_NAME}"
# resolve to absolute against the INVOCATION dir — the stages run after `cd "$PKG"`, so a
# relative CACHE/OUT would otherwise be read relative to $PKG (surprising). realpath -m does
# not require the path to exist yet.
CACHE="$(realpath -m "$CACHE")"
OUT="$(realpath -m "$OUT")"
LOG_DIR="$REPO/logs/pretrain_value_head/$RUN_NAME"
mkdir -p "$LOG_DIR"

# ---- guards ----------------------------------------------------------------------------------
[ -x "$VENV/bin/python" ] || { echo "ERROR: venv missing: $VENV" >&2; exit 1; }
case "$STAGE" in all|labels|extract|train) ;; *) echo "ERROR: bad STAGE=$STAGE" >&2; exit 1 ;; esac

echo "=== pretrain_value_head | STAGE=$STAGE | venv=$VENV ==="
echo "  episodes=$EPISODES stride=$STRIDE arm=$ARM neg=$NEG | pool=$POOL epochs=$EPOCHS lr=$LR tw=$TIME_WARP"
echo "  cache=$CACHE"
echo "  out=$OUT   log=$LOG_DIR"
cd "$PKG"

# ---- stage 1: build labels + frame cache -----------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = labels ] || [ "$STAGE" = extract ]; then
  echo "--- [1/3] build_labels ---"
  "$VENV/bin/python" build_labels.py \
    --out "$CACHE" --episodes "$EPISODES" --stride "$STRIDE" --arm "$ARM" --neg "$NEG" \
    2>&1 | tee "$LOG_DIR/1_build_labels.log"
fi

# ---- stage 2: extract frozen-backbone features (GPU) -----------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = extract ]; then
  echo "--- [2/3] extract_features (GPU) ---"
  "$VENV/bin/python" extract_features.py \
    --cache "$CACHE" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/2_extract_features.log"
fi

# ---- stage 3: train the head on the cache ----------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = train ]; then
  echo "--- [3/3] train ---"
  [ -f "$CACHE/pairs.parquet" ] || { echo "ERROR: no cache at $CACHE — run STAGE=extract first" >&2; exit 1; }
  TW_FLAG=""; [ "$TIME_WARP" = 1 ] && TW_FLAG="--time-warp"
  "$VENV/bin/python" train.py \
    --cache "$CACHE" --pool "$POOL" --epochs "$EPOCHS" --lr "$LR" \
    --segs-per-batch "$SEGS_PER_BATCH" $TW_FLAG --device "$DEVICE" --out "$OUT" --n-bins "$N_BINS" \
    2>&1 | tee "$LOG_DIR/3_train.log"
  echo "head + metrics -> $OUT"
fi

echo "=== done -> $LOG_DIR ==="
