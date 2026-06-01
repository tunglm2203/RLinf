#!/bin/bash
# Eval GR00T N1.5 on LIBERO.  Usage: bash run_eval.sh [spatial|long] [checkpoint]   (defaults: long, the task's SFT ckpt)
set -eo pipefail

export GROOT_CKPT_SPATIAL="/mnt/workspace/HUGGINGFACE_HOME/hub/RLinf-Gr00t-SFT-Spatial"
export GROOT_CKPT_LONG="/mnt/workspace/HUGGINGFACE_HOME/hub/RLinf-Gr00t-SFT-Long"

# Task -> config + SFT base + action-chunk count
TASK="${1:-${EVAL_TASK:-long}}"
case "$TASK" in
  spatial)           CONFIG_NAME="libero_spatial_ppo_gr00t_5090_eval"; SFT_BASE="$GROOT_CKPT_SPATIAL"; CHUNKS=5 ;;
  long|libero_10|10) CONFIG_NAME="libero_10_ppo_gr00t_5090_eval";      SFT_BASE="$GROOT_CKPT_LONG";    CHUNKS=10 ;;
  *) echo "Unknown task '$TASK' (expected: spatial | long)" >&2; exit 1 ;;
esac
CKPT="${2:-${EVAL_CKPT_PATH:-$SFT_BASE}}"; CKPT="${CKPT%/}"

export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
cd "$REPO_PATH"

# HF dir (has config.json) -> eval directly; else an RL full_weights.pt -> convert to HF first.
if [ -f "$CKPT/config.json" ]; then
  EVAL_CKPT="$CKPT"
else
  PT=""
  for cand in "$CKPT/actor/model_state_dict/full_weights.pt" "$CKPT/model_state_dict/full_weights.pt" "$CKPT/full_weights.pt" "$CKPT"; do
    if [ -f "$cand" ] && [[ "$cand" == *.pt ]]; then PT="$cand"; break; fi
  done
  [ -n "$PT" ]        || { echo "ERROR: '$CKPT' is neither an HF dir nor an RL checkpoint (.pt)." >&2; exit 1; }
  [ -d "$SFT_BASE" ]  || { echo "ERROR: RL->HF conversion needs the SFT base: $SFT_BASE" >&2; exit 1; }
  EVAL_CKPT="${PT%/model_state_dict/full_weights.pt}"; EVAL_CKPT="${EVAL_CKPT%/actor}_hf_export"
  echo "Converting RL checkpoint -> $EVAL_CKPT"
  GROOT_RL_CKPT_PT="$PT" GROOT_HF_OUT="$EVAL_CKPT" GROOT_SFT_BASE="$SFT_BASE" \
    python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
      --config-path "${REPO_PATH}/rlinf/utils/ckpt_convertor/fsdp_convertor/config" \
      --config-name gr00t_model_convertor model.num_action_chunks="$CHUNKS"
fi
export GROOT_EVAL_CKPT="$EVAL_CKPT"

# Rendering / platform / memory (egl verified on this box; expandable_segments avoids frag OOM)
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export LIBERO_TYPE=${LIBERO_TYPE:-standard}
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH}
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# W&B: offline unless a valid key is exported
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
[ -z "$WANDB_API_KEY" ] && export WANDB_MODE="${WANDB_MODE:-offline}" || export WANDB_MODE="${WANDB_MODE:-online}"

LOG_DIR="${REPO_PATH}/logs/${CONFIG_NAME}"
mkdir -p "$LOG_DIR"
echo "Task=$TASK  Config=$CONFIG_NAME  Checkpoint=$GROOT_EVAL_CKPT"
RAY_DEBUG="legacy" python "${EMBODIED_PATH}/eval_embodied_agent.py" \
  --config-path "${EMBODIED_PATH}/config" --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
