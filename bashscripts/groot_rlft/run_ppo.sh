#!/bin/bash
# GR00T N1.5 PPO fine-tuning on LIBERO (2x RTX 5090).  Usage: bash run_ppo.sh [spatial|long]   (default: spatial)
set -eo pipefail

# Activate the gr00t virtual environment
source /opt/venv/gr00t/bin/activate

# W&B (online when a key is set, else offline)
export WANDB_API_KEY="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"
export WANDB_ENTITY="tunglm"
[ -z "$WANDB_API_KEY" ] && export WANDB_MODE="${WANDB_MODE:-offline}" || export WANDB_MODE="${WANDB_MODE:-online}"

# SFT checkpoints (the configs read these env vars)
export GROOT_CKPT_SPATIAL="/mnt/workspace/HUGGINGFACE_HOME/hub/RLinf-Gr00t-SFT-Spatial"
export GROOT_CKPT_LONG="/mnt/workspace/HUGGINGFACE_HOME/hub/RLinf-Gr00t-SFT-Long"

# Task -> config + checkpoint
TASK="${1:-long}"
case "$TASK" in
  spatial)           CONFIG_NAME="libero_spatial_ppo_gr00t_5090"; CKPT="$GROOT_CKPT_SPATIAL" ;;
  long|libero_10|10) CONFIG_NAME="libero_10_ppo_gr00t_5090";      CKPT="$GROOT_CKPT_LONG" ;;
  *) echo "Unknown task '$TASK' (expected: spatial | long)" >&2; exit 1 ;;
esac
[ -d "$CKPT" ] || { echo "ERROR: checkpoint dir for '$TASK' not found: $CKPT" >&2; exit 1; }

# Rendering / platform (LIBERO MuJoCo EGL offscreen; ROBOT_PLATFORM sets action dim + normalization)
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export LIBERO_TYPE=${LIBERO_TYPE:-standard}

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Task=$TASK  Config=$CONFIG_NAME  Checkpoint=$CKPT"
# run_embodiment.sh attaches to (or auto-starts) Ray and runs the Hydra entrypoint.
RAY_DEBUG=${RAY_DEBUG:-legacy} bash examples/embodiment/run_embodiment.sh "$CONFIG_NAME" "$ROBOT_PLATFORM"
