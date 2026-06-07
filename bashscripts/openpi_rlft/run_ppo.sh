#!/bin/bash
# PPO fine-tuning on LIBERO-10 / long (2x RTX 5090).  Usage: bash run_ppo.sh [long]   (default: long)
set -euo pipefail

source /opt/venv/openpi/bin/activate

# W&B (online when a mode is set, else offline)
export WANDB_API_KEY="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"
export WANDB_ENTITY="tunglm"
WANDB_MODE="online"

EXP_NAME="pi05_libero10_ppo_flownoise_minimal_config"

# SFT checkpoint (the config reads this env var, Pi0.5 uses ONE LIBERO checkpoint for all suites)
export OPENPI_CKPT="/mnt/workspace/HUGGINGFACE_HOME/hub/RLinf-Pi05-LIBERO-SFT"

# Task -> config + checkpoint  (only 'long' is supported on the 5090 setup)
TASK="${1:-long}"
case "$TASK" in
  long|libero_10|10) CONFIG_NAME="libero_10_ppo_openpi_pi05_5090"; CKPT="$OPENPI_CKPT" ;;
  *) echo "Unknown task '$TASK' (only 'long' is supported)" >&2; exit 1 ;;
esac
[ -d "$CKPT" ] || { echo "ERROR: checkpoint dir for '$TASK' not found: $CKPT" >&2; exit 1; }

# Experiment name (drives the W&B run + logs/<EXP_NAME>/ dir; the config reads $EXP_NAME).
# Defaults to the config name so runs don't collide. Override with: EXP_NAME=my_tag bash run_ppo.sh long
export EXP_NAME="${EXP_NAME:-$CONFIG_NAME}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Rendering / platform (venv: MuJoCo offscreen; ROBOT_PLATFORM sets action dim + normalization)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # reduce fragmentation OOM (openpi NCCL 2.28.9)

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Task=$TASK  Config=$CONFIG_NAME  Checkpoint=$CKPT  Exp=$EXP_NAME"
# run_embodiment.sh attaches to (or auto-starts) Ray and runs the Hydra entrypoint.
RAY_DEBUG=${RAY_DEBUG:-legacy} bash examples/embodiment/run_embodiment.sh "$CONFIG_NAME" "$ROBOT_PLATFORM"
