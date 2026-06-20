#!/bin/bash
# PPO fine-tuning (flow-SDE, openpi LRs) of GR00T N1.7 on LIBERO-10 / long (2x RTX 5090).
# Mirrors run_ppo_flowsde_lropenpi.sh (N1.5) but for the N1.7 model: separate gr00t_n1d7 venv,
# the Cosmos-Reason2-2B backbone, and the N1.7-validated memory profile (micro=2, offload, full_shard).
# Usage: bash run_ppo_gr00t_n1d7.sh [long]   (default: long)
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"   # the venv's activate references $PYTHONPATH; guard it under `set -u`
source /opt/venv/gr00t_n1d7/bin/activate

# W&B: provide credentials via the environment, do NOT hardcode the key.
export WANDB_API_KEY="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"
export WANDB_ENTITY="tunglm"
# Defaults to online if a key is present, else offline (local only).
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_MODE="${WANDB_MODE:-online}"; else export WANDB_MODE="${WANDB_MODE:-offline}"; fi

EXP_NAME="gr00t17_libero10_ppo_flowsde_lropenpi"

# N1.7 checkpoint (official NVIDIA bootstrap) + VLM backbone. Override via env if your paths differ.
export GROOT_N1D7_CKPT="${GROOT_N1D7_CKPT:-/mnt/workspace/HUGGINGFACE_HOME/hub/GR00T-N1.7-LIBERO/libero_10}"
export GROOT_N1D7_BACKBONE="${GROOT_N1D7_BACKBONE:-/mnt/workspace/HUGGINGFACE_HOME/hub/Cosmos-Reason2-2B}"

# Task -> config  (only 'long' is supported on the 5090 setup)
TASK="${1:-long}"
case "$TASK" in
  long|libero_10|10) CONFIG_NAME="libero_10_ppo_gr00t_n1d7_5090" ;;
  *) echo "Unknown task '$TASK' (only 'long' is supported)" >&2; exit 1 ;;
esac
[ -d "$GROOT_N1D7_CKPT" ]     || { echo "ERROR: N1.7 checkpoint not found: $GROOT_N1D7_CKPT" >&2; exit 1; }
[ -d "$GROOT_N1D7_BACKBONE" ] || { echo "ERROR: N1.7 backbone not found: $GROOT_N1D7_BACKBONE" >&2; exit 1; }

# Experiment name (drives the W&B run + logs/<EXP_NAME>/ dir; the config reads $EXP_NAME).
export EXP_NAME="${EXP_NAME:-$CONFIG_NAME}"

# Rendering / platform (venv: MuJoCo osmesa offscreen; ROBOT_PLATFORM sets action dim + normalization)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Task=$TASK  Config=$CONFIG_NAME  Ckpt=$GROOT_N1D7_CKPT  Backbone=$GROOT_N1D7_BACKBONE  Exp=$EXP_NAME"
# run_embodiment.sh attaches to (or auto-starts) Ray and runs the Hydra entrypoint.
RAY_DEBUG=${RAY_DEBUG:-legacy} bash examples/embodiment/run_embodiment.sh "$CONFIG_NAME" "$ROBOT_PLATFORM"
