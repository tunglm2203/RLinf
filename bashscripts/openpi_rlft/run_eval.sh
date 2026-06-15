#!/bin/bash
# Eval Pi0.5 (openpi) on LIBERO-10 / long (2x RTX 5090).
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"   # the venv's activate references $PYTHONPATH; guard it under `set -u`
source /opt/venv/openpi/bin/activate

WANDB_MODE="disabled"   # Keep this to disable log wandb online


# =========== USER's SETUP ============================================================================================
NUM_EPISODES=100                      # total episodes you want
ENVS=20                               # total envs run concurrently

#NOISE_METHOD="flow_noise"
NOISE_METHOD="flow_sde"

#EXP_NAME="pi05_libero10_sft_eval"
#EXP_NAME="pi05_libero10_ppo_flownoise_minicfg_eval"
EXP_NAME="pi05_libero10_ppo_flowsde_minicfg_eval"

# Checkpoint to evaluate (comment out to eval the SFT base directly).
#CKPT="/mnt/workspace/vla_bench/RLinf/logs/pi05_libero10_ppo_flownoise_minicfg/pi05_libero10_ppo_flownoise_minicfg/checkpoints/global_step_150"
CKPT="/mnt/workspace/vla_bench/RLinf/logs/pi05_libero10_ppo_flowsde_minicfg/pi05_libero10_ppo_flowsde_minicfg/checkpoints/global_step_150"

# =====================================================================================================================

# --- guards (near the existing checks) ---
NUM_TASKS=10
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); NUM_GPUS=${NUM_GPUS:-1}
(( ENVS % (NUM_TASKS*NUM_GPUS) == 0 )) || { echo "ENVS must be a multiple of num_tasks*gpus ($((NUM_TASKS*NUM_GPUS))) for even per-task split" >&2; exit 1; }
(( NUM_EPISODES % ENVS == 0 ))         || { echo "NUM_EPISODES must be a multiple of ENVS ($ENVS)" >&2; exit 1; }
N_EVAL_ITERS=$(( NUM_EPISODES / ENVS ))

# SFT base checkpoint: Keep this even when evaluating an RL checkpoint, it builds the model and supplies norm stats
export OPENPI_CKPT="/mnt/workspace/HUGGINGFACE_HOME/hub/RLinf-Pi05-LIBERO-SFT"

# Task -> eval config  (only 'long' is supported on the 5090 setup)
TASK="${1:-long}"
case "$TASK" in
  long|libero_10|10) CONFIG_NAME="libero_10_ppo_openpi_pi05_5090_eval" ;;
  *) echo "Unknown task '$TASK' (only 'long' is supported)" >&2; exit 1 ;;
esac
[ -d "$OPENPI_CKPT" ] || { echo "ERROR: SFT base not found: $OPENPI_CKPT" >&2; exit 1; }

# Resolve $CKPT into a runner.ckpt_path override (empty -> eval the SFT base directly).
CKPT_OVERRIDE=""
if [ -n "${CKPT:-}" ]; then
  PT="$CKPT"; [ -d "$PT" ] && PT="$PT/actor/model_state_dict/full_weights.pt"
  [ -f "$PT" ] || { echo "ERROR: weights .pt not found: $PT" >&2; exit 1; }
  CKPT_OVERRIDE="runner.ckpt_path=$PT"
fi

# Experiment name (drives the W&B run + logs/<EXP_NAME>/ dir; the config reads $EXP_NAME).
export EXP_NAME="${EXP_NAME:-$CONFIG_NAME}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Rendering / platform (venv: MuJoCo osmesa offscreen; ROBOT_PLATFORM sets action dim + normalization)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export HYDRA_FULL_ERROR=1

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_PATH"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"   # config searchpath uses ${oc.env:EMBODIED_PATH}
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"            # eval entrypoint imports rlinf from the repo
LOG_DIR="logs/${EXP_NAME}"; mkdir -p "$LOG_DIR"
echo "Task=$TASK  Config=$CONFIG_NAME  Noise=$NOISE_METHOD  Base=$OPENPI_CKPT  Checkpoint=${CKPT_OVERRIDE:-(direct SFT)}"

# Eval has its own entrypoint (EmbodiedEvalRunner). --config-path is absolute: Hydra resolves
# it relative to the .py file, not the cwd.
RAY_DEBUG=${RAY_DEBUG:-legacy} python examples/embodiment/eval_embodied_agent.py \
  --config-path "${REPO_PATH}/examples/embodiment/config" --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="$LOG_DIR" actor.model.openpi.noise_method="$NOISE_METHOD" \
  env.eval.total_num_envs="$ENVS" algorithm.eval_rollout_epoch="$N_EVAL_ITERS" \
  ${CKPT_OVERRIDE} 2>&1 | tee "$LOG_DIR/eval_embodiment.log"
