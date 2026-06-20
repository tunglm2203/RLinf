#!/bin/bash
# Eval GR00T N1.7 on LIBERO-10 (balanced 10/task, greedy). Mirrors run_eval.sh (N1.5) but for N1.7:
# separate gr00t_n1d7 venv + the Cosmos-Reason2-2B backbone. Logs locally (WANDB disabled).
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"   # the venv's activate references $PYTHONPATH; guard it under `set -u`
source /opt/venv/gr00t_n1d7/bin/activate

WANDB_MODE="disabled"   # keep eval local (no wandb upload)

# =========== USER's SETUP ============================================================================================
NUM_EPISODES=100   # total eval episodes (balanced across the 10 LIBERO-10 tasks)
ENVS=20            # envs run concurrently (N1.7 2B backbone is heavy on 2x32GB; raise only if you have headroom)

# Default SFT name; override via `EXP_NAME=... bash run_eval_gr00t_n1d7.sh` for other checkpoints
# (e.g. the PPO eval) so each run writes a distinct logs/<EXP_NAME>/ dir.
EXP_NAME="${EXP_NAME:-gr00t17_libero10_sft_eval}"

# N1.7 checkpoint to eval. Default = official SFT bootstrap (HF dir). For a trained RL checkpoint, point
# this at an already-exported HF dir (see note at the bottom re: PT->HF conversion for N1.7).
#EVAL_CKPT_PATH="/mnt/workspace/vla_bench/RLinf/logs/<run>/checkpoints/global_step_150"
# ====================================================================================================================

# --- guards: keep per-task counts balanced and the episode split exact (same checks as run_eval.sh) ---
NUM_TASKS=10
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); NUM_GPUS=${NUM_GPUS:-1}
(( ENVS % (NUM_TASKS*NUM_GPUS) == 0 )) || { echo "ENVS must be a multiple of num_tasks*gpus ($((NUM_TASKS*NUM_GPUS))) for even per-task split" >&2; exit 1; }
(( NUM_EPISODES % ENVS == 0 ))         || { echo "NUM_EPISODES must be a multiple of ENVS ($ENVS)" >&2; exit 1; }
N_EVAL_ITERS=$(( NUM_EPISODES / ENVS ))

# N1.7 backbone (VLM); required to build the model.
export GROOT_N1D7_BACKBONE="${GROOT_N1D7_BACKBONE:-/mnt/workspace/HUGGINGFACE_HOME/hub/Cosmos-Reason2-2B}"
# Default checkpoint: official NVIDIA N1.7 LIBERO bootstrap (libero_10 / long).
GROOT_N1D7_SFT_BOOTSTRAP="${GROOT_N1D7_SFT_BOOTSTRAP:-/mnt/workspace/HUGGINGFACE_HOME/hub/GR00T-N1.7-LIBERO/libero_10}"

TASK="${1:-long}"
case "$TASK" in
  long|libero_10|10) CONFIG_NAME="libero_10_ppo_gr00t_n1d7_5090_eval"; CHUNKS=16 ;;
  *) echo "Unknown task '$TASK' (expected: long)" >&2; exit 1 ;;
esac
[ -d "$GROOT_N1D7_BACKBONE" ] || { echo "ERROR: N1.7 backbone not found: $GROOT_N1D7_BACKBONE" >&2; exit 1; }

CKPT="${2:-${EVAL_CKPT_PATH:-$GROOT_N1D7_SFT_BOOTSTRAP}}"; CKPT="${CKPT%/}"

# Distinct run/log dir per checkpoint so SFT and RL evals don't clobber each other.
CKPT_TAG="$(basename "$CKPT")"
case "$CKPT_TAG" in
  global_step_*) CKPT_TAG="$(basename "$(dirname "$(dirname "$CKPT")")")_${CKPT_TAG}" ;;
esac

export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_PATH"
# HF dir (has config.json) -> eval directly; else an RL full_weights.pt -> convert to HF first.
# N1.7 conversion builds the model on CPU (Cosmos-Reason2-2B backbone + DiT head), loads the trained
# weights, and re-saves as sharded safetensors. The exported dir holds the FULL state dict, so eval
# restores backbone + head; the backbone path only scaffolds construction (see gr00t_n1d7_model_convertor.yaml).
if [ -f "$CKPT/config.json" ]; then
  EVAL_CKPT="$CKPT"
else
  PT=""
  for cand in "$CKPT/actor/model_state_dict/full_weights.pt" "$CKPT/model_state_dict/full_weights.pt" "$CKPT/full_weights.pt" "$CKPT"; do
    if [ -f "$cand" ] && [[ "$cand" == *.pt ]]; then PT="$cand"; break; fi
  done
  [ -n "$PT" ]                       || { echo "ERROR: '$CKPT' is neither an HF dir nor an RL checkpoint (.pt)." >&2; exit 1; }
  [ -d "$GROOT_N1D7_SFT_BOOTSTRAP" ] || { echo "ERROR: RL->HF conversion needs the SFT base: $GROOT_N1D7_SFT_BOOTSTRAP" >&2; exit 1; }
  EVAL_CKPT="${PT%/model_state_dict/full_weights.pt}"; EVAL_CKPT="${EVAL_CKPT%/actor}_hf_export"
  echo "Converting N1.7 RL checkpoint -> $EVAL_CKPT"
  GROOT_N1D7_RL_CKPT_PT="$PT" GROOT_N1D7_HF_OUT="$EVAL_CKPT" \
  GROOT_N1D7_SFT_BASE="$GROOT_N1D7_SFT_BOOTSTRAP" GROOT_N1D7_BACKBONE="$GROOT_N1D7_BACKBONE" \
    python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
      --config-path "${REPO_PATH}/rlinf/utils/ckpt_convertor/fsdp_convertor/config" \
      --config-name gr00t_n1d7_model_convertor model.num_action_chunks="$CHUNKS"
fi
export GROOT_N1D7_EVAL_CKPT="$EVAL_CKPT"

export EXP_NAME="${EXP_NAME:-${CONFIG_NAME}__${CKPT_TAG}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Rendering / platform / memory
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"   # config searchpath uses ${oc.env:EMBODIED_PATH}
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"            # eval entrypoint imports rlinf from the repo
LOG_DIR="logs/${EXP_NAME}"; mkdir -p "$LOG_DIR"
echo "Task=$TASK  Config=$CONFIG_NAME  Exp=$EXP_NAME  Episodes=${NUM_EPISODES} (envs=${ENVS} x iters=${N_EVAL_ITERS})  Ckpt=$GROOT_N1D7_EVAL_CKPT  Backbone=$GROOT_N1D7_BACKBONE"

RAY_DEBUG=${RAY_DEBUG:-legacy} python examples/embodiment/eval_embodied_agent.py \
  --config-path "${REPO_PATH}/examples/embodiment/config" --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  env.eval.total_num_envs="${ENVS}" algorithm.eval_rollout_epoch="${N_EVAL_ITERS}" \
  2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
