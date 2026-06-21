#!/bin/bash
# Eval GR00T N1.x on LIBERO-10
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"   # the venv's activate references $PYTHONPATH; guard it under `set -u`
source /opt/venv/gr00t/bin/activate

WANDB_MODE="disabled"   # keep eval local (no wandb upload)

# =========== USER's SETUP ============================================================================================
NUM_EPISODES=100   # total eval episodes (balanced across the 10 LIBERO-10 tasks)
ENVS=20            # envs run concurrently

# Eval horizon (episode-step truncation), overridable; config default is 480.
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-480}"

#EVAL_CKPT_PATH="${EVAL_CKPT_PATH:-/mnt/workspace/vla_bench/RLinf/logs/gr00t15_libero10_ppo_flowsde_minicfg_lropenpi/gr00t15_libero10_ppo_flowsde_minicfg_lropenpi/checkpoints/global_step_150}"
# ====================================================================================================================

# --- guards: keep per-task counts balanced and the episode split exact (same checks as openpi run_eval.sh) ---
NUM_TASKS=10
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); NUM_GPUS=${NUM_GPUS:-1}
(( ENVS % (NUM_TASKS*NUM_GPUS) == 0 )) || { echo "ENVS must be a multiple of num_tasks*gpus ($((NUM_TASKS*NUM_GPUS))) for even per-task split" >&2; exit 1; }
(( NUM_EPISODES % ENVS == 0 ))         || { echo "NUM_EPISODES must be a multiple of ENVS ($ENVS)" >&2; exit 1; }
N_EVAL_ITERS=$(( NUM_EPISODES / ENVS ))

# SFT base checkpoint: Keep this even when evaluating an RL checkpoint, it builds the model and supplies norm stats
export GROOT_CKPT_LONG="/mnt/workspace/HUGGINGFACE_HOME/hub/RLinf-Gr00t-SFT-Long"

# Task -> config + SFT base + action-chunk count
TASK="${1:-long}"
case "$TASK" in
  long|libero_10|10) CONFIG_NAME="libero_10_ppo_gr00t_5090_eval";      SFT_BASE="$GROOT_CKPT_LONG";    CHUNKS="${CHUNKS:-10}" ;;
  *) echo "Unknown task '$TASK' (expected: spatial | long)" >&2; exit 1 ;;
esac
[ -d "$GROOT_CKPT_LONG" ] || { echo "ERROR: SFT base not found: $GROOT_CKPT_LONG" >&2; exit 1; }

# Experiment name follows the chunk/horizon rule (overridable via EXP_NAME=...).
EXP_NAME="${EXP_NAME:-eval_gr00t15_libero10_sft_chunk${CHUNKS}_horizon${MAX_EPISODE_STEPS}}"

CKPT="${2:-${EVAL_CKPT_PATH:-$SFT_BASE}}"; CKPT="${CKPT%/}"

# Distinct run/log dir per checkpoint so SFT and PPO evals don't clobber each other
CKPT_TAG="$(basename "$CKPT")"
case "$CKPT_TAG" in
  global_step_*) CKPT_TAG="$(basename "$(dirname "$(dirname "$CKPT")")")_${CKPT_TAG}" ;;
esac

export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_PATH"
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

# Experiment name (drives the W&B run + logs/<EXP_NAME>/ dir; the config reads $EXP_NAME).
export EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_${CKPT_TAG}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Rendering / platform / memory (egl verified on this box; expandable_segments avoids frag OOM)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"   # config searchpath uses ${oc.env:EMBODIED_PATH}
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"            # eval entrypoint imports rlinf from the repo
LOG_DIR="logs/${EXP_NAME}"; mkdir -p "$LOG_DIR"
echo "Task=$TASK  Config=$CONFIG_NAME  Exp=$EXP_NAME  Episodes=${NUM_EPISODES} (envs=${ENVS} x iters=${N_EVAL_ITERS})  Checkpoint=$GROOT_EVAL_CKPT"

RAY_DEBUG=${RAY_DEBUG:-legacy} python examples/embodiment/eval_embodied_agent.py \
  --config-path "${REPO_PATH}/examples/embodiment/config" --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  env.eval.total_num_envs="${ENVS}" algorithm.eval_rollout_epoch="${N_EVAL_ITERS}" \
  actor.model.num_action_chunks="${CHUNKS}" \
  env.eval.max_episode_steps="${MAX_EPISODE_STEPS}" \
  env.eval.max_steps_per_rollout_epoch="${MAX_EPISODE_STEPS}" \
  2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
