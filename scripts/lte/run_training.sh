#!/usr/bin/env bash
# Runs on Brev after RMBench HDF5 has been converted to a local LeRobot-v3 cache.
set -Eeuo pipefail

[[ $# -eq 3 ]] || { echo "Usage: $0 <RMBench-root> <smolvla|lte|both> <training-config>" >&2; exit 2; }
RMBENCH_ROOT=$1
MODE=$2
CONFIG_FILE=$3
case "${MODE}" in smolvla|lte|both) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
[[ -d "${RMBENCH_ROOT}/XPolicyLab" ]] || { echo "RMBench checkout not found: ${RMBENCH_ROOT}" >&2; exit 1; }
[[ -r "${CONFIG_FILE}" ]] || { echo "Training configuration not readable: ${CONFIG_FILE}" >&2; exit 1; }

CREDENTIAL_FILE="${RMBENCH_ROOT}/scripts/lte/.env"
[[ -r "${CREDENTIAL_FILE}" ]] && { set -a; source "${CREDENTIAL_FILE}"; set +a; }
# Credentials are intentionally sourced first.  The versioned task config is
# authoritative for all non-secret controls, which also protects a remote
# workspace that still has obsolete training variables in its old .env.
set -a; source "${CONFIG_FILE}"; set +a
[[ -z "${RMBENCH_TRAIN_STEPS_OVERRIDE:-}" ]] || TRAIN_STEPS="${RMBENCH_TRAIN_STEPS_OVERRIDE}"
[[ -z "${RMBENCH_RUN_TAG_OVERRIDE:-}" ]] || RMBENCH_RUN_TAG="${RMBENCH_RUN_TAG_OVERRIDE}"

XPL_ROOT="${RMBENCH_ROOT}/XPolicyLab"
POLICY_ROOT="${XPL_ROOT}/policy"
source "${POLICY_ROOT}/SmolVLA/conda_init.sh"
smolvla_activate_conda "${SMOLVLA_CONDA_ENV:-smolvla}"
export PYTHONPATH="${RMBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME="${HOME}/.cache/huggingface/lerobot"

# A local run may use the editable LTE adapter checkout rather than RMBench's
# pinned XPolicyLab submodule. Brev overlays the adapter into the submodule,
# so this remains unset there.
if [[ -n "${LTE_SMOLVLA_SOURCE:-}" ]]; then
  [[ -f "${LTE_SMOLVLA_SOURCE}/train.py" ]] || {
    echo "LTE_SMOLVLA_SOURCE is not an LTE_SmolVLA source directory: ${LTE_SMOLVLA_SOURCE}" >&2
    exit 1
  }
  LTE_PACKAGE_PARENT="$(cd "${LTE_SMOLVLA_SOURCE}/../../.." && pwd)"
  export PYTHONPATH="${LTE_PACKAGE_PARENT}:${PYTHONPATH}"
  # Python inserts its working directory before PYTHONPATH. Leave RMBench so
  # its XPolicyLab submodule cannot shadow the editable adapter checkout.
  cd "${LTE_PACKAGE_PARENT}"
fi

BENCH=RMBench
TASK="${RMBENCH_TASK:?RMBENCH_TASK is required}"
ROBOT="${RMBENCH_ENV_CFG:?RMBENCH_ENV_CFG is required}"
ACTION=joint
SEED="${RMBENCH_SEED:-0}"
RUN_TAG="${RMBENCH_RUN_TAG:-}"
NUM_PROCESSES="${RMBENCH_NUM_PROCESSES:-1}"
if [[ -n "${RMBENCH_DATASET_REPO_IDS:-}" ]]; then
  DATASET_REPO_ID="lte-multi:${RMBENCH_DATASET_REPO_IDS}"
  DATASET_ROOT="${RMBENCH_DATASET_ROOT:-${HF_LEROBOT_HOME}}"
  IFS=',' read -r -a DATASET_REPOS <<<"${RMBENCH_DATASET_REPO_IDS}"
  for repo in "${DATASET_REPOS[@]}"; do
    [[ -d "${DATASET_ROOT}/${repo}" ]] || { echo "LeRobot dataset is missing: ${DATASET_ROOT}/${repo}" >&2; exit 1; }
  done
else
  DATASET_REPO_ID="${RMBENCH_DATASET_REPO_ID:-local/rmbench-${TASK}-${ROBOT}-v30}"
  DATASET_ROOT="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
  [[ -d "${DATASET_ROOT}" ]] || { echo "Converted LeRobot dataset is missing: ${DATASET_ROOT}" >&2; exit 1; }
fi
[[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "RMBENCH_NUM_PROCESSES must be a positive integer" >&2
  exit 2
}

# `torchrun` exposes one CUDA device to each Accelerate process.  Keep the
# historical one-GPU behaviour by default, but make an explicit multi-GPU
# configuration safe: fail before loading the model if the Brev machine has
# fewer GPUs than requested.
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l)"
[[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && (( GPU_COUNT >= NUM_PROCESSES )) || {
  echo "Requested ${NUM_PROCESSES} GPU processes but only ${GPU_COUNT} CUDA devices are available." >&2
  exit 1
}
if [[ -n "${RMBENCH_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${RMBENCH_CUDA_VISIBLE_DEVICES}"
else
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((NUM_PROCESSES - 1))")"
  export CUDA_VISIBLE_DEVICES
fi

: "${SMOLVLA_BASE_POLICY:?SMOLVLA_BASE_POLICY is required}"
: "${RMBENCH_IMAGE_HEIGHT:?RMBENCH_IMAGE_HEIGHT is required}"
: "${RMBENCH_IMAGE_WIDTH:?RMBENCH_IMAGE_WIDTH is required}"
: "${RMBENCH_STATE_ACTION_DIM:?RMBENCH_STATE_ACTION_DIM is required}"

# This is the common SmolVLA configuration.  LTE receives it through its
# base-policy inheritance while its loader creates the extra temporal modules.
COMMON_POLICY_ARGS=(
  --policy.device=cuda --policy.push_to_hub=false
  --policy.chunk_size="${SMOLVLA_CHUNK_SIZE:?SMOLVLA_CHUNK_SIZE is required}"
  --policy.n_action_steps="${SMOLVLA_N_ACTION_STEPS:?SMOLVLA_N_ACTION_STEPS is required}"
  --policy.n_obs_steps="${SMOLVLA_N_OBS_STEPS:?SMOLVLA_N_OBS_STEPS is required}"
  --policy.train_expert_only="${SMOLVLA_TRAIN_EXPERT_ONLY:?SMOLVLA_TRAIN_EXPERT_ONLY is required}"
  --policy.train_state_proj="${SMOLVLA_TRAIN_STATE_PROJ:?SMOLVLA_TRAIN_STATE_PROJ is required}"
  --policy.freeze_vision_encoder="${SMOLVLA_FREEZE_VISION_ENCODER:?SMOLVLA_FREEZE_VISION_ENCODER is required}"
  --policy.optimizer_lr="${SMOLVLA_OPTIMIZER_LR:?SMOLVLA_OPTIMIZER_LR is required}"
  --policy.optimizer_weight_decay="${SMOLVLA_OPTIMIZER_WEIGHT_DECAY:-1e-10}"
  --policy.optimizer_grad_clip_norm="${SMOLVLA_OPTIMIZER_GRAD_CLIP_NORM:-10}"
  --policy.scheduler_warmup_steps="${SMOLVLA_SCHEDULER_WARMUP_STEPS:?SMOLVLA_SCHEDULER_WARMUP_STEPS is required}"
  --policy.scheduler_decay_steps="${SMOLVLA_SCHEDULER_DECAY_STEPS:?SMOLVLA_SCHEDULER_DECAY_STEPS is required}"
  --policy.scheduler_decay_lr="${SMOLVLA_SCHEDULER_DECAY_LR:?SMOLVLA_SCHEDULER_DECAY_LR is required}"
  --policy.adapt_to_pi_aloha="${SMOLVLA_ADAPT_TO_PI_ALOHA:?SMOLVLA_ADAPT_TO_PI_ALOHA is required}"
  --policy.use_delta_joint_actions_aloha="${SMOLVLA_USE_DELTA_JOINT_ACTIONS_ALOHA:?SMOLVLA_USE_DELTA_JOINT_ACTIONS_ALOHA is required}"
)
if [[ "${RMBENCH_SINGLE_CAMERA:-false}" == true ]]; then
FEATURE_ARGS=(
  --policy.input_features="{\"observation.state\":{\"type\":\"STATE\",\"shape\":[${RMBENCH_STATE_ACTION_DIM}]},\"observation.images.camera1\":{\"type\":\"VISUAL\",\"shape\":[3,${RMBENCH_IMAGE_HEIGHT},${RMBENCH_IMAGE_WIDTH}]}}"
  --policy.output_features="{\"action\":{\"type\":\"ACTION\",\"shape\":[${RMBENCH_STATE_ACTION_DIM}]}}"
)
RENAME_MAP='{"observation.image":"observation.images.camera1"}'
else
FEATURE_ARGS=(
  --policy.input_features="{\"observation.state\":{\"type\":\"STATE\",\"shape\":[${RMBENCH_STATE_ACTION_DIM}]},\"observation.images.camera1\":{\"type\":\"VISUAL\",\"shape\":[3,${RMBENCH_IMAGE_HEIGHT},${RMBENCH_IMAGE_WIDTH}]},\"observation.images.camera2\":{\"type\":\"VISUAL\",\"shape\":[3,${RMBENCH_IMAGE_HEIGHT},${RMBENCH_IMAGE_WIDTH}]},\"observation.images.camera3\":{\"type\":\"VISUAL\",\"shape\":[3,${RMBENCH_IMAGE_HEIGHT},${RMBENCH_IMAGE_WIDTH}]}}"
  --policy.output_features="{\"action\":{\"type\":\"ACTION\",\"shape\":[${RMBENCH_STATE_ACTION_DIM}]}}"
)
RENAME_MAP='{"observation.images.cam_high":"observation.images.camera1","observation.images.cam_left_wrist":"observation.images.camera2","observation.images.cam_right_wrist":"observation.images.camera3"}'
fi

run_directory() {
  local policy_root=$1
  local directory="${policy_root}/checkpoints"
  [[ -z "${RUN_TAG}" ]] || directory+="/${RUN_TAG}"
  printf '%s/%s-%s-%s-%s-%s\n' "${directory}" "${BENCH}" "${TASK}" "${ROBOT}" "${ACTION}" "${SEED}"
}

job_name() {
  local policy=$1
  local steps="${TRAIN_STEPS:-100000}"
  local steps_label
  if (( steps % 1000 == 0 )); then
    steps_label="$((steps / 1000))k"
  else
    steps_label="${steps}steps"
  fi
  case "${policy}" in
    smolvla) printf '%s\n' "${TASK}_smolvla_${steps_label}" ;;
    lte) printf '%s\n' "${TASK}_lte_smolvla_${steps_label}" ;;
    *) echo "Unknown W&B policy label: ${policy}" >&2; return 2 ;;
  esac
}

WDB_ENABLED=false; [[ -n "${WANDB_API_KEY:-}" && "${RMBENCH_DISABLE_WANDB:-false}" != true ]] && WDB_ENABLED=true
WANDB_ARGS=(--wandb.enable="${WDB_ENABLED}")
if [[ "${WDB_ENABLED}" == true ]]; then
  : "${WANDB_ENTITY:?WANDB_ENTITY is required when WANDB_API_KEY is set}"
  : "${WANDB_PROJECT:?WANDB_PROJECT is required when WANDB_API_KEY is set}"
  WANDB_ARGS+=(--wandb.entity="${WANDB_ENTITY}" --wandb.project="${WANDB_PROJECT}")
fi

launch_module() {
  if (( NUM_PROCESSES == 1 )); then
    python -m "$@"
  else
    torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_PROCESSES}" -m "$@"
  fi
}

run_baseline() {
  local output_dir
  output_dir="$(run_directory "${POLICY_ROOT}/SmolVLA")"
  echo "[RMBench SmolVLA] dataset=${DATASET_REPO_ID} output=${output_dir}"
  if (( NUM_PROCESSES == 1 )); then
    baseline_launcher=(lerobot-train)
  else
    baseline_launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_PROCESSES}" -m lerobot.scripts.lerobot_train)
  fi
  "${baseline_launcher[@]}" \
    --policy.path="${SMOLVLA_BASE_POLICY}" "${COMMON_POLICY_ARGS[@]}" "${FEATURE_ARGS[@]}" \
    --dataset.repo_id="${DATASET_REPO_ID}" --dataset.root="${DATASET_ROOT}" --dataset.video_backend=pyav \
    --output_dir="${output_dir}" --job_name="$(job_name smolvla)" \
    --batch_size="${SMOLVLA_BATCH_SIZE:?SMOLVLA_BATCH_SIZE is required}" \
    --steps="${TRAIN_STEPS:-100000}" --save_freq="${TRAIN_SAVE_FREQ:-10000}" --log_freq="${TRAIN_LOG_FREQ:-10}" \
    --num_workers="${TRAIN_NUM_WORKERS:-8}" "${WANDB_ARGS[@]}" --eval_freq=0 \
    --rename_map="${RENAME_MAP}" \
    --seed="${SEED}"
}

run_lte() {
  [[ "${ACTION}" == joint ]] || { echo "LTE_SmolVLA supports joint actions only" >&2; exit 2; }
  local output_dir
  output_dir="$(run_directory "${POLICY_ROOT}/LTE_SmolVLA")"
  echo "[RMBench LTE_SmolVLA] dataset=${DATASET_REPO_ID} output=${output_dir}"
  launch_module XPolicyLab.policy.LTE_SmolVLA.train \
    --policy.type=lte_smolvla --policy.base_policy_path="${SMOLVLA_BASE_POLICY}" "${COMMON_POLICY_ARGS[@]}" "${FEATURE_ARGS[@]}" \
    --dataset.repo_id="${DATASET_REPO_ID}" --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav --dataset.image_transforms.enable=false \
    --output_dir="${output_dir}" --job_name="$(job_name lte)" --batch_size="${LTE_BATCH_SIZE:-8}" \
    --policy.max_causal_history_length="${LTE_MAX_CAUSAL_HISTORY_LENGTH:?LTE_MAX_CAUSAL_HISTORY_LENGTH is required}" \
    --policy.differential_lr="${LTE_DIFFERENTIAL_LR:-false}" \
    --policy.backbone_lr_scale="${LTE_BACKBONE_LR_SCALE:-0.1}" \
    --policy.temporal_embedding_cache_build_batch_size="${LTE_TEMPORAL_EMBEDDING_CACHE_BUILD_BATCH_SIZE:-32}" \
    --policy.temporal_encode_chunk_size="${LTE_TEMPORAL_ENCODE_CHUNK_SIZE:-32}" \
    --policy.temporal_embedding_cache_refresh_steps="${LTE_TEMPORAL_EMBEDDING_CACHE_REFRESH_STEPS:-5000}" \
    --policy.temporal_embedding_cache_enabled="${LTE_TEMPORAL_EMBEDDING_CACHE_ENABLED:-true}" \
    --policy.temporal_image_cache_on_gpu="${LTE_TEMPORAL_IMAGE_CACHE_ON_GPU:-false}" \
    --policy.temporal_image_cache_in_ram="${LTE_TEMPORAL_IMAGE_CACHE_IN_RAM:-false}" \
    --policy.temporal_image_cache_build_num_threads="${LTE_TEMPORAL_IMAGE_CACHE_BUILD_NUM_THREADS:-32}" \
    --policy.temporal_image_cache_build_chunk_size="${LTE_TEMPORAL_IMAGE_CACHE_BUILD_CHUNK_SIZE:-2048}" \
    --policy.temporal_query_history_only="${LTE_TEMPORAL_QUERY_HISTORY_ONLY:-false}" \
    --policy.temporal_embedding_normalize="${LTE_TEMPORAL_EMBEDDING_NORMALIZE:-true}" \
    --steps="${TRAIN_STEPS:-100000}" --save_freq="${TRAIN_SAVE_FREQ:-10000}" --log_freq="${TRAIN_LOG_FREQ:-10}" \
    --num_workers="${TRAIN_NUM_WORKERS:-4}" "${WANDB_ARGS[@]}" --eval_freq=0 --seed="${SEED}" \
    --rename_map="${RENAME_MAP}"
}

case "${MODE}" in
  smolvla) run_baseline ;;
  lte) run_lte ;;
  both) run_baseline; run_lte ;;
esac
