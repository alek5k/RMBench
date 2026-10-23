#!/usr/bin/env bash
# Start a config-driven simulator evaluation with a SmolVLA policy server.
set -Eeuo pipefail

[[ $# -ge 2 && $# -le 4 ]] || {
  echo "Usage: $0 <RMBench-root> <eval-config.yaml>" >&2
  echo "Legacy: $0 <RMBench-root> <smolvla|lte> [checkpoint] [episodes]" >&2
  exit 2
}
RMBENCH_ROOT="$(cd "$1" && pwd)"
export PATH="${HOME}/miniforge3/bin:${HOME}/miniconda3/bin:${PATH}"

if [[ "$2" == smolvla || "$2" == lte ]]; then
  MODE=$2
  case "${MODE}" in smolvla) POLICY=SmolVLA ;; lte) POLICY=LTE_SmolVLA ;; esac
  TASK=battery_try
  ENV_CFG=aloha_agilex
  CKPT_NAME=${3:-battery_try}
  POLICY_CONDA_ENV=smolvla
  EVAL_CONDA_ENV=rmbench-eval
  POLICY_GPU_ID=0
  ENV_GPU_ID=0
  ACTION_TYPE=joint
  TEST_NUM=${4:-100}
  NUM_WORKERS=1
  PROTOCOL=ws
  SEED=0
  TASK_CONFIG=demo_clean
  INSTRUCTION_TYPE=seen
  EXPERT_CHECK=true
  FREQUENCY=30
  MAX_SEED_ATTEMPTS=$((TEST_NUM * 50))
  WANDB_ENABLED=false
  WANDB_ENTITY_CONFIG=
  WANDB_PROJECT_CONFIG=
  WANDB_RUN_NAME=
  CONFIG_PATH="legacy command-line arguments"
else
  [[ $# -eq 2 ]] || { echo "Config-driven evaluation accepts exactly two arguments" >&2; exit 2; }
  CONFIG_PATH=$2
  if [[ "${CONFIG_PATH}" != /* ]]; then
    CONFIG_PATH="${RMBENCH_ROOT}/${CONFIG_PATH}"
  fi
  [[ -f "${CONFIG_PATH}" ]] || { echo "Evaluation config not found: ${CONFIG_PATH}" >&2; exit 2; }
  CONFIG_PYTHON=${RMBENCH_CONFIG_PYTHON:-python3}
  eval "$("${CONFIG_PYTHON}" "${RMBENCH_ROOT}/scripts/lte/eval_config.py" "${CONFIG_PATH}" shell)"
fi

[[ "${TEST_NUM}" =~ ^[1-9][0-9]*$ ]] || { echo "episodes must be a positive integer" >&2; exit 2; }

CHECKPOINT_LABEL="$(basename "${CKPT_NAME}")"
CHECKPOINT_PARENT="$(basename "$(dirname "${CKPT_NAME}")")"
if [[ "${CHECKPOINT_PARENT}" =~ ^0*[0-9]+$ ]]; then
  CHECKPOINT_STEP="$((10#${CHECKPOINT_PARENT}))"
else
  CHECKPOINT_STEP="${CHECKPOINT_LABEL}"
fi
RUN="${MODE}-eval-${TASK}-$(date +%Y%m%d-%H%M%S)"
LOG="${RMBENCH_ROOT}/logs/${RUN}.log"
WANDB_LOG="${RMBENCH_ROOT}/logs/${RUN}-wandb.log"
mkdir -p "${RMBENCH_ROOT}/logs"

CREDENTIAL_FILE="${RMBENCH_ROOT}/scripts/lte/.env"
if [[ -r "${CREDENTIAL_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CREDENTIAL_FILE}"
  set +a
fi
if [[ "${WANDB_ENABLED}" == true ]]; then
  : "${WANDB_API_KEY:?WANDB_API_KEY is required when wandb.enabled is true}"
  export WANDB_ENTITY="${WANDB_ENTITY_CONFIG:-${WANDB_ENTITY:-}}"
  export WANDB_PROJECT="${WANDB_PROJECT_CONFIG:-${WANDB_PROJECT:-}}"
  : "${WANDB_ENTITY:?wandb.entity is required when wandb.enabled is true}"
  : "${WANDB_PROJECT:?wandb.project is required when wandb.enabled is true}"
fi

ADDITIONAL_INFO="ckpt_name=${CKPT_NAME},action_type=${ACTION_TYPE},test_num=${TEST_NUM},num_workers=${NUM_WORKERS},task_config=${TASK_CONFIG},instruction_type=${INSTRUCTION_TYPE},expert_check=${EXPERT_CHECK},frequency=${FREQUENCY},max_seed_attempts=${MAX_SEED_ATTEMPTS}"
EVAL_COMMAND="
set -Eeuo pipefail
cd '${RMBENCH_ROOT}'
POLICY_DIR='XPolicyLab/policy/${POLICY}'
UTILS_DIR='XPolicyLab/utils'
POLICY_PORT=\$(bash \"\${UTILS_DIR}/get_free_port.sh\")
cleanup() { [[ -z \"\${SERVER_PID:-}\" ]] || kill \"\${SERVER_PID}\" 2>/dev/null || true; }
trap cleanup EXIT
bash \"\${POLICY_DIR}/setup_eval_policy_server.sh\" RMBench '${TASK}' '${CKPT_NAME}' '${ENV_CFG}' '${ACTION_TYPE}' '${SEED}' '${POLICY_GPU_ID}' '${POLICY_CONDA_ENV}' \"\${POLICY_PORT}\" &
SERVER_PID=\$!
bash \"\${UTILS_DIR}/wait_for_policy_server.sh\" localhost \"\${POLICY_PORT}\" \"\${SERVER_PID}\" 'Policy server' 1200
bash \"\${POLICY_DIR}/setup_eval_env_client.sh\" RMBench '${TASK}' '${CKPT_NAME}' '${ENV_CFG}' '${ACTION_TYPE}' '${SEED}' '${ENV_GPU_ID}' '${EVAL_CONDA_ENV}' '${ADDITIONAL_INFO}' \"\${POLICY_PORT}\" localhost '${PROTOCOL}'
"
screen -L -Logfile "${LOG}" -dmS "${RUN}" bash -lc "${EVAL_COMMAND}"
printf 'Evaluation launched in screen: %s\nLog: %s\nConfig: %s\nWorkers: %s\n' \
  "${RUN}" "${LOG}" "${CONFIG_PATH}" "${NUM_WORKERS}"

if [[ "${WANDB_ENABLED}" == true ]]; then
  if [[ -z "${WANDB_RUN_NAME}" ]]; then
    WANDB_RUN_NAME="${TASK}_${MODE}_eval_${CHECKPOINT_STEP}"
  fi
  screen -L -Logfile "${WANDB_LOG}" -dmS "${RUN}-wandb" bash -lc \
    "cd '${RMBENCH_ROOT}'; exec '${HOME}/miniforge3/envs/${POLICY_CONDA_ENV}/bin/python' scripts/lte/log_eval_wandb.py --log-file '${LOG}' --mode '${MODE}' --task '${TASK}' --env-cfg '${ENV_CFG}' --checkpoint '${CKPT_NAME}' --seed '${SEED}' --run-name '${WANDB_RUN_NAME}' --screen-session '${RUN}'"
  printf 'W&B evaluation logger: %s-wandb\nW&B log: %s\n' "${RUN}" "${WANDB_LOG}"
fi
