#!/usr/bin/env bash
# Start a simulator evaluation with a SmolVLA policy server and separate RMBench client.
set -Eeuo pipefail

[[ $# -ge 2 && $# -le 4 ]] || { echo "Usage: $0 <RMBench-root> <smolvla|lte> [checkpoint-path-or-name] [episodes]" >&2; exit 2; }
RMBENCH_ROOT="$(cd "$1" && pwd)"
MODE="$2"
case "${MODE}" in smolvla) POLICY=SmolVLA ;; lte) POLICY=LTE_SmolVLA ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac

TASK=battery_try
ENV_CFG=aloha_agilex
CKPT_NAME="${3:-battery_try}"
TEST_NUM="${4:-100}"
[[ "${TEST_NUM}" =~ ^[1-9][0-9]*$ ]] || { echo "episodes must be a positive integer" >&2; exit 2; }
CHECKPOINT_LABEL="$(basename "${CKPT_NAME}")"
CHECKPOINT_PARENT="$(basename "$(dirname "${CKPT_NAME}")")"
if [[ "${CHECKPOINT_PARENT}" =~ ^0*[0-9]+$ ]]; then
  CHECKPOINT_STEP="$((10#${CHECKPOINT_PARENT}))"
else
  CHECKPOINT_STEP="${CHECKPOINT_LABEL}"
fi
case "${MODE}" in
  smolvla) WANDB_POLICY=smolvla ;;
  lte) WANDB_POLICY=lte_smolvla ;;
esac
RUN="${MODE}-eval-${TASK}-$(date +%Y%m%d-%H%M%S)"
LOG="${RMBENCH_ROOT}/logs/${RUN}.log"
WANDB_LOG="${RMBENCH_ROOT}/logs/${RUN}-wandb.log"
mkdir -p "${RMBENCH_ROOT}/logs"

export PATH="${HOME}/miniforge3/bin:${PATH}"
WDB_ENABLED=false
CREDENTIAL_FILE="${RMBENCH_ROOT}/scripts/lte/.env"
if [[ -r "${CREDENTIAL_FILE}" ]]; then
  set -a; source "${CREDENTIAL_FILE}"; set +a
  [[ -n "${WANDB_API_KEY:-}" ]] && WDB_ENABLED=true
fi
if [[ "${WDB_ENABLED}" == true ]]; then
  : "${WANDB_ENTITY:?WANDB_ENTITY is required when WANDB_API_KEY is set}"
  : "${WANDB_PROJECT:?WANDB_PROJECT is required when WANDB_API_KEY is set}"
fi

EVAL_COMMAND="
set -Eeuo pipefail
cd '${RMBENCH_ROOT}'
POLICY_DIR='XPolicyLab/policy/${POLICY}'
UTILS_DIR='XPolicyLab/utils'
POLICY_PORT=\$(bash \"\${UTILS_DIR}/get_free_port.sh\")
cleanup() { [[ -z \"\${SERVER_PID:-}\" ]] || kill \"\${SERVER_PID}\" 2>/dev/null || true; }
trap cleanup EXIT
bash \"\${POLICY_DIR}/setup_eval_policy_server.sh\" RMBench '${TASK}' '${CKPT_NAME}' '${ENV_CFG}' joint 0 0 smolvla \"\${POLICY_PORT}\" &
SERVER_PID=\$!
bash \"\${UTILS_DIR}/wait_for_policy_server.sh\" localhost \"\${POLICY_PORT}\" \"\${SERVER_PID}\" 'Policy server' 1200
bash \"\${POLICY_DIR}/setup_eval_env_client.sh\" RMBench '${TASK}' '${CKPT_NAME}' '${ENV_CFG}' joint 0 0 rmbench-eval 'ckpt_name=${CKPT_NAME},action_type=joint,test_num=${TEST_NUM}' \"\${POLICY_PORT}\" localhost
"
screen -L -Logfile "${LOG}" -dmS "${RUN}" bash -lc "${EVAL_COMMAND}"
printf 'Evaluation launched in screen: %s\nLog: %s\n' "${RUN}" "${LOG}"

if [[ "${WDB_ENABLED}" == true ]]; then
  WANDB_RUN="${TASK}_${WANDB_POLICY}_eval_${CHECKPOINT_STEP}"
  screen -L -Logfile "${WANDB_LOG}" -dmS "${RUN}-wandb" bash -lc \
    "cd '${RMBENCH_ROOT}'; exec '${HOME}/miniforge3/envs/smolvla/bin/python' scripts/lte/log_eval_wandb.py --log-file '${LOG}' --mode '${MODE}' --task '${TASK}' --env-cfg '${ENV_CFG}' --checkpoint '${CKPT_NAME}' --seed 0 --run-name '${WANDB_RUN}' --screen-session '${RUN}'"
  printf 'W&B evaluation logger: %s-wandb\nW&B log: %s\n' "${RUN}" "${WANDB_LOG}"
fi
