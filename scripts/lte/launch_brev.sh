#!/usr/bin/env bash
# Upload a credential-only .env, configuration, and LTE overlay before bootstrapping Brev.
set -Eeuo pipefail

[[ $# -ge 1 && $# -le 3 ]] || { echo "Usage: $0 <brev-instance> [smolvla|lte|both] [training-config]" >&2; exit 2; }
INSTANCE=$1
MODE=${2:-both}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RMBENCH_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE=${3:-"${RMBENCH_ROOT}/config/lte/battery_try.env"}
CREDENTIAL_FILE="${SCRIPT_DIR}/.env"
LTE_SOURCE="${LTE_SMOLVLA_SOURCE:-/home/sydney1/Repos/XPolicyLab/policy/LTE_SmolVLA}"
case "${MODE}" in smolvla|lte|both) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
command -v brev >/dev/null 2>&1 || { echo "brev CLI is required" >&2; exit 1; }
[[ -r "${CONFIG_FILE}" ]] || { echo "Training config not readable: ${CONFIG_FILE}" >&2; exit 1; }
[[ -r "${CREDENTIAL_FILE}" ]] || { echo "Credential file not readable: ${CREDENTIAL_FILE}" >&2; exit 1; }
[[ -d "${LTE_SOURCE}" ]] || { echo "LTE adapter source not found: ${LTE_SOURCE}" >&2; exit 1; }
set -a; source "${CONFIG_FILE}"; set +a
BOOTSTRAP_DIR=.lte-bootstrap

ARCHIVE="$(mktemp -p /tmp rmbench-lte-smolvla-XXXXXX.tar.gz)"
trap 'rm -f "${ARCHIVE}"' EXIT
tar --exclude='__pycache__' --exclude='.pytest_cache' --exclude='data' --exclude='checkpoints' \
  -czf "${ARCHIVE}" -C "$(dirname "$(dirname "${LTE_SOURCE}")")" "policy/LTE_SmolVLA"

brev exec "${INSTANCE}" "mkdir -p '${BOOTSTRAP_DIR}' && chmod 700 '${BOOTSTRAP_DIR}'"
brev copy "${CREDENTIAL_FILE}" "${INSTANCE}:${BOOTSTRAP_DIR}/credentials.env"
brev copy "${CONFIG_FILE}" "${INSTANCE}:${BOOTSTRAP_DIR}/training.env"
brev copy "${ARCHIVE}" "${INSTANCE}:${BOOTSTRAP_DIR}/lte_smolvla_overlay.tar.gz"
brev copy "${SCRIPT_DIR}/run_training.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/run_training.sh"
brev copy "${SCRIPT_DIR}/setup_remote.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/setup_remote.sh"
brev copy "${SCRIPT_DIR}/run_eval.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/run_eval.sh"
brev copy "${SCRIPT_DIR}/log_eval_wandb.py" "${INSTANCE}:${BOOTSTRAP_DIR}/log_eval_wandb.py"
brev copy "${SCRIPT_DIR}/run_checkpoint_evals.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/run_checkpoint_evals.sh"
brev copy "${SCRIPT_DIR}/lte_smolvla_cli_override.patch" "${INSTANCE}:${BOOTSTRAP_DIR}/lte_smolvla_cli_override.patch"
brev exec "${INSTANCE}" "chmod 600 '${BOOTSTRAP_DIR}/credentials.env'; chmod 700 '${BOOTSTRAP_DIR}/setup_remote.sh' '${BOOTSTRAP_DIR}/run_training.sh' '${BOOTSTRAP_DIR}/run_eval.sh' '${BOOTSTRAP_DIR}/run_checkpoint_evals.sh' '${BOOTSTRAP_DIR}/log_eval_wandb.py'; nohup bash '${BOOTSTRAP_DIR}/setup_remote.sh' '${BOOTSTRAP_DIR}' '${MODE}' '${BOOTSTRAP_DIR}/training.env' > '${BOOTSTRAP_DIR}/setup.log' 2>&1 < /dev/null & echo \"Remote setup PID: \$! (log: ~/${BOOTSTRAP_DIR}/setup.log)\""
