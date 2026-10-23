#!/usr/bin/env bash
# Upload credential, YAML recipe, configuration reader, and LTE overlay before bootstrapping Brev.
set -Eeuo pipefail

[[ $# -eq 3 ]] || { echo "Usage: $0 <brev-instance> <smolvla|lte> <training-config.yaml>" >&2; exit 2; }
INSTANCE=$1
MODE=$2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RMBENCH_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE=$3
CREDENTIAL_FILE="${SCRIPT_DIR}/.env"
CONFIG_READER="${SCRIPT_DIR}/training_config.py"
case "${MODE}" in smolvla|lte) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
command -v brev >/dev/null 2>&1 || { echo "brev CLI is required" >&2; exit 1; }
[[ -r "${CONFIG_FILE}" ]] || { echo "Training config not readable: ${CONFIG_FILE}" >&2; exit 1; }
python3 "${CONFIG_READER}" "${CONFIG_FILE}" validate
LTE_SOURCE="$(python3 "${CONFIG_READER}" "${CONFIG_FILE}" get lte_smolvla_source)"
LTE_SOURCE="${LTE_SOURCE:-${LTE_SMOLVLA_SOURCE:-/home/sydney1/Repos/XPolicyLab/policy/LTE_SmolVLA}}"
[[ -d "${LTE_SOURCE}" ]] || { echo "LTE adapter source not found: ${LTE_SOURCE}" >&2; exit 1; }
BOOTSTRAP_DIR=.lte-bootstrap

ARCHIVE="$(mktemp -p /tmp rmbench-lte-smolvla-XXXXXX.tar.gz)"
trap 'rm -f "${ARCHIVE}"' EXIT
tar --exclude='__pycache__' --exclude='.pytest_cache' --exclude='data' --exclude='checkpoints' \
  -czf "${ARCHIVE}" -C "$(dirname "$(dirname "${LTE_SOURCE}")")" "policy/LTE_SmolVLA"

brev exec "${INSTANCE}" "mkdir -p '${BOOTSTRAP_DIR}' && chmod 700 '${BOOTSTRAP_DIR}'"
if [[ -r "${CREDENTIAL_FILE}" ]]; then
  brev copy "${CREDENTIAL_FILE}" "${INSTANCE}:${BOOTSTRAP_DIR}/credentials.env"
fi
REMOTE_CONFIG="${BOOTSTRAP_DIR}/$(basename "${CONFIG_FILE}")"
brev copy "${CONFIG_FILE}" "${INSTANCE}:${REMOTE_CONFIG}"
brev copy "${ARCHIVE}" "${INSTANCE}:${BOOTSTRAP_DIR}/lte_smolvla_overlay.tar.gz"
brev copy "${SCRIPT_DIR}/training_config.py" "${INSTANCE}:${BOOTSTRAP_DIR}/training_config.py"
brev copy "${SCRIPT_DIR}/run_training.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/run_training.sh"
brev copy "${SCRIPT_DIR}/setup_remote.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/setup_remote.sh"
brev copy "${SCRIPT_DIR}/run_eval.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/run_eval.sh"
brev copy "${SCRIPT_DIR}/log_eval_wandb.py" "${INSTANCE}:${BOOTSTRAP_DIR}/log_eval_wandb.py"
brev copy "${SCRIPT_DIR}/run_checkpoint_evals.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/run_checkpoint_evals.sh"
brev exec "${INSTANCE}" "test ! -f '${BOOTSTRAP_DIR}/credentials.env' || chmod 600 '${BOOTSTRAP_DIR}/credentials.env'; chmod 700 '${BOOTSTRAP_DIR}/setup_remote.sh' '${BOOTSTRAP_DIR}/run_training.sh' '${BOOTSTRAP_DIR}/training_config.py' '${BOOTSTRAP_DIR}/run_eval.sh' '${BOOTSTRAP_DIR}/run_checkpoint_evals.sh' '${BOOTSTRAP_DIR}/log_eval_wandb.py'; nohup bash '${BOOTSTRAP_DIR}/setup_remote.sh' '${BOOTSTRAP_DIR}' '${MODE}' '${REMOTE_CONFIG}' > '${BOOTSTRAP_DIR}/setup.log' 2>&1 < /dev/null & echo \"Remote setup PID: \$! (log: ~/${BOOTSTRAP_DIR}/setup.log)\""
