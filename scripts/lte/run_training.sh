#!/usr/bin/env bash
# Run XPolicyLab SmolVLA or LTE_SmolVLA with exact arguments from a YAML recipe.
set -Eeuo pipefail

usage() {
  echo "Usage: $0 [--dry-run] <RMBench-root> <smolvla|lte> <training-config.yaml>" >&2
  exit 2
}

DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=true
  shift
fi
[[ $# -eq 3 ]] || usage
RMBENCH_ROOT=$1
MODE=$2
CONFIG_FILE=$3
case "${MODE}" in smolvla|lte) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
[[ -d "${RMBENCH_ROOT}/XPolicyLab" ]] || { echo "RMBench checkout not found: ${RMBENCH_ROOT}" >&2; exit 1; }
[[ -r "${CONFIG_FILE}" ]] || { echo "Training configuration not readable: ${CONFIG_FILE}" >&2; exit 1; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_READER="${SCRIPT_DIR}/training_config.py"
if [[ ! -r "${CONFIG_READER}" ]]; then
  CONFIG_READER="${RMBENCH_ROOT}/scripts/lte/training_config.py"
fi
[[ -r "${CONFIG_READER}" ]] || { echo "YAML configuration reader is missing: ${CONFIG_READER}" >&2; exit 1; }
python3 "${CONFIG_READER}" "${CONFIG_FILE}" validate
config_get() { python3 "${CONFIG_READER}" "${CONFIG_FILE}" get "$1"; }

CREDENTIAL_FILE="${RMBENCH_ROOT}/scripts/lte/.env"
[[ -r "${CREDENTIAL_FILE}" ]] && { set -a; source "${CREDENTIAL_FILE}"; set +a; }

XPL_ROOT="${RMBENCH_ROOT}/XPolicyLab"
POLICY_ROOT="${XPL_ROOT}/policy"
if [[ "${DRY_RUN}" != true ]]; then
  source "${POLICY_ROOT}/SmolVLA/conda_init.sh"
  SMOLVLA_CONDA_ENV="$(config_get conda_env)"
  smolvla_activate_conda "${SMOLVLA_CONDA_ENV:-smolvla}"
fi
export PYTHONPATH="${RMBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME="${HOME}/.cache/huggingface/lerobot"

LTE_SMOLVLA_SOURCE="$(config_get lte_smolvla_source)"
if [[ -n "${LTE_SMOLVLA_SOURCE}" ]]; then
  [[ -f "${LTE_SMOLVLA_SOURCE}/train.py" ]] || {
    echo "lte_smolvla_source is not an LTE_SmolVLA source directory: ${LTE_SMOLVLA_SOURCE}" >&2
    exit 1
  }
  LTE_PACKAGE_PARENT="$(cd "${LTE_SMOLVLA_SOURCE}/../../.." && pwd)"
  export PYTHONPATH="${LTE_PACKAGE_PARENT}:${PYTHONPATH}"
  cd "${LTE_PACKAGE_PARENT}"
fi

NUM_PROCESSES="$(config_get num_processes)"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
[[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || { echo "launcher.num_processes must be a positive integer" >&2; exit 2; }
CUDA_DEVICES="$(config_get cuda_visible_devices)"
if [[ -n "${CUDA_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((NUM_PROCESSES - 1))")"
fi

if [[ "${DRY_RUN}" != true ]]; then
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l)"
  [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && (( GPU_COUNT >= NUM_PROCESSES )) || {
    echo "Requested ${NUM_PROCESSES} GPU processes but only ${GPU_COUNT} CUDA devices are available." >&2
    exit 1
  }
fi

print_or_run() {
  if [[ "${DRY_RUN}" == true ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

baseline_command() {
  mapfile -t TRAIN_ARGS < <(python3 "${CONFIG_READER}" "${CONFIG_FILE}" args)
  if (( NUM_PROCESSES == 1 )); then
    BASE_COMMAND=(lerobot-train)
  else
    BASE_COMMAND=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_PROCESSES}" -m lerobot.scripts.lerobot_train)
  fi
  print_or_run "${BASE_COMMAND[@]}" "${TRAIN_ARGS[@]}"
}

lte_command() {
  mapfile -t TRAIN_ARGS < <(python3 "${CONFIG_READER}" "${CONFIG_FILE}" args)
  if (( NUM_PROCESSES == 1 )); then
    LTE_COMMAND=(python -m XPolicyLab.policy.LTE_SmolVLA.train)
  else
    LTE_COMMAND=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_PROCESSES}" -m XPolicyLab.policy.LTE_SmolVLA.train)
  fi
  print_or_run "${LTE_COMMAND[@]}" "${TRAIN_ARGS[@]}"
}

case "${MODE}" in
  smolvla) baseline_command ;;
  lte) lte_command ;;
esac
