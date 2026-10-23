#!/usr/bin/env bash
# Bootstrap an RMBench dataset and launch an exact YAML-defined XPolicyLab training command.
set -Eeuo pipefail

[[ $# -eq 3 ]] || { echo "Usage: $0 <bootstrap-directory> <smolvla|lte> <config-file>" >&2; exit 2; }
BOOTSTRAP_DIR=$1
MODE=$2
CONFIG_SOURCE=$3
case "${MODE}" in smolvla|lte) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
[[ -r "${CONFIG_SOURCE}" ]] || { echo "Training configuration not readable: ${CONFIG_SOURCE}" >&2; exit 1; }
BOOTSTRAP_DIR="$(cd "${BOOTSTRAP_DIR}" && pwd)"
CONFIG_READER="${BOOTSTRAP_DIR}/training_config.py"
python3 "${CONFIG_READER}" "${CONFIG_SOURCE}" validate
config_get() { python3 "${CONFIG_READER}" "${CONFIG_SOURCE}" get "$1"; }

mkdir -p "${HOME}/.lte-bootstrap"
exec 9>"${HOME}/.lte-bootstrap/setup.lock"
flock -n 9 || { echo "Another LTE bootstrap is already running on this instance." >&2; exit 75; }

RMBENCH_ROOT="${HOME}/RMBench-lte"
RMBENCH_URL=https://github.com/alek5k/RMBench.git
RMBENCH_REF=main
XPOLICYLAB_URL=https://github.com/alek5k/XPolicyLab.git
XPOLICYLAB_REF=main
CONDA_BASE="${HOME}/miniforge3"
CONDA_ENV="$(config_get conda_env)"
CONDA_ENV="${CONDA_ENV:-smolvla}"
CREDENTIAL_FILE="${BOOTSTRAP_DIR}/credentials.env"

if [[ ! -d "${RMBENCH_ROOT}/.git" ]]; then
  [[ ! -e "${RMBENCH_ROOT}" ]] || { echo "Refusing to reuse non-git path: ${RMBENCH_ROOT}" >&2; exit 1; }
  git clone --branch "${RMBENCH_REF}" "${RMBENCH_URL}" "${RMBENCH_ROOT}"
else
  [[ "$(git -C "${RMBENCH_ROOT}" remote get-url origin)" == "${RMBENCH_URL}" ]] || {
    echo "Existing RMBench workspace has a different origin." >&2; exit 1;
  }
fi
git -C "${RMBENCH_ROOT}" config submodule.XPolicyLab.url "${XPOLICYLAB_URL}"
git -C "${RMBENCH_ROOT}" submodule sync --recursive
git -C "${RMBENCH_ROOT}" submodule update --init --recursive XPolicyLab
XPOLICYLAB_ROOT="${RMBENCH_ROOT}/XPolicyLab"
XPL_STATUS="$(git -C "${XPOLICYLAB_ROOT}" status --porcelain --untracked-files=all)"
XPL_DIRTY="$(printf '%s\n' "${XPL_STATUS}" | grep -vE '^\?\? policy/LTE_SmolVLA/' || true)"
[[ -z "${XPL_DIRTY}" ]] || { echo "Refusing to replace a dirty XPolicyLab workspace." >&2; exit 1; }
# Older launches installed LTE_SmolVLA as an untracked overlay. It became
# tracked on XPolicyLab main, so remove only that known overlay before checkout
# to prevent Git from rejecting the transition as an untracked-file conflict.
if printf '%s\n' "${XPL_STATUS}" | grep -qE '^\?\? policy/LTE_SmolVLA/'; then
  rm -rf "${XPOLICYLAB_ROOT}/policy/LTE_SmolVLA"
fi
git -C "${XPOLICYLAB_ROOT}" remote set-url origin "${XPOLICYLAB_URL}"
git -C "${XPOLICYLAB_ROOT}" fetch origin "${XPOLICYLAB_REF}"
git -C "${XPOLICYLAB_ROOT}" checkout --detach "origin/${XPOLICYLAB_REF}"

mkdir -p "${RMBENCH_ROOT}/config/lte" "${RMBENCH_ROOT}/scripts/lte"
CONFIG_REMOTE="${RMBENCH_ROOT}/config/lte/$(basename "${CONFIG_SOURCE}")"
install -m 644 "${CONFIG_SOURCE}" "${CONFIG_REMOTE}"
install -m 755 "${CONFIG_READER}" "${RMBENCH_ROOT}/scripts/lte/training_config.py"
[[ ! -r "${CREDENTIAL_FILE}" ]] || install -m 600 "${CREDENTIAL_FILE}" "${RMBENCH_ROOT}/scripts/lte/.env"

if [[ "${MODE}" == lte ]]; then
  [[ -f "${BOOTSTRAP_DIR}/lte_smolvla_overlay.tar.gz" ]] || { echo "LTE overlay was not uploaded." >&2; exit 1; }
  tar -xzf "${BOOTSTRAP_DIR}/lte_smolvla_overlay.tar.gz" -C "${XPOLICYLAB_ROOT}"
fi

if [[ ! -x "${CONDA_BASE}/bin/conda" ]]; then
  installer="${BOOTSTRAP_DIR}/Miniforge3-Linux-x86_64.sh"
  [[ -f "${installer}" ]] || curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "${installer}"
  bash "${installer}" -b -p "${CONDA_BASE}"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "${CONDA_ENV}"; then conda create -y -n "${CONDA_ENV}" python=3.10 pip; fi
export CONDA_BASE CONDA_ROOT="${CONDA_BASE}" SMOVLA_CONDA_ENV="${CONDA_ENV}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  command -v sudo >/dev/null 2>&1 || { echo "ffmpeg is required for LeRobot video conversion." >&2; exit 1; }
  sudo -n apt-get update && sudo -n apt-get install -y ffmpeg || { echo "Unable to install ffmpeg non-interactively." >&2; exit 1; }
fi

if [[ "${MODE}" == lte ]]; then
  bash "${XPOLICYLAB_ROOT}/policy/LTE_SmolVLA/install.sh"
else
  bash "${XPOLICYLAB_ROOT}/policy/SmolVLA/install.sh"
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
TASK="$(config_get task)"
ENV_CFG="$(config_get env_cfg)"
DATASET_REPO_ID="$(python3 "${CONFIG_READER}" "${CONFIG_SOURCE}" args | python3 -c 'import sys; print(next(line.split("=", 1)[1].strip() for line in sys.stdin if line.startswith("--dataset.repo_id=")))')"
DOWNLOAD_REPOS="$(config_get dataset.repo_ids)"
DOWNLOAD_DATASET="$(config_get dataset.download)"
HF_LEROBOT_HOME="${HOME}/.cache/huggingface/lerobot"
export PYTHONPATH="${RMBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME

if [[ -n "${DOWNLOAD_REPOS}" ]]; then
  IFS=',' read -r -a REPOS <<<"${DOWNLOAD_REPOS}"
  for repo in "${REPOS[@]}"; do
    root="${HF_LEROBOT_HOME}/${repo}"
    if [[ ! -f "${root}/meta/info.json" ]]; then
      python - "${repo}" "${root}" <<'PY'
from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
PY
    fi
    [[ -f "${root}/meta/info.json" ]] || { echo "Dataset download failed: ${repo}" >&2; exit 1; }
  done
elif [[ "${DOWNLOAD_DATASET}" == true ]]; then
  DATASET_ROOT="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
  if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    python - "${DATASET_REPO_ID}" "${DATASET_ROOT}" <<'PY'
from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
PY
  fi
  [[ -f "${DATASET_ROOT}/meta/info.json" ]] || { echo "Dataset download failed: ${DATASET_REPO_ID}" >&2; exit 1; }
elif [[ "${DATASET_REPO_ID}" == local/* ]]; then
  DATASET_ROOT="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
  DATASET_INFO="${DATASET_ROOT}/meta/info.json"
  DATA_ROOT="${RMBENCH_ROOT}/data/demo_clean/${TASK}/${ENV_CFG}/data"
  if [[ ! -d "${DATA_ROOT}" ]]; then (cd "${RMBENCH_ROOT}" && bash scripts/_download_data.sh "${TASK}"); fi
  EXPECTED_EPISODES="$(find "${DATA_ROOT}" -maxdepth 1 -type f \( -name '*.hdf5' -o -name '*.h5' \) | wc -l)"
  (( EXPECTED_EPISODES > 0 )) || { echo "No source HDF5 files found in ${DATA_ROOT}" >&2; exit 1; }
  (( EXPECTED_EPISODES <= 200 )) || EXPECTED_EPISODES=200
  COMPLETED_EPISODES=0
  if [[ -f "${DATASET_INFO}" ]]; then
    COMPLETED_EPISODES="$(python - "${DATASET_INFO}" <<'PY'
import json, sys
try: print(int(json.load(open(sys.argv[1])).get("total_episodes", 0)))
except Exception: print(0)
PY
)"
  fi
  if [[ "${COMPLETED_EPISODES}" != "${EXPECTED_EPISODES}" ]]; then
    [[ ! -e "${DATASET_ROOT}" ]] || mv "${DATASET_ROOT}" "${DATASET_ROOT}.partial-$(date +%Y%m%d-%H%M%S)"
    cd "${XPOLICYLAB_ROOT}"
    python scripts/transform_lerobot_v30_format.py "demo_clean.${TASK}.${ENV_CFG}" \
      --data_type RoboDojo --repo_id "${DATASET_REPO_ID}" --max_episode 200
  fi
  [[ -f "${DATASET_INFO}" ]] || { echo "LeRobot conversion failed." >&2; exit 1; }
fi

install -m 700 "${BOOTSTRAP_DIR}/run_training.sh" "${RMBENCH_ROOT}/.brev-run-smolvla.sh"
install -m 700 "${BOOTSTRAP_DIR}/run_eval.sh" "${RMBENCH_ROOT}/scripts/lte/run_eval.sh"
install -m 700 "${BOOTSTRAP_DIR}/log_eval_wandb.py" "${RMBENCH_ROOT}/scripts/lte/log_eval_wandb.py"
install -m 700 "${BOOTSTRAP_DIR}/run_checkpoint_evals.sh" "${RMBENCH_ROOT}/scripts/lte/run_checkpoint_evals.sh"
mkdir -p "${RMBENCH_ROOT}/logs"
if [[ "$(config_get wait_for_evals)" == true ]]; then
  while screen -ls 2>/dev/null | grep -Eq '[0-9]+\.(smolvla|lte)-eval-'; do sleep 60; done
fi
SESSION="smolvla-${MODE}-${TASK}-$(date +%Y%m%d-%H%M%S)"
LOGFILE="${RMBENCH_ROOT}/logs/${SESSION}.log"
if command -v screen >/dev/null 2>&1; then
  screen -L -Logfile "${LOGFILE}" -dmS "${SESSION}" bash "${RMBENCH_ROOT}/.brev-run-smolvla.sh" "${RMBENCH_ROOT}" "${MODE}" "${CONFIG_REMOTE}"
  echo "Training launched in screen: ${SESSION}"
else
  nohup bash "${RMBENCH_ROOT}/.brev-run-smolvla.sh" "${RMBENCH_ROOT}" "${MODE}" "${CONFIG_REMOTE}" >"${LOGFILE}" 2>&1 < /dev/null &
  echo "Training launched as PID $! (screen unavailable)"
fi
printf 'Log: %s\nRemote root: %s\n' "${LOGFILE}" "${RMBENCH_ROOT}"
