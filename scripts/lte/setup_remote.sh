#!/usr/bin/env bash
# Bootstrap an RMBench dataset, convert a task locally when requested, and start training.
set -Eeuo pipefail

[[ $# -eq 3 ]] || { echo "Usage: $0 <bootstrap-directory> <smolvla|lte|both> <config-file>" >&2; exit 2; }
BOOTSTRAP_DIR=$1
MODE=$2
CONFIG_SOURCE=$3
case "${MODE}" in smolvla|lte|both) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
[[ -r "${CONFIG_SOURCE}" ]] || { echo "Training configuration not readable: ${CONFIG_SOURCE}" >&2; exit 1; }
set -a; source "${CONFIG_SOURCE}"; set +a
BOOTSTRAP_DIR="$(cd "${BOOTSTRAP_DIR}" && pwd)"

# Brev may reconnect a dropped command. Keep a second invocation from installing
# into or converting the same workspace concurrently.
mkdir -p "${HOME}/.lte-bootstrap"
exec 9>"${HOME}/.lte-bootstrap/setup.lock"
flock -n 9 || {
  echo "Another LTE bootstrap is already running on this instance." >&2
  exit 75
}

RMBENCH_ROOT="${HOME}/RMBench-lte"
RMBENCH_URL=https://github.com/alek5k/RMBench.git
RMBENCH_REF=main
XPOLICYLAB_URL=https://github.com/alek5k/XPolicyLab.git
XPOLICYLAB_REF=main
CONDA_BASE="${HOME}/miniforge3"
CONDA_ENV=smolvla
CREDENTIAL_FILE="${BOOTSTRAP_DIR}/credentials.env"

if [[ ! -d "${RMBENCH_ROOT}/.git" ]]; then
  [[ ! -e "${RMBENCH_ROOT}" ]] || { echo "Refusing to reuse non-git path: ${RMBENCH_ROOT}" >&2; exit 1; }
  git clone --branch "${RMBENCH_REF}" "${RMBENCH_URL}" "${RMBENCH_ROOT}"
else
  [[ "$(git -C "${RMBENCH_ROOT}" remote get-url origin)" == "${RMBENCH_URL}" ]] || {
    echo "Existing RMBench workspace has a different origin; choose a new RMBENCH_REMOTE_DIR." >&2; exit 1;
  }
fi
git -C "${RMBENCH_ROOT}" config submodule.XPolicyLab.url "${XPOLICYLAB_URL}"
git -C "${RMBENCH_ROOT}" submodule sync --recursive
git -C "${RMBENCH_ROOT}" submodule update --init --recursive XPolicyLab
XPOLICYLAB_ROOT="${RMBENCH_ROOT}/XPolicyLab"
XPL_DIRTY="$(git -C "${XPOLICYLAB_ROOT}" status --porcelain --untracked-files=all | grep -vE '^\?\? policy/LTE_SmolVLA/' || true)"
[[ -z "${XPL_DIRTY}" ]] || {
  echo "Refusing to replace a dirty XPolicyLab workspace; choose a new RMBENCH_REMOTE_DIR." >&2; exit 1;
}
git -C "${XPOLICYLAB_ROOT}" remote set-url origin "${XPOLICYLAB_URL}"
git -C "${XPOLICYLAB_ROOT}" fetch origin "${XPOLICYLAB_REF}"
git -C "${XPOLICYLAB_ROOT}" checkout --detach "origin/${XPOLICYLAB_REF}"

mkdir -p "${RMBENCH_ROOT}/config/lte" "${RMBENCH_ROOT}/scripts/lte"
CONFIG_REMOTE="${RMBENCH_ROOT}/config/lte/$(basename "${CONFIG_SOURCE}")"
install -m 644 "${CONFIG_SOURCE}" "${CONFIG_REMOTE}"
if [[ -r "${CREDENTIAL_FILE}" ]]; then install -m 600 "${CREDENTIAL_FILE}" "${RMBENCH_ROOT}/scripts/lte/.env"; fi
[[ -r "${RMBENCH_ROOT}/scripts/lte/.env" ]] || { echo "Credential file is missing." >&2; exit 1; }

if [[ "${MODE}" == lte || "${MODE}" == both ]]; then
  [[ -f "${BOOTSTRAP_DIR}/lte_smolvla_overlay.tar.gz" ]] || { echo "LTE overlay was not uploaded." >&2; exit 1; }
  [[ -f "${BOOTSTRAP_DIR}/lte_smolvla_cli_override.patch" ]] || { echo "LTE CLI override patch was not uploaded." >&2; exit 1; }
  tar -xzf "${BOOTSTRAP_DIR}/lte_smolvla_overlay.tar.gz" -C "${XPOLICYLAB_ROOT}"
  patch --batch --forward -p0 -d "${XPOLICYLAB_ROOT}" < "${BOOTSTRAP_DIR}/lte_smolvla_cli_override.patch" || {
    echo "Unable to apply the LTE CLI override patch; check the local LTE_SmolVLA source version." >&2
    exit 1
  }
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

if [[ "${MODE}" == lte || "${MODE}" == both ]]; then
  bash "${XPOLICYLAB_ROOT}/policy/LTE_SmolVLA/install.sh"
else
  bash "${XPOLICYLAB_ROOT}/policy/SmolVLA/install.sh"
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"
RMBENCH_BENCH_NAME=RMBench
RMBENCH_DATASET_GROUP=demo_clean
RMBENCH_TASK="${RMBENCH_TASK:?RMBENCH_TASK is required}"
RMBENCH_ENV_CFG="${RMBENCH_ENV_CFG:?RMBENCH_ENV_CFG is required}"
RMBENCH_LEROBOT_REPO_ID="${RMBENCH_DATASET_REPO_ID:-local/rmbench-${RMBENCH_TASK}-${RMBENCH_ENV_CFG}-v30}"
HF_LEROBOT_HOME="${HOME}/.cache/huggingface/lerobot"
export PYTHONPATH="${RMBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME
DATASET_ROOT="${HF_LEROBOT_HOME}/${RMBENCH_LEROBOT_REPO_ID}"
DATASET_INFO="${DATASET_ROOT}/meta/info.json"

if [[ -n "${RMBENCH_DATASET_REPO_IDS:-}" ]]; then
  # Temporal benchmark profile: download the two already-converted LeRobot
  # datasets independently. They remain separate on disk and LTE's dedicated
  # multi-dataset adapter concatenates them with safe global frame indices.
  set -a; source "${RMBENCH_ROOT}/scripts/lte/.env"; set +a
  IFS=',' read -r -a MULTI_REPOS <<<"${RMBENCH_DATASET_REPO_IDS}"
  for repo in "${MULTI_REPOS[@]}"; do
    MULTI_ROOT="${HF_LEROBOT_HOME}/${repo}"
    if [[ ! -f "${MULTI_ROOT}/meta/info.json" ]]; then
      python - "${repo}" "${MULTI_ROOT}" <<'PY'
from huggingface_hub import snapshot_download
import sys

snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
PY
    fi
    [[ -f "${MULTI_ROOT}/meta/info.json" ]] || {
      echo "LeRobot download did not create ${MULTI_ROOT}/meta/info.json" >&2
      exit 1
    }
  done
elif [[ -n "${RMBENCH_DATASET_REPO_ID:-}" ]]; then
  # This is the exact full-suite LeRobot-v3 dataset used by SimpleMemVLA. It
  # already contains all RMBench tasks, so do not download or concatenate the
  # per-task HDF5 demonstrations.
  if [[ ! -f "${DATASET_INFO}" ]]; then
    set -a; source "${RMBENCH_ROOT}/scripts/lte/.env"; set +a
    python - "${RMBENCH_LEROBOT_REPO_ID}" "${DATASET_ROOT}" <<'PY'
from huggingface_hub import snapshot_download
import sys

snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
PY
  fi
  [[ -f "${DATASET_INFO}" ]] || {
    echo "Full RMBench download did not create ${DATASET_INFO}" >&2
    exit 1
  }
else
  DATA_ROOT="${RMBENCH_ROOT}/data/${RMBENCH_DATASET_GROUP}/${RMBENCH_TASK}/${RMBENCH_ENV_CFG}/data"
  if [[ ! -d "${DATA_ROOT}" ]]; then (cd "${RMBENCH_ROOT}" && bash scripts/_download_data.sh "${RMBENCH_TASK}"); fi
  EXPECTED_EPISODES="$(find "${DATA_ROOT}" -maxdepth 1 -type f \( -name '*.hdf5' -o -name '*.h5' \) | wc -l)"
  (( EXPECTED_EPISODES > 0 )) || { echo "No source HDF5 files found in ${DATA_ROOT}" >&2; exit 1; }
  (( EXPECTED_EPISODES <= 200 )) || EXPECTED_EPISODES=200
  COMPLETED_EPISODES=0
  if [[ -f "${DATASET_INFO}" ]]; then
    COMPLETED_EPISODES="$(python - "${DATASET_INFO}" <<'PY'
import json
import sys

try:
    print(int(json.load(open(sys.argv[1])).get("total_episodes", 0)))
except Exception:
    print(0)
PY
)"
  fi
  if [[ "${COMPLETED_EPISODES}" != "${EXPECTED_EPISODES}" ]]; then
    if [[ -e "${DATASET_ROOT}" ]]; then
      PARTIAL_DATASET_ROOT="${DATASET_ROOT}.partial-$(date +%Y%m%d-%H%M%S)"
      echo "Preserving incomplete conversion at ${PARTIAL_DATASET_ROOT}"
      mv "${DATASET_ROOT}" "${PARTIAL_DATASET_ROOT}"
    fi
    cd "${XPOLICYLAB_ROOT}"
    python scripts/transform_lerobot_v30_format.py \
      "${RMBENCH_DATASET_GROUP}.${RMBENCH_TASK}.${RMBENCH_ENV_CFG}" \
      --data_type RoboDojo \
      --repo_id "${RMBENCH_LEROBOT_REPO_ID}" --max_episode 200
  fi
  [[ -f "${DATASET_INFO}" ]] || { echo "LeRobot conversion did not create ${DATASET_INFO}" >&2; exit 1; }
  COMPLETED_EPISODES="$(python - "${DATASET_INFO}" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1])).get("total_episodes", 0)))
PY
)"
  [[ "${COMPLETED_EPISODES}" == "${EXPECTED_EPISODES}" ]] || {
    echo "LeRobot conversion produced ${COMPLETED_EPISODES}/${EXPECTED_EPISODES} episodes." >&2
    exit 1
  }
fi

install -m 700 "${BOOTSTRAP_DIR}/run_training.sh" "${RMBENCH_ROOT}/.brev-run-smolvla.sh"
install -m 700 "${BOOTSTRAP_DIR}/run_eval.sh" "${RMBENCH_ROOT}/scripts/lte/run_eval.sh"
install -m 700 "${BOOTSTRAP_DIR}/log_eval_wandb.py" "${RMBENCH_ROOT}/scripts/lte/log_eval_wandb.py"
install -m 700 "${BOOTSTRAP_DIR}/run_checkpoint_evals.sh" "${RMBENCH_ROOT}/scripts/lte/run_checkpoint_evals.sh"
mkdir -p "${RMBENCH_ROOT}/logs"
if [[ "${RMBENCH_WAIT_FOR_EVALS:-false}" == true ]]; then
  echo "Waiting for active evaluation screens before training on this GPU."
  while screen -ls 2>/dev/null | grep -Eq '[0-9]+\.(smolvla|lte)-eval-'; do
    sleep 60
  done
fi
SESSION="smolvla-${MODE}-${RMBENCH_TASK}-$(date +%Y%m%d-%H%M%S)"
LOGFILE="${RMBENCH_ROOT}/logs/${SESSION}.log"
if command -v screen >/dev/null 2>&1; then
  screen -L -Logfile "${LOGFILE}" -dmS "${SESSION}" \
    bash "${RMBENCH_ROOT}/.brev-run-smolvla.sh" "${RMBENCH_ROOT}" "${MODE}" "${CONFIG_REMOTE}"
  echo "Training launched in screen: ${SESSION}"
else
  nohup bash "${RMBENCH_ROOT}/.brev-run-smolvla.sh" "${RMBENCH_ROOT}" "${MODE}" "${CONFIG_REMOTE}" >"${LOGFILE}" 2>&1 < /dev/null &
  echo "Training launched as PID $! (screen unavailable)"
fi
printf 'Log: %s\nRemote root: %s\n' "${LOGFILE}" "${RMBENCH_ROOT}"
