#!/usr/bin/env bash
# Verify and install an immutable RMBench source release, then launch training.
set -Eeuo pipefail

[[ $# -eq 3 ]] || { echo "Usage: $0 <bootstrap-directory> <smolvla|lte> <bundle-id>" >&2; exit 2; }
BOOTSTRAP_DIR=$1
MODE=$2
BUNDLE_ID=$3
case "${MODE}" in smolvla|lte) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
[[ "${BUNDLE_ID}" =~ ^[0-9a-f]{20}$ ]] || { echo "Invalid source bundle ID: ${BUNDLE_ID}" >&2; exit 2; }
BOOTSTRAP_DIR="$(cd "${BOOTSTRAP_DIR}" && pwd)"

mkdir -p "${HOME}/.lte-bootstrap"
exec 9>"${HOME}/.lte-bootstrap/setup.lock"
flock -n 9 || { echo "Another LTE bootstrap is already running on this instance." >&2; exit 75; }

ARCHIVE="${BOOTSTRAP_DIR}/rmbench-source-${BUNDLE_ID}.tar.zst"
CHECKSUM="${ARCHIVE}.sha256"
RELEASES_ROOT="${RMBENCH_RELEASES_ROOT:-${HOME}/RMBench-releases}"
STATE_ROOT="${RMBENCH_STATE_ROOT:-${HOME}/RMBench-state}"
RELEASE_DIR="${RELEASES_ROOT}/${BUNDLE_ID}"
RMBENCH_ROOT="${RELEASE_DIR}/RMBench"
CURRENT_LINK="${RELEASES_ROOT}/current"
CREDENTIAL_FILE="${BOOTSTRAP_DIR}/credentials.env"
STATE_CREDENTIAL_FILE="${STATE_ROOT}/credentials.env"
CONDA_BASE="${HOME}/miniforge3"

[[ -r "${ARCHIVE}" ]] || { echo "Source bundle not readable: ${ARCHIVE}" >&2; exit 1; }
[[ -r "${CHECKSUM}" ]] || { echo "Source bundle checksum not readable: ${CHECKSUM}" >&2; exit 1; }
if ! command -v zstd >/dev/null 2>&1; then
  command -v sudo >/dev/null 2>&1 || { echo "zstd is required to extract the source bundle." >&2; exit 1; }
  sudo -n apt-get update && sudo -n apt-get install -y zstd || { echo "Unable to install zstd non-interactively." >&2; exit 1; }
fi
(
  cd "${BOOTSTRAP_DIR}"
  sha256sum -c "$(basename "${CHECKSUM}")"
)

python3 - "${ARCHIVE}" <<'PY'
import os
import posixpath
import subprocess
import sys
import tarfile

archive = sys.argv[1]
process = subprocess.Popen(["zstd", "-q", "-d", "-c", archive], stdout=subprocess.PIPE)
assert process.stdout is not None
with tarfile.open(fileobj=process.stdout, mode="r|") as source:
    for member in source:
        normalized = posixpath.normpath(member.name)
        if member.name.startswith("/") or normalized == ".." or normalized.startswith("../"):
            raise SystemExit(f"Unsafe archive member: {member.name}")
        if not (normalized == "RMBench" or normalized.startswith("RMBench/")):
            raise SystemExit(f"Archive member is outside RMBench/: {member.name}")
        if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
            raise SystemExit(f"Unsupported archive member: {member.name}")
        if member.issym() or member.islnk():
            target = member.linkname
            if os.path.isabs(target):
                raise SystemExit(f"Unsafe archive link: {member.name} -> {target}")
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(normalized), target))
            if not (resolved == "RMBench" or resolved.startswith("RMBench/")):
                raise SystemExit(f"Escaping archive link: {member.name} -> {target}")
if process.wait() != 0:
    raise SystemExit("Unable to inspect source bundle")
PY

mkdir -p "${RELEASES_ROOT}" \
  "${STATE_ROOT}"/{checkpoints,envs,eval_result,logs,runs,wandb} \
  "${STATE_ROOT}/assets"/{embodiments,objects} \
  "${STATE_ROOT}/data"/{demo_clean,download_cache}
if [[ ! -d "${RMBENCH_ROOT}" ]]; then
  [[ ! -e "${RELEASE_DIR}" ]] || { echo "Incomplete release path already exists: ${RELEASE_DIR}" >&2; exit 1; }
  TEMP_RELEASE="${RELEASES_ROOT}/.${BUNDLE_ID}.extracting.$$"
  trap 'rm -rf "${TEMP_RELEASE:-}"' EXIT
  mkdir "${TEMP_RELEASE}"
  zstd -q -d -c "${ARCHIVE}" | tar -xf - -C "${TEMP_RELEASE}" --no-same-owner --no-same-permissions
  [[ -f "${TEMP_RELEASE}/RMBench/SOURCE_BUNDLE.json" ]] || { echo "Bundle manifest is missing." >&2; exit 1; }
  [[ -d "${TEMP_RELEASE}/RMBench/XPolicyLab/policy" ]] || { echo "Bundled XPolicyLab source is missing." >&2; exit 1; }
  read -r EXTRACTED_ID MANIFEST_FORMAT MANIFEST_RECIPE < <(python3 - "${TEMP_RELEASE}/RMBench/SOURCE_BUNDLE.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["bundle_id"], manifest["format"], manifest["recipe"])
PY
)
  [[ "${EXTRACTED_ID}" == "${BUNDLE_ID}" ]] || { echo "Bundle manifest ID mismatch." >&2; exit 1; }
  [[ "${MANIFEST_FORMAT}" == rmbench-source-bundle-v1 ]] || { echo "Unsupported bundle manifest format: ${MANIFEST_FORMAT}" >&2; exit 1; }
  [[ "${MANIFEST_RECIPE}" =~ ^config/lte/[A-Za-z0-9._-]+\.ya?ml$ ]] || { echo "Unsafe recipe path in bundle manifest." >&2; exit 1; }
  mv "${TEMP_RELEASE}" "${RELEASE_DIR}"
  TEMP_RELEASE=
fi

if [[ -r "${CREDENTIAL_FILE}" ]]; then
  install -m 600 "${CREDENTIAL_FILE}" "${STATE_CREDENTIAL_FILE}"
elif [[ -e "${STATE_CREDENTIAL_FILE}" ]]; then
  chmod 600 "${STATE_CREDENTIAL_FILE}"
fi
# Dataset conversion below can fetch the RMBench archives from Hugging Face.
# Load credentials here as well as in run_training.sh, without echoing them.
if [[ -r "${STATE_CREDENTIAL_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${STATE_CREDENTIAL_FILE}"
  set +a
fi

# Large generated trees are external, while tracked source directories remain
# in the release and expose only their mutable children through symlinks.
link_state_path() {
  local source_path=$1 state_path=$2
  mkdir -p "$(dirname "${source_path}")" "${state_path}"
  [[ ! -e "${source_path}" || -L "${source_path}" ]] || { echo "Refusing to replace bundled mutable path: ${source_path}" >&2; exit 1; }
  rm -f "${source_path}"
  ln -s "${state_path}" "${source_path}"
}
link_state_path "${RMBENCH_ROOT}/assets/embodiments" "${STATE_ROOT}/assets/embodiments"
link_state_path "${RMBENCH_ROOT}/assets/objects" "${STATE_ROOT}/assets/objects"
link_state_path "${RMBENCH_ROOT}/data/demo_clean" "${STATE_ROOT}/data/demo_clean"
link_state_path "${RMBENCH_ROOT}/data/download_cache" "${STATE_ROOT}/data/download_cache"
link_state_path "${RMBENCH_ROOT}/envs/curobo" "${STATE_ROOT}/envs/curobo"
for name in checkpoints eval_result logs; do
  link_state_path "${RMBENCH_ROOT}/${name}" "${STATE_ROOT}/${name}"
done
ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"
mv -Tf "${CURRENT_LINK}.new" "${CURRENT_LINK}"

if [[ "${RMBENCH_SETUP_VALIDATE_ONLY:-false}" == true ]]; then
  printf 'Validated bundle: %s\nRelease: %s\nState: %s\n' "${BUNDLE_ID}" "${RMBENCH_ROOT}" "${STATE_ROOT}"
  exit 0
fi

CONFIG_READER="${RMBENCH_ROOT}/scripts/lte/training_config.py"
CONFIG_RELATIVE="$(python3 - "${RMBENCH_ROOT}/SOURCE_BUNDLE.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["recipe"])
PY
)"
CONFIG_SOURCE="${RMBENCH_ROOT}/${CONFIG_RELATIVE}"
[[ -r "${CONFIG_SOURCE}" ]] || { echo "Bundled training configuration not readable: ${CONFIG_SOURCE}" >&2; exit 1; }
python3 "${CONFIG_READER}" "${CONFIG_SOURCE}" validate
config_get() { python3 "${CONFIG_READER}" "${CONFIG_SOURCE}" get "$1"; }
CONDA_ENV="$(config_get conda_env)"
CONDA_ENV="${CONDA_ENV:-smolvla}"

if [[ ! -x "${CONDA_BASE}/bin/conda" ]]; then
  installer="${BOOTSTRAP_DIR}/Miniforge3-Linux-x86_64.sh"
  [[ -f "${installer}" ]] || curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "${installer}"
  bash "${installer}" -b -p "${CONDA_BASE}"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "${CONDA_ENV}"; then conda create -y -n "${CONDA_ENV}" python=3.10 pip; fi
export CONDA_BASE CONDA_ROOT="${CONDA_BASE}" SMOVLA_CONDA_ENV="${CONDA_ENV}"
SMOVLA_SOURCE_ROOT="${STATE_ROOT}/envs/lerobot-v0.4.4"
export SMOVLA_ROOT="${SMOVLA_SOURCE_ROOT}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  command -v sudo >/dev/null 2>&1 || { echo "ffmpeg is required for LeRobot video conversion." >&2; exit 1; }
  sudo -n apt-get update && sudo -n apt-get install -y ffmpeg || { echo "Unable to install ffmpeg non-interactively." >&2; exit 1; }
fi

if [[ "${MODE}" == lte ]]; then
  bash "${RMBENCH_ROOT}/XPolicyLab/policy/LTE_SmolVLA/install.sh"
else
  bash "${RMBENCH_ROOT}/XPolicyLab/policy/SmolVLA/install.sh"
fi
# SmolVLA's installer may retain LeRobot in the shared state directory rather
# than materialising it inside an immutable source release (notably when the
# LTE adapter reuses the already-installed baseline). Resolve the actual
# editable source from the active interpreter instead of assuming a bundled
# policy/SmolVLA/smovla checkout.
INSTALLED_SMOVLA_SOURCE="$("${CONDA_BASE}/envs/${CONDA_ENV}/bin/python" - <<'PY'
from pathlib import Path
import lerobot

print(Path(lerobot.__file__).resolve().parents[2])
PY
)"
[[ -f "${INSTALLED_SMOVLA_SOURCE}/src/lerobot/rl/wandb_utils.py" ]] || {
  echo "SmolVLA installation did not expose its LeRobot source: ${INSTALLED_SMOVLA_SOURCE}" >&2
  exit 1
}
python3 "${RMBENCH_ROOT}/scripts/lte/patch_lerobot_wandb_artifact_names.py" "${INSTALLED_SMOVLA_SOURCE}"
# install.sh installs LeRobot editable from SMOVLA_ROOT.  Keep the copy bundled
# with this immutable release intact: replacing it with a state-directory
# symlink would mutate the release and is unnecessary for imports or training.

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
TASK="$(config_get task)"
ENV_CFG="$(config_get env_cfg)"
DATASET_REPO_ID="$(python3 "${CONFIG_READER}" "${CONFIG_SOURCE}" args | python3 -c 'import sys; print(next(line.split("=", 1)[1].strip() for line in sys.stdin if line.startswith("--dataset.repo_id=")))')"
DOWNLOAD_REPOS="$(config_get dataset.repo_ids)"
DOWNLOAD_DATASET="$(config_get dataset.download)"
LOCAL_TASKS="$(config_get dataset.tasks)"
TEMPORAL_DATASETS="$(config_get dataset.temporal)"
MERGE_REPO_ID="$(config_get dataset.merge_repo_id)"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
export PYTHONPATH="${RMBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME
DATASET_TRAIN_ROOT="$(python3 "${CONFIG_READER}" "${CONFIG_SOURCE}" args | python3 -c 'import sys; print(next(line.split("=", 1)[1].strip() for line in sys.stdin if line.startswith("--dataset.root=")))')"

if [[ "${TEMPORAL_DATASETS}" == true ]]; then
  python -m pip install 'zarr<3'
  # Multi-task temporal recipes point directly at the common dataset parent,
  # whereas a single-task LeRobotDataset must point at its task directory.
  # The converter always expects the parent and creates <parent>/<task-name>.
  TEMPORAL_OUTPUT_ROOT="${DATASET_TRAIN_ROOT}"
  case "${TASK}" in
    waitatgoal)
      [[ "$(basename "${DATASET_TRAIN_ROOT}")" == waitatgoal_v4 ]] && TEMPORAL_OUTPUT_ROOT="$(dirname "${DATASET_TRAIN_ROOT}")"
      ;;
    liftqa)
      [[ "$(basename "${DATASET_TRAIN_ROOT}")" == liftqa_v2 ]] && TEMPORAL_OUTPUT_ROOT="$(dirname "${DATASET_TRAIN_ROOT}")"
      ;;
  esac
  python "${RMBENCH_ROOT}/scripts/lte/prepare_temporal_datasets.py" \
    --output-root "${TEMPORAL_OUTPUT_ROOT}"
fi

if [[ -n "${LOCAL_TASKS}" ]]; then
  [[ -n "${DOWNLOAD_REPOS}" ]] || { echo "launcher.dataset.tasks requires launcher.dataset.repo_ids." >&2; exit 2; }
  IFS=',' read -r -a LOCAL_TASK_LIST <<<"${LOCAL_TASKS}"
  IFS=',' read -r -a LOCAL_REPO_LIST <<<"${DOWNLOAD_REPOS}"
  (( ${#LOCAL_TASK_LIST[@]} == ${#LOCAL_REPO_LIST[@]} )) || {
    echo "launcher.dataset.tasks and launcher.dataset.repo_ids must have matching lengths." >&2
    exit 2
  }
  for index in "${!LOCAL_TASK_LIST[@]}"; do
    task_name="${LOCAL_TASK_LIST[$index]}"
    repo_id="${LOCAL_REPO_LIST[$index]}"
    dataset_root="${HF_LEROBOT_HOME}/${repo_id}"
    if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
      (cd "${RMBENCH_ROOT}" && bash scripts/_download_data.sh "${task_name}")
      (cd "${RMBENCH_ROOT}/XPolicyLab" && python scripts/transform_lerobot_v30_format.py \
        "demo_clean.${task_name}.*" --data_type RoboDojo --repo_id "${repo_id}" --max_episode 200)
    fi
    [[ -f "${dataset_root}/meta/info.json" ]] || { echo "LeRobot conversion failed for ${task_name}." >&2; exit 1; }
  done
  # LeRobot 0.4.4's generic multi-dataset loader is deliberately disabled.
  # For non-LTE joint controls, materialise a normal merged dataset instead,
  # retaining task instructions and per-episode boundaries for SmolVLA.
  if [[ -n "${MERGE_REPO_ID}" ]]; then
    MERGE_ROOT="${HF_LEROBOT_HOME}/${MERGE_REPO_ID}"
    if [[ ! -f "${MERGE_ROOT}/meta/info.json" ]]; then
      [[ ! -e "${MERGE_ROOT}" ]] || mv "${MERGE_ROOT}" "${MERGE_ROOT}.partial-$(date +%Y%m%d-%H%M%S)"
      MERGE_ARGS=(
        --output-repo-id "${MERGE_REPO_ID}"
        --output-root "${MERGE_ROOT}"
      )
      for repo_id in "${LOCAL_REPO_LIST[@]}"; do
        MERGE_ARGS+=(--source "${repo_id}" "${HF_LEROBOT_HOME}/${repo_id}")
      done
      python "${RMBENCH_ROOT}/scripts/lte/merge_lerobot_datasets.py" "${MERGE_ARGS[@]}"
    fi
    [[ -f "${MERGE_ROOT}/meta/info.json" ]] || { echo "Merged LeRobot dataset creation failed." >&2; exit 1; }
  fi
elif [[ -n "${DOWNLOAD_REPOS}" && "${DOWNLOAD_DATASET}" == true ]]; then
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
# Temporal datasets above are already provisioned as local LeRobot datasets.
# Do not fall through to RMBench's unrelated RoboDojo HDF5 conversion path.
elif [[ "${DATASET_REPO_ID}" == local/* && "${TEMPORAL_DATASETS}" != true ]]; then
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
    cd "${RMBENCH_ROOT}/XPolicyLab"
    python scripts/transform_lerobot_v30_format.py "demo_clean.${TASK}.${ENV_CFG}" \
      --data_type RoboDojo --repo_id "${DATASET_REPO_ID}" --max_episode 200
  fi
  [[ -f "${DATASET_INFO}" ]] || { echo "LeRobot conversion failed." >&2; exit 1; }
fi

# Recipes that use an explicitly staged local dataset must never silently fall
# back to a Hub download.  Fail before launching if a component is missing.
if [[ -n "${MERGE_REPO_ID}" ]]; then
  [[ -f "${DATASET_TRAIN_ROOT}/meta/info.json" ]] || {
    echo "Merged dataset is missing: ${DATASET_TRAIN_ROOT}" >&2
    exit 1
  }
elif [[ -n "${DOWNLOAD_REPOS}" && "${DOWNLOAD_DATASET}" != true ]]; then
  IFS=',' read -r -a STAGED_REPOS <<<"${DOWNLOAD_REPOS}"
  for repo_id in "${STAGED_REPOS[@]}"; do
    [[ -f "${DATASET_TRAIN_ROOT}/${repo_id}/meta/info.json" ]] || {
      echo "Staged dataset is missing: ${DATASET_TRAIN_ROOT}/${repo_id}" >&2
      exit 1
    }
  done
fi

if [[ "$(config_get wait_for_evals)" == true ]]; then
  while screen -ls 2>/dev/null | grep -Eq '[0-9]+\.(smolvla|lte)-eval-'; do sleep 60; done
fi
POST_TRAIN_TEMPORAL_EVALS="$(config_get post_train_temporal_evals)"
RUNNER="${RMBENCH_ROOT}/scripts/lte/run_training.sh"
if [[ "${POST_TRAIN_TEMPORAL_EVALS}" == true ]]; then
  RUNNER="${RMBENCH_ROOT}/scripts/lte/run_training_then_temporal_evals.sh"
fi
SESSION="smolvla-${MODE}-${TASK}-$(date +%Y%m%d-%H%M%S)"
LOGFILE="${STATE_ROOT}/logs/${SESSION}.log"
RUN_CONFIG_DIR="${STATE_ROOT}/runs/${SESSION}/config/lte"
mkdir -p "${RUN_CONFIG_DIR}"
CONFIG_REMOTE="${RUN_CONFIG_DIR}/$(basename "${CONFIG_SOURCE}")"
install -m 644 "${CONFIG_SOURCE}" "${CONFIG_REMOTE}"
if command -v screen >/dev/null 2>&1; then
  screen -L -Logfile "${LOGFILE}" -dmS "${SESSION}" env \
    RMBENCH_STATE_ROOT="${STATE_ROOT}" RMBENCH_CREDENTIAL_FILE="${STATE_CREDENTIAL_FILE}" \
    RMBENCH_OUTPUT_ROOT="${STATE_ROOT}/checkpoints" PYTHONDONTWRITEBYTECODE=1 WANDB_DIR="${STATE_ROOT}/wandb" \
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" SMOVLA_ROOT="${SMOVLA_ROOT}" \
    bash "${RUNNER}" "${RMBENCH_ROOT}" "${MODE}" "${CONFIG_REMOTE}"
  echo "Training launched in screen: ${SESSION}"
else
  nohup env RMBENCH_STATE_ROOT="${STATE_ROOT}" RMBENCH_CREDENTIAL_FILE="${STATE_CREDENTIAL_FILE}" \
    RMBENCH_OUTPUT_ROOT="${STATE_ROOT}/checkpoints" PYTHONDONTWRITEBYTECODE=1 WANDB_DIR="${STATE_ROOT}/wandb" \
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" SMOVLA_ROOT="${SMOVLA_ROOT}" \
    bash "${RUNNER}" "${RMBENCH_ROOT}" "${MODE}" "${CONFIG_REMOTE}" >"${LOGFILE}" 2>&1 < /dev/null &
  echo "Training launched as PID $! (screen unavailable)"
fi
printf 'Bundle: %s\nRelease: %s\nState: %s\nLog: %s\n' "${BUNDLE_ID}" "${RMBENCH_ROOT}" "${STATE_ROOT}" "${LOGFILE}"
