#!/usr/bin/env bash
# Provision the pinned policy environment and only the selected training dataset.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: setup_smolvla_training.sh <RMBench release> <training-config>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${STATE}/uv-cache" UV_PYTHON_INSTALL_DIR="${STATE}/uv-python" UV_LINK_MODE=copy
export HF_HOME="${STATE}/huggingface" HF_LEROBOT_HOME="${STATE}/datasets/lerobot"
export SMOVLA_ROOT="${STATE}/deps/smovla" PYTHONPATH="${ROOT}:${ROOT}/XPolicyLab"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "${STATE}/deps" "${STATE}/datasets" "${STATE}/embedding_cache" "${STATE}/checkpoints" "${STATE}/logs"
set -a
source "${STATE}/credentials.env"
set +a
bash "${ROOT}/scripts/tempo/install_smolvla_environment.sh"
PYTHON="${TEMPO_SMOLVLA_ENV:-${HOME}/RMBench-state/envs/tempo-smolvla}/bin/python"
DATASET=$("$PYTHON" - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['training']['repo_id'])
PY
)
case "$DATASET" in
  local/waitatgoal_v4|local/liftqa_v2)
    uv pip install --python "$PYTHON" 'zarr<3'
    bash -c 'cd "$1" && "$2" scripts/lte/prepare_temporal_datasets.py --output-root "$3/datasets" --datasets "$4"' \
      setup-temporal "$ROOT" "$PYTHON" "$STATE" "${DATASET#local/}"
    ;;
  local/rmbench-battery_try-aloha_agilex-v30)
    mkdir -p "${STATE}/data/demo_clean" "${STATE}/data/download_cache"
    for name in demo_clean download_cache; do
      [[ -e "${ROOT}/data/${name}" ]] || ln -s "${STATE}/data/${name}" "${ROOT}/data/${name}"
    done
    "$PYTHON" "${ROOT}/data/_download.py" battery_try --revision 5d8ff21038fba32ea549e9eee8497aaba9aab5cf
    if [[ ! -f "${HF_LEROBOT_HOME}/${DATASET}/meta/info.json" ]]; then
      cd "${ROOT}/XPolicyLab"
      "$PYTHON" scripts/transform_lerobot_v30_format.py demo_clean.battery_try.aloha_agilex \
        --data_type RoboDojo --repo_id "$DATASET" --max_episode 200 --resolution 240x320
    fi
    ;;
  local/rmbench-cover_blocks-aloha_agilex-v30)
    # Official Hugging Face source is the default on a fresh host. Existing
    # complete conversions are verified, never overwritten or assumed uploaded.
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/tempo/prepare_rmbench_dataset.py" \
      --root "$ROOT" --config "$ROOT/config/tempo/prepare_cover_blocks_queries.yaml"
    ;;
  *) echo "No dataset preparation recipe for ${DATASET}; refusing to train on an assumed dataset." >&2; exit 1 ;;
esac
"$PYTHON" - "$CONFIG" <<'PY'
import json,sys,yaml
from pathlib import Path
t=yaml.safe_load(open(sys.argv[1]))['training']
info=json.loads((Path(t['dataset_root'])/'meta/info.json').read_text())
assert info['codebase_version']=='v3.0'
if t['repo_id']=='local/waitatgoal_v4':
    assert (info['total_episodes'],info['total_frames'])==(203,14008)
    assert info['features']['action']['shape']==[2]
    assert info['features']['observation.state']['shape']==[2]
    assert info['features']['observation.image']['shape']==[3,144,144]
print('Prepared dataset:',t['repo_id'],info['total_episodes'],'episodes,',info['total_frames'],'frames',flush=True)
PY
