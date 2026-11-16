#!/usr/bin/env bash
# Provision the prepared source release, smoke-test real weights, then train.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 <RMBench-source-root> <training-config>" >&2; exit 2; }
RMBENCH_ROOT="$(realpath "$1")"
CONFIG="$(realpath "$2")"
STATE_ROOT="${TEMPO_STATE_ROOT:-/data/RMBench-state}"
ENV_ROOT="${TEMPO_ENV_ROOT:-${HOME}/RMBench-state/envs/tempo-pi05}"
PYTHON="${ENV_ROOT}/bin/python"
OPENPI_ROOT="${RMBENCH_ROOT}/XPolicyLab/policy/Pi_05/openpi"
export PATH="${HOME}/.local/bin:${ENV_ROOT}/bin:${PATH}"
export UV_LINK_MODE=copy
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${STATE_ROOT}/huggingface"
export HF_LEROBOT_HOME="${STATE_ROOT}/datasets/lerobot"
export OPENPI_DATA_HOME="${STATE_ROOT}/openpi-cache"
export WANDB_DIR="${STATE_ROOT}/wandb"
export PYTHONPATH="${RMBENCH_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
mkdir -p "${STATE_ROOT}"/{assets,checkpoints,datasets,embedding_cache,logs,wandb,data/download_cache,data/demo_clean}
if [[ -f "${STATE_ROOT}/credentials.env" ]]; then
    set -a
    source "${STATE_ROOT}/credentials.env"
    set +a
fi
[[ -x "${PYTHON}" ]] || { echo "Run install_environment.sh first" >&2; exit 1; }
uv pip install --python "${PYTHON}" --no-deps -e "${OPENPI_ROOT}" -e "${OPENPI_ROOT}/packages/openpi-client"
TRANSFORMERS_DIR="$("${PYTHON}" -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
cp -r "${OPENPI_ROOT}/src/openpi/models_pytorch/transformers_replace/." "${TRANSFORMERS_DIR}/"
"${PYTHON}" -c 'import torch; from XPolicyLab.policy.TEMPO_Pi05.modeling_tempo import TEMPOPi05; print("TEMPO imports and CUDA", torch.cuda.is_available())'
while [[ ! -f "${STATE_ROOT}/models/pi05_base_pytorch/model.safetensors" ]]; do
    screen -ls | grep -q '[.]tempo-model-download[[:space:]]' || {
        echo 'Pi05 base download is incomplete and no downloader is running.' >&2; exit 1;
    }
    echo 'Waiting for the checksum-verified Pi05 base download...'
    sleep 10
done

# Store downloaded source data on the attached disk, using the official HF archive.
for directory in demo_clean download_cache; do
    target="${RMBENCH_ROOT}/data/${directory}"
    if [[ ! -e "${target}" ]]; then
        ln -s "${STATE_ROOT}/data/${directory}" "${target}"
    fi
done
"${PYTHON}" "${RMBENCH_ROOT}/data/_download.py" battery_try \
    --revision 5d8ff21038fba32ea549e9eee8497aaba9aab5cf
DATASET_ROOT="${HF_LEROBOT_HOME}/local/rmbench-battery_try-aloha_agilex-v30"
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    cd "${RMBENCH_ROOT}/XPolicyLab"
    "${PYTHON}" scripts/transform_lerobot_v30_format.py demo_clean.battery_try.aloha_agilex \
        --data_type RoboDojo --repo_id local/rmbench-battery_try-aloha_agilex-v30 \
        --max_episode 200 --resolution 240x320
fi
"${PYTHON}" - "${DATASET_ROOT}" <<'PY'
import json, pathlib, sys
info = json.loads((pathlib.Path(sys.argv[1]) / 'meta/info.json').read_text())
assert info['total_episodes'] == 50, info
assert info['total_frames'] > 30000, info
print('Battery dataset:', info['total_episodes'], 'episodes,', info['total_frames'], 'frames')
PY
cd "${RMBENCH_ROOT}"
if [[ ! -f "${STATE_ROOT}/assets/tempo/battery_try/norm_stats.json" ]]; then
    "${PYTHON}" -m XPolicyLab.policy.TEMPO_Pi05.compute_norm_stats --config "${CONFIG}"
fi
"${PYTHON}" -m XPolicyLab.policy.TEMPO_Pi05.smoke_test --config "${CONFIG}"
exec "${PYTHON}" -m XPolicyLab.policy.TEMPO_Pi05.train --config "${CONFIG}"
