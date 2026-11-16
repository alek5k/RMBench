#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${TEMPO_SMOLVLA_ENV:-${HOME}/RMBench-state/envs/tempo-smolvla}"
SMOVLA_ROOT="${SMOVLA_ROOT:-/data/RMBench-state/deps/smovla}"
export PATH="${HOME}/.local/bin:${PATH}" UV_LINK_MODE=copy
if [[ ! -d "${SMOVLA_ROOT}/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/huggingface/lerobot.git "${SMOVLA_ROOT}"
    git -C "${SMOVLA_ROOT}" checkout --detach 8fff0fde7c79f23a93d845d1a50e985de01f8b8a
fi
[[ "$(git -C "${SMOVLA_ROOT}" rev-parse HEAD)" == 8fff0fde7c79f23a93d845d1a50e985de01f8b8a ]]
[[ -x "${ENV_ROOT}/bin/python" ]] || uv venv --python 3.11 "${ENV_ROOT}"
uv pip install --python "${ENV_ROOT}/bin/python" --index-url "${TEMPO_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}" \
    'torch==2.10.0' 'torchvision==0.25.0'
uv pip install --python "${ENV_ROOT}/bin/python" -e "${SMOVLA_ROOT}[smolvla]" \
    'transformers==4.57.1' 'datasets==4.1.1' 'pyarrow==21.0.0' 'pandas==2.2.3' \
    'huggingface-hub==0.35.3' 'torchcodec==0.10.0' h5py pyyaml websockets msgpack-numpy \
    --override "${SCRIPT_DIR}/overrides.txt"
"${ENV_ROOT}/bin/python" -c 'import torch; from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; assert torch.cuda.is_available(); print("SmolVLA environment ready", torch.__version__)'
