#!/usr/bin/env bash
# PyTorch OpenPI environment for TEMPO on a CUDA 12.6-compatible L40S host.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT="${TEMPO_STATE_ROOT:-${HOME}/RMBench-state}"
ENV_ROOT="${STATE_ROOT}/envs/tempo-pi05"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE=copy
command -v uv >/dev/null || { echo 'Install uv first.' >&2; exit 1; }
uv python install 3.11
[[ -x "${ENV_ROOT}/bin/python" ]] || uv venv --python 3.11 "${ENV_ROOT}"
uv pip install --python "${ENV_ROOT}/bin/python" \
    --index-url https://download.pytorch.org/whl/cu126 'torch==2.10.0' 'torchvision==0.25.0'
uv pip install --python "${ENV_ROOT}/bin/python" -r "${SCRIPT_DIR}/requirements.txt" \
    --override "${SCRIPT_DIR}/overrides.txt"
"${ENV_ROOT}/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
"${ENV_ROOT}/bin/python" -c 'import pandas, pyarrow, cv2; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print("Data dependencies import OK")'
