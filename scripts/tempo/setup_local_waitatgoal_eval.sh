#!/usr/bin/env bash
# Separate, pinned policy environment; do not modify existing local train/sim envs.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export TEMPO_SMOLVLA_ENV="${ROOT}/.venv-tempo-smolvla"
export SMOVLA_ROOT="${ROOT}/.deps/tempo-smovla"
# Blackwell RTX 5090 requires the CUDA 12.8 wheel; policy/library versions match Brev.
export TEMPO_TORCH_INDEX=https://download.pytorch.org/whl/cu128
export PATH="${HOME}/.local/bin:${PATH}" UV_LINK_MODE=copy
mkdir -p "${ROOT}/.deps"
bash "${ROOT}/scripts/tempo/install_smolvla_environment.sh"
uv pip install --python "${TEMPO_SMOLVLA_ENV}/bin/python" \
  pygame==2.6.1 'pymunk>=6.6,<7' shapely scikit-image scipy \
  --override "${ROOT}/scripts/tempo/overrides.txt"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
"${TEMPO_SMOLVLA_ENV}/bin/python" - "$ROOT" <<'PY'
import importlib.util,sys
from pathlib import Path
source=Path(sys.argv[1]).parent/'LTE-DP/ltedp/envs/waitatgoal/waitatgoal_env.py'
spec=importlib.util.spec_from_file_location('tempo_waitatgoal_setup_check',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
env=module.WaitAtGoal(render_size=144,render_mode='rgb_array')
obs,_=env.reset(seed=0)
env.step([0.,0.])
print('Local WaitAtGoal reset/step passed; image:',obs['full_image'].shape)
env.close()
PY
