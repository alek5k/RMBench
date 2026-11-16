#!/usr/bin/env bash
# Never install simulator packages into the active policy-training environment.
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: setup_temporal_eval_remote.sh <release-root>' >&2; exit 2; }
ROOT=$(realpath "$1")
STATE=/data/RMBench-state
ENV_ROOT="${STATE}/envs/tempo-temporal-eval"
POLICY_PYTHON="${HOME}/RMBench-state/envs/tempo-smolvla/bin/python"
UV="${HOME}/.local/bin/uv"
export UV_LINK_MODE=copy UV_CACHE_DIR="${STATE}/uv-cache"
mkdir -p "${STATE}/envs"
exec 9>"${STATE}/envs/tempo-temporal-eval.setup.lock"
flock 9
[[ -x "${ENV_ROOT}/bin/python" ]] || "$UV" venv --python "$POLICY_PYTHON" "$ENV_ROOT"
"$UV" pip install --python "${ENV_ROOT}/bin/python" \
  -r "$ROOT/scripts/tempo/requirements-temporal-eval.txt" \
  --override "$ROOT/scripts/tempo/overrides.txt"
"${ENV_ROOT}/bin/python" - <<'PY'
import importlib.metadata as metadata
import hashlib,json
from pathlib import Path
import mujoco, pygame, pymunk
assert mujoco.__version__=='3.5.0'
root=Path('/data/RMBench-state/deps/temporal-eval')
manifest=json.loads((root/'ENVIRONMENT_SOURCE.json').read_text())
for relative,expected in manifest['files'].items():
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Temporal environment source mismatch: {relative}')
print('Temporal simulator packages ready:', {key:metadata.version(key) for key in
      ('mujoco','robosuite','robosuite_models','numpy','pymunk')},flush=True)
PY
