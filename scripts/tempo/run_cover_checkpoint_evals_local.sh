#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
set -a
source "$ROOT/scripts/lte/.env"
set +a
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
unset LD_LIBRARY_PATH
cd "$ROOT"
exec "$ROOT/.venv-tempo-smolvla/bin/python" "$ROOT/scripts/tempo/run_cover_checkpoint_evals_local.py" \
  --config "$ROOT/config/tempo/eval_cover_blocks_queries128_100k_200k_local.yaml" "$@"
