#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
set -a
source "$ROOT/scripts/lte/.env"
set +a
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp TMP=/tmp TEMP=/tmp
unset LD_LIBRARY_PATH
cd "$ROOT"
exec "$ROOT/.venv-tempo-smolvla/bin/python" "$ROOT/scripts/tempo/repair_cover_checkpoint_evals_local.py" \
  --config "$ROOT/config/tempo/repair_cover_blocks_queries128_evals_local.yaml" "$@"
