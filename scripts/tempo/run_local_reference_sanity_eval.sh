#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
set -a
source "$ROOT/scripts/lte/.env"
set +a
cd "$ROOT"
exec "$ROOT/.venv-tempo-smolvla/bin/python" "$ROOT/scripts/tempo/run_local_reference_sanity_eval.py" \
  --config "$ROOT/config/tempo/eval_battery_try_tempo_reference_100k_local_sanity.yaml" "$@"
