#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
set -a
source "$ROOT/scripts/lte/.env"
set +a
cd "$ROOT"
exec "$ROOT/.venv-tempo-smolvla/bin/python" "$ROOT/scripts/tempo/run_local_checkpoint_evals.py" \
  --config "$ROOT/config/tempo/eval_battery_try_tempo_horizon10_local_all_checkpoints.yaml" "$@"
