#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_temporal_eval_local.sh <pinned-source-root> <eval.yaml>' >&2; exit 2; }
LAUNCHER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
set -a
source /home/sydney1/Repos/RMBench/scripts/lte/.env
set +a
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp TMP=/tmp TEMP=/tmp
exec /home/sydney1/Repos/RMBench/.venv-tempo-smolvla/bin/python \
    "$LAUNCHER_ROOT/scripts/tempo/run_temporal_eval_local.py" --root "$1" --config "$2"
