#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: run_queued_local_temporal_evals.sh <queue.yaml>' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
set -a
source /home/sydney1/Repos/RMBench/scripts/lte/.env
set +a
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp TMP=/tmp TEMP=/tmp
exec /home/sydney1/Repos/RMBench/.venv-tempo-smolvla/bin/python \
    "$ROOT/scripts/tempo/queue_local_temporal_evals.py" --config "$1"
