#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_liftqa_native64_local.sh <source-root> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
set -a
source /home/sydney1/Repos/RMBench/scripts/lte/.env
set +a
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp TMP=/tmp TEMP=/tmp
unset LD_LIBRARY_PATH
exec /home/sydney1/Repos/RMBench/.venv-tempo-smolvla/bin/python \
    "$ROOT/scripts/tempo/run_liftqa_native64_local.py" --root "$ROOT" --config "$CONFIG"
