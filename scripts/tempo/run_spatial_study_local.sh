#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_spatial_study_local.sh <source-root> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
set -a
source /home/sydney1/Repos/RMBench/scripts/lte/.env
set +a
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Keep standard /tmp placement as requested. Do not clean it while workers run:
# Python multiprocessing stores its live resource-sharing sockets there.
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
# Native-filesystem serialization, then sequential copy to the existing WD
# checkpoint path. Same saves/contents/frequency; no recovery or extra saves.
export TEMPO_CHECKPOINT_STAGING_DIR=/tmp
unset LD_LIBRARY_PATH
exec /home/sydney1/Repos/RMBench/.venv-tempo-smolvla/bin/python "$ROOT/scripts/tempo/run_spatial_study_local.py" \
    --root "$ROOT" --config "$CONFIG"
