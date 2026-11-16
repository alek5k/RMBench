#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_queued_temporal_eval_remote.sh <source-root> <queue.yaml> <state.json>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=/data/RMBench-state
export PYTHONPATH="$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$STATE/deps/smovla/src"
export PYTHONUNBUFFERED=1
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$SCRIPT_DIR/queue_temporal_eval_after_training.py" \
    --root "$ROOT" --config "$CONFIG" --state "$3"
