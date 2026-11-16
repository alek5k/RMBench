#!/usr/bin/env bash
# Shared sequential runner for matched native SmolVLA / TEMPO horizon studies.
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_horizon_seed_suite_remote.sh <release> <suite.yaml> <state.json>' >&2; exit 2; }
ROOT=$(realpath "$1")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$SCRIPT_DIR:$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:/data/RMBench-state/deps/smovla/src"
export PYTHONUNBUFFERED=1
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$SCRIPT_DIR/queue_smolvla_horizon_baselines.py" \
    --root "$ROOT" --config "$2" --state "$3"
