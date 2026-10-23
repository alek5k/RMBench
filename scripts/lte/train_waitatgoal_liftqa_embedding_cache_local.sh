#!/usr/bin/env bash
# Launch a selected WaitAtGoal + LiftQA LTE recipe locally.
set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 <training-config.yaml>" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
RMBENCH_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
exec bash "$SCRIPT_DIR/run_training.sh" "$RMBENCH_ROOT" lte "$1"
