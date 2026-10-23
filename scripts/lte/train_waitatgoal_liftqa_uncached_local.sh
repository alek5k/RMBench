#!/usr/bin/env bash
# Launch the 20k, two-GPU joint LTE SmolVLA direct-decode recipe locally.
set -Eeuo pipefail

[[ $# -eq 0 ]] || { echo "Usage: $0" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
RMBENCH_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
exec bash "$SCRIPT_DIR/run_training.sh" "$RMBENCH_ROOT" lte \
  "$RMBENCH_ROOT/config/lte/waitatgoal_liftqa_lte_smolvla_2gpu_20k_uncached.env"
