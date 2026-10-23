#!/usr/bin/env bash
# Launch the 15k, two-GPU joint temporal-benchmark LTE SmolVLA recipe locally.
set -Eeuo pipefail

[[ $# -eq 0 ]] || { echo "Usage: $0" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RMBENCH_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
exec bash "${SCRIPT_DIR}/run_training.sh" "${RMBENCH_ROOT}" lte \
  "${RMBENCH_ROOT}/config/lte/waitatgoal_liftqa_lte_smolvla_2gpu_15k_gpu_queries.env"
