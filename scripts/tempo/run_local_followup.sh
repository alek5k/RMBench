#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $# -eq 1 ]] || { echo 'Usage: run_local_followup.sh <followup-config.yaml>' >&2; exit 2; }
if [[ -f "${ROOT}/scripts/lte/.env" ]]; then
  set -a
  source "${ROOT}/scripts/lte/.env"
  set +a
fi
exec "${ROOT}/.venv-tempo-smolvla/bin/python" "${ROOT}/scripts/tempo/followup_local_eval.py" --config "$1"
