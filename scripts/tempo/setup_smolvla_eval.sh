#!/usr/bin/env bash
# Provision a separate simulator environment, leaving the active policy env alone.
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: setup_smolvla_eval.sh <RMBench-root>' >&2; exit 2; }
ROOT=$(realpath "$1")
export RMBENCH_STATE_ROOT=/data/RMBench-state
export RMBENCH_CREDENTIAL_FILE=/data/RMBench-state/credentials.env
mkdir -p "${RMBENCH_STATE_ROOT}/assets/objects" "${RMBENCH_STATE_ROOT}/assets/embodiments"
exec 9>"${RMBENCH_STATE_ROOT}/tempo-eval-setup.lock"
flock 9
if [[ ! -x "${HOME}/miniforge3/bin/conda" ]]; then
    installer="${HOME}/RMBench-state/bootstrap/eval/Miniforge3-Linux-x86_64.sh"
    mkdir -p "$(dirname "$installer")"
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$installer"
    bash "$installer" -b -p "${HOME}/miniforge3"
fi
for name in objects embodiments; do
    if [[ ! -e "${ROOT}/assets/${name}" ]]; then
        ln -s "${RMBENCH_STATE_ROOT}/assets/${name}" "${ROOT}/assets/${name}"
    fi
done
export PATH="${HOME}/miniforge3/bin:${PATH}"
export PYTHONPATH="${ROOT}:${ROOT}/XPolicyLab"
if [[ ! -f "${RMBENCH_STATE_ROOT}/tempo-eval-environment-ready" ]]; then
    bash "${ROOT}/scripts/lte/setup_eval_remote.sh" "$ROOT"
fi
cd "$ROOT"
python_eval="${HOME}/miniforge3/envs/rmbench-eval/bin/python"
"$python_eval" scripts/update_embodiment_config_path.py
"$python_eval" -c 'import torch, sapien, mplib, curobo, envs; print("Simulator imports passed")'
touch "${RMBENCH_STATE_ROOT}/tempo-eval-environment-ready"
