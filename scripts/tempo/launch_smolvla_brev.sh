#!/usr/bin/env bash
# Immutable source deployment for TEMPO SmolVLA; credentials travel separately.
set -Eeuo pipefail
[[ $# -eq 2 || $# -eq 3 ]] || { echo 'Usage: launch_smolvla_brev.sh <instance> <config/tempo/recipe.yaml> [eval-config.yaml]' >&2; exit 2; }
INSTANCE=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RMBENCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XPOLICYLAB_ROOT="${XPOLICYLAB_ROOT:-$(dirname "${RMBENCH_ROOT}")/XPolicyLab}"
EVAL_RECIPE=""
if [[ $# -eq 3 ]]; then
    EVAL_RECIPE="$(realpath --relative-to="${RMBENCH_ROOT}" "$3")"
    [[ "${EVAL_RECIPE}" =~ ^config/tempo/[A-Za-z0-9._-]+\.yaml$ ]] || { echo 'Eval recipe must live in config/tempo' >&2; exit 2; }
fi
BUILD_ROOT="$(mktemp -d /tmp/tempo-smolvla-source-XXXXXX)"
BUILD_JSON="$(python3 "${RMBENCH_ROOT}/scripts/lte/build_source_bundle.py" \
    --rmbench-root "${RMBENCH_ROOT}" --xpolicylab-root "${XPOLICYLAB_ROOT}" \
    --recipe "$(realpath "$2")" --output-dir "${BUILD_ROOT}")"
eval "$(python3 - "${BUILD_JSON}" <<'PY'
import json,shlex,sys
for key,value in json.loads(sys.argv[1]).items():
    print(f'{key.upper()}={shlex.quote(value)}')
PY
)"
ssh -T "${INSTANCE}" 'mkdir -p RMBench-state/bootstrap/tempo; chmod 700 RMBench-state/bootstrap/tempo'
scp "${ARCHIVE}" "${CHECKSUM}" "${INSTANCE}:RMBench-state/bootstrap/tempo/"
if [[ -r "${RMBENCH_ROOT}/scripts/lte/.env" ]]; then
    scp "${RMBENCH_ROOT}/scripts/lte/.env" "${INSTANCE}:RMBench-state/bootstrap/tempo/credentials.env"
fi
ssh -T "${INSTANCE}" bash -s -- "${BUNDLE_ID}" "${RECIPE}" "${EVAL_RECIPE}" <<'REMOTE'
set -Eeuo pipefail
bundle_id=$1
recipe=$2
eval_recipe=$3
[[ "${bundle_id}" =~ ^[0-9a-f]{20}$ && "${recipe}" =~ ^config/tempo/[A-Za-z0-9._-]+\.yaml$ ]]
[[ -z "${eval_recipe}" || "${eval_recipe}" =~ ^config/tempo/[A-Za-z0-9._-]+\.yaml$ ]]
if screen -ls | grep -q '[.]tempo-smolvla-train[[:space:]]'; then
    echo 'TEMPO SmolVLA is already running' >&2; exit 1
fi
bootstrap="${HOME}/RMBench-state/bootstrap/tempo"
state=/data/RMBench-state
release="/data/RMBench-releases/${bundle_id}"
mkdir -p "${state}/logs" "${state}/wandb" "${release}"
cd "${bootstrap}"
sha256sum -c "rmbench-source-${bundle_id}.tar.zst.sha256"
if [[ ! -f "${release}/RMBench/SOURCE_BUNDLE.json" ]]; then
    tar --zstd -xf "rmbench-source-${bundle_id}.tar.zst" -C "${release}" --no-same-owner
fi
if [[ -f credentials.env ]]; then
    chmod 600 credentials.env
    install -m 600 credentials.env "${state}/credentials.env"
fi
root="${release}/RMBench"
log="${state}/logs/tempo-smolvla-$(date -u +%Y%m%dT%H%M%SZ).log"
printf '%s\n' "${root}" > "${state}/tempo_smolvla_source_root.txt"
eval_argument=""
if [[ -n "${eval_recipe}" ]]; then
    [[ -f "${root}/${eval_recipe}" ]]
    eval_argument=" '${root}/${eval_recipe}'"
fi
screen -dmS tempo-smolvla-train bash -lc "{ bash '${root}/scripts/tempo/setup_smolvla_training.sh' '${root}' '${root}/${recipe}' && bash '${root}/scripts/tempo/run_smolvla_remote.sh' '${root}' '${root}/${recipe}'${eval_argument}; } > '${log}' 2>&1"
printf 'Source: %s\nLog: %s\nScreen: tempo-smolvla-train\n' "${root}" "${log}"
REMOTE
