#!/usr/bin/env bash
# Deploy an immutable RMBench + standalone XPolicyLab source bundle.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 <brev-instance> <config/tempo/recipe.yaml>" >&2; exit 2; }
INSTANCE=$1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RMBENCH_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
XPOLICYLAB_ROOT="${XPOLICYLAB_ROOT:-$(dirname "${RMBENCH_ROOT}")/XPolicyLab}"
CONFIG="$(realpath "$2")"
BUILD_ROOT="$(mktemp -d /tmp/tempo-source-XXXXXX)"
BUILD_JSON="$(python3 "${RMBENCH_ROOT}/scripts/lte/build_source_bundle.py" \
    --rmbench-root "${RMBENCH_ROOT}" --xpolicylab-root "${XPOLICYLAB_ROOT}" \
    --recipe "${CONFIG}" --output-dir "${BUILD_ROOT}")"
eval "$(python3 - "${BUILD_JSON}" <<'PY'
import json, shlex, sys
for key, value in json.loads(sys.argv[1]).items():
    print(f'{key.upper()}={shlex.quote(value)}')
PY
)"
REMOTE_BOOTSTRAP=RMBench-state/bootstrap/tempo
ssh -T "${INSTANCE}" "mkdir -p '${REMOTE_BOOTSTRAP}'; chmod 700 '${REMOTE_BOOTSTRAP}'"
scp "${ARCHIVE}" "${CHECKSUM}" "${INSTANCE}:${REMOTE_BOOTSTRAP}/"
if [[ -r "${RMBENCH_ROOT}/scripts/lte/.env" ]]; then
    scp "${RMBENCH_ROOT}/scripts/lte/.env" "${INSTANCE}:${REMOTE_BOOTSTRAP}/credentials.env"
fi
ssh -T "${INSTANCE}" "bash -s" -- "${BUNDLE_ID}" "${RECIPE}" <<'REMOTE'
set -Eeuo pipefail
bundle_id=$1
recipe=$2
[[ "${bundle_id}" =~ ^[0-9a-f]{20}$ && "${recipe}" =~ ^config/tempo/[A-Za-z0-9._-]+\.yaml$ ]]
bootstrap="${HOME}/RMBench-state/bootstrap/tempo"
state=/data/RMBench-state
release="/data/RMBench-releases/${bundle_id}"
mkdir -p "${state}"/{logs,wandb} "${release}"
cd "${bootstrap}"
sha256sum -c "rmbench-source-${bundle_id}.tar.zst.sha256"
if [[ ! -f "${release}/RMBench/SOURCE_BUNDLE.json" ]]; then
    tar --zstd -xf "rmbench-source-${bundle_id}.tar.zst" -C "${release}" --no-same-owner
fi
if [[ -f credentials.env ]]; then
    chmod 600 credentials.env
    install -m 600 credentials.env "${state}/credentials.env"
fi
if screen -ls | grep -q '[.]tempo-battery-100k[[:space:]]'; then
    echo 'TEMPO screen session is already running; refusing duplicate launch.' >&2
    exit 1
fi
root="${release}/RMBench"
printf '%s\n' "${root}" > "${state}/tempo_source_root.txt"
screen -dmS tempo-battery-100k bash -lc "bash '${root}/scripts/tempo/run_remote.sh' '${root}' '${root}/${recipe}' > '${state}/logs/tempo-battery-100k.log' 2>&1"
printf 'Source: %s\nLog: %s\nScreen: tempo-battery-100k\n' "${root}" "${state}/logs/tempo-battery-100k.log"
REMOTE
printf 'Bundle: %s\nLocal manifest: %s\n' "${BUNDLE_ID}" "${MANIFEST}"
