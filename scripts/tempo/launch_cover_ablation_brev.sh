#!/usr/bin/env bash
# Bundle current source; use the existing pinned runtime and verified official data.
set -Eeuo pipefail
[[ $# -eq 2 || $# -eq 3 ]] || { echo 'Usage: launch_cover_ablation_brev.sh <instance> <queue.yaml> [--bootstrap]' >&2; exit 2; }
INSTANCE=$1
BOOTSTRAP=0
if [[ $# -eq 3 ]]; then [[ "$3" == --bootstrap ]] || exit 2; BOOTSTRAP=1; fi
case "$INSTANCE" in normal-scarlet-vole|sound-coral-echidna) ;; *) echo 'Instance outside the current two-L40S study' >&2; exit 2;; esac
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
QUEUE=$(realpath --relative-to="$ROOT" "$2")
[[ "$QUEUE" =~ ^config/tempo/[A-Za-z0-9._-]+\.yaml$ ]] || exit 2
RUN_ID=$("$ROOT/.venv-tempo-smolvla/bin/python" - "$ROOT/$QUEUE" <<'PY'
from pathlib import Path
import sys,yaml
p=Path(sys.argv[1]);q=yaml.safe_load(p.read_text())
print(yaml.safe_load((p.parent/q['training_config']).read_text())['training']['wandb_run_id'])
PY
)
BUILD_ROOT=$(mktemp -d /tmp/cover-ablation-source-XXXXXX)
BUILD_JSON=$(python3 "$ROOT/scripts/lte/build_source_bundle.py" --rmbench-root "$ROOT" \
  --xpolicylab-root "$(dirname "$ROOT")/XPolicyLab" --recipe "$ROOT/$QUEUE" --output-dir "$BUILD_ROOT")
eval "$(python3 - "$BUILD_JSON" <<'PY'
import json,shlex,sys
for key,value in json.loads(sys.argv[1]).items():print(f'{key.upper()}={shlex.quote(value)}')
PY
)"
ssh -T "$INSTANCE" 'mkdir -p RMBench-state/bootstrap/cover-ablation; chmod 700 RMBench-state/bootstrap/cover-ablation'
scp "$ARCHIVE" "$CHECKSUM" "$INSTANCE:RMBench-state/bootstrap/cover-ablation/"
scp "$ROOT/scripts/lte/.env" "$INSTANCE:RMBench-state/bootstrap/cover-ablation/credentials.env"
if [[ "$BOOTSTRAP" -eq 1 ]]; then
  # Only reference configuration and small public-dataset normalization stats;
  # no trained weights, private datasets, environments or model caches.
  REFERENCE="$ROOT/logs/cover-blocks-queries128-300k-261110/reference/query-study"
  ssh -T "$INSTANCE" 'mkdir -p RMBench-state/bootstrap/cover-ablation/reference'
  scp "$REFERENCE/reference_resolved_config.json" "$INSTANCE:RMBench-state/bootstrap/cover-ablation/reference/"
  scp -r "$REFERENCE/processors" "$INSTANCE:RMBench-state/bootstrap/cover-ablation/reference/"
fi
ssh -T "$INSTANCE" bash -s -- "$BUNDLE_ID" "$QUEUE" "$RUN_ID" "$BOOTSTRAP" <<'REMOTE'
set -Eeuo pipefail
umask 077
bundle_id=$1
queue=$2
run_id=$3
bootstrap_fresh=$4
[[ "$bundle_id" =~ ^[0-9a-f]{20}$ && "$queue" =~ ^config/tempo/[A-Za-z0-9._-]+\.yaml$ && "$run_id" =~ ^[0-9a-f]{8}$ ]] || exit 2
bootstrap="$HOME/RMBench-state/bootstrap/cover-ablation"
state=/data/RMBench-state
release="/data/RMBench-releases/$bundle_id"
python="$HOME/RMBench-state/envs/tempo-smolvla/bin/python"
if [[ "$bootstrap_fresh" -eq 0 ]]; then
  [[ -x "$python" ]] || python="$state/envs/tempo-smolvla/bin/python"
  [[ -x "$python" ]]
fi
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo 'GPU occupied; nothing interrupted' >&2; exit 1; }
cd "$bootstrap"
sha256sum -c "rmbench-source-$bundle_id.tar.zst.sha256"
mkdir -p "$release" "$state/logs"
if [[ ! -f "$release/RMBench/SOURCE_BUNDLE.json" ]]; then
  tar --zstd -xf "rmbench-source-$bundle_id.tar.zst" -C "$release" --no-same-owner
fi
chmod 600 credentials.env
install -m 600 credentials.env "$state/credentials.env"
root="$release/RMBench"
if [[ "$bootstrap_fresh" -eq 1 ]]; then
  mkdir -p "$state/references/cover-blocks-query-study" "$state/assets/objects" "$state/assets/embodiments"
  cp -a "$bootstrap/reference/." "$state/references/cover-blocks-query-study/"
fi
for name in objects embodiments; do
  [[ -d "$state/assets/$name" ]]
  [[ -e "$root/assets/$name" ]] || ln -s "$state/assets/$name" "$root/assets/$name"
done
[[ ! -e "$state/cover-ablation-$run_id.json" ]] || { echo 'Existing queue retained; refusing overwrite' >&2; exit 1; }
log="$state/logs/cover-ablation-$run_id.log"
[[ ! -e "$log" ]] || { echo 'Existing run log retained; refusing overwrite' >&2; exit 1; }
if [[ "$bootstrap_fresh" -eq 1 ]]; then
  screen -dmS "cover-ablation-$run_id" bash -lc "{ bash '$root/scripts/tempo/setup_cover_ablation_fresh.sh' '$root' '$root/$queue' && bash '$root/scripts/tempo/run_cover_ablation_remote.sh' '$root' '$root/$queue'; } > '$log' 2>&1"
else
  screen -dmS "cover-ablation-$run_id" bash -lc "exec bash '$root/scripts/tempo/run_cover_ablation_remote.sh' '$root' '$root/$queue' > '$log' 2>&1"
fi
printf 'Source: %s\nRun ID: %s\nLog: %s\nScreen: cover-ablation-%s\n' "$root" "$run_id" "$log" "$run_id"
REMOTE
