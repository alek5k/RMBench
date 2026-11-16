#!/usr/bin/env bash
set -Eeuo pipefail
MODEL_ROOT="${TEMPO_MODEL_ROOT:-/data/RMBench-state/models/pi05_base_pytorch}"
REVISION=3e7adadea2068316728d07e22a2bb84361debb1c
EXPECTED=f722b8d13a9c70ee2fe7451568fce1ded46d28d30f0fb7b89b801237514d85b1
mkdir -p "${MODEL_ROOT}"
if [[ -f "${MODEL_ROOT}/model.safetensors" ]]; then
    printf '%s  %s\n' "${EXPECTED}" "${MODEL_ROOT}/model.safetensors" | sha256sum -c -
    exit
fi
curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    "https://huggingface.co/yyshi0619/pi05-base-pytorch/resolve/${REVISION}/model.safetensors" \
    --output "${MODEL_ROOT}/model.safetensors.partial"
printf '%s  %s\n' "${EXPECTED}" "${MODEL_ROOT}/model.safetensors.partial" | sha256sum -c -
mv "${MODEL_ROOT}/model.safetensors.partial" "${MODEL_ROOT}/model.safetensors"
curl --fail --location --retry 3 \
    "https://huggingface.co/yyshi0619/pi05-base-pytorch/resolve/${REVISION}/config.json" \
    --output "${MODEL_ROOT}/config.json"
printf '%s\n' "yyshi0619/pi05-base-pytorch@${REVISION}" > "${MODEL_ROOT}/SOURCE.txt"
