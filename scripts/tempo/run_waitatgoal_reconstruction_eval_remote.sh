#!/usr/bin/env bash
# Prepare a simulator-only overlay, smoke the final checkpoint, then evaluate it.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_waitatgoal_reconstruction_eval_remote.sh <source-root> <eval.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
unset LD_LIBRARY_PATH
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
bash "$ROOT/scripts/tempo/setup_temporal_eval_remote.sh" "$ROOT"
bash "$ROOT/scripts/tempo/run_temporal_eval_remote.sh" "$ROOT" "$CONFIG" --smoke
exec bash "$ROOT/scripts/tempo/run_temporal_eval_remote.sh" "$ROOT" "$CONFIG"
