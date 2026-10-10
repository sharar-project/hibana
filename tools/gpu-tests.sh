#!/usr/bin/env bash
# Run the GPU suite (tests/gpu) from this checkout's Python environment with the model caches, the calibrated W3
# artifact and the results directory the launcher uses. Builds the environment first when it is missing or stale.
#
# Usage: tools/gpu-tests.sh [pytest arguments]      e.g. tools/gpu-tests.sh tests/gpu/test_qsa.py -x
# Nothing else may use the GPU meanwhile.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
fail() { echo "gpu-tests: $*" >&2; exit 1; }
# The launcher's paths (config/gb10.env if present, else defaults; tools/launcher-config.sh); the environment wins.
declare -A conf
while IFS='=' read -r k v; do conf[$k]=$v; done < <(bash "$SCRIPT_DIR/launcher-config.sh" print qwen38)
HF_CACHE="${SHARAR_HF_CACHE:-${conf[SHARAR_HF_CACHE]}}"
FN_CACHE="${SHARAR_FLASHNEXT_CACHE:-${conf[SHARAR_FLASHNEXT_CACHE]}}"
RESULTS="${SHARAR_RESULTS:-${conf[SHARAR_RESULTS]}}"
W3="${SHARAR_W3_CHECKPOINT:-${conf[SHARAR_W3_CHECKPOINT]:-}}"
[[ -z "$W3" || "$W3" == auto ]] && W3="$RESULTS/w3-checkpoints/qwen38-current"   # as run-sharar.sh resolves auto
VENV="${SHARAR_VENV:-${conf[SHARAR_VENV]:-$REPO_DIR/.venv}}"
[[ -d "$HF_CACHE" ]] || fail "missing 27B model cache (SHARAR_HF_CACHE): $HF_CACHE"
SHARAR_VENV="$VENV" bash "$SCRIPT_DIR/setup-venv.sh" --check || SHARAR_VENV="$VENV" bash "$SCRIPT_DIR/setup-venv.sh" >&2 \
  || fail "could not build the Python environment (tools/setup-venv.sh)"
envs=(PATH="$VENV/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda PYTHONPATH="$REPO_DIR" PYTHONDONTWRITEBYTECODE=1
  HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SHARAR_RUN_GPU=1 SHARAR_HF_CACHE="$HF_CACHE")
[[ -d "$FN_CACHE" ]] && envs+=(SHARAR_FLASHNEXT_CACHE="$FN_CACHE")
[[ -d "$RESULTS" ]] && envs+=(SHARAR_RESULTS="$RESULTS")
# The served 27B profile: calibrated 3-bit decode copies (tests that need the exact path set their own config).
[[ -d "$W3" ]] && envs+=(SHARAR_W3_CHECKPOINT="$(realpath "$W3")" SHARAR_W3_DECODE=1)
[[ $# -gt 0 ]] || set -- tests/gpu
cd "$REPO_DIR"
exec env "${envs[@]}" "$SCRIPT_DIR/lxcfs-run.sh" "$VENV/bin/python" -m pytest -q -p no:cacheprovider "$@"
