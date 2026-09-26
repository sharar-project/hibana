#!/usr/bin/env bash
# Run the GPU suite (tests/gpu) in the runtime image with this checkout mounted, the model caches, the
# calibrated W3 artifact and the results directory at their container paths. One command instead of the
# six environment variables the suite needs inside a container (#54).
#
# Usage: tools/gpu-tests.sh [pytest arguments]      e.g. tools/gpu-tests.sh tests/gpu/test_qsa.py -x
# Image: SHARAR_TEST_IMAGE, else the one config/flashnext.env serves. Nothing else may use the GPU meanwhile.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
fail() { echo "gpu-tests: $*" >&2; exit 1; }
config_value() { grep -E "^$1=" "$2" 2>/dev/null | tail -1 | cut -d= -f2-; }
GB10_ENV="$REPO_DIR/config/gb10.env"; FN_ENV="$REPO_DIR/config/flashnext.env"
HF_CACHE="${SHARAR_HF_CACHE:-$(config_value SHARAR_HF_CACHE "$GB10_ENV")}"
FN_CACHE="${SHARAR_FLASHNEXT_CACHE:-$(config_value SHARAR_FLASHNEXT_CACHE "$FN_ENV")}"
RESULTS="${SHARAR_RESULTS:-$(config_value SHARAR_RESULTS "$GB10_ENV")}"
W3="${SHARAR_W3_CHECKPOINT:-$(config_value SHARAR_W3_CHECKPOINT "$GB10_ENV")}"
IMAGE="${SHARAR_TEST_IMAGE:-$(config_value SHARAR_IMAGE "$FN_ENV")}"
[[ -d "$HF_CACHE" ]] || fail "missing 27B model cache (SHARAR_HF_CACHE): $HF_CACHE"
[[ -n "$IMAGE" ]] && docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "missing image $IMAGE (SHARAR_TEST_IMAGE)"
DRIVER_LIB_DIR=/usr/lib/aarch64-linux-gnu
source "$SCRIPT_DIR/gb10-container.sh"
args=(docker run --rm --name "sharar-gpu-tests-$$" --security-opt no-new-privileges=true --cap-drop ALL --shm-size 16g
  --user "$(id -u):$(id -g)" --workdir /workspace/sharar
  --volume "$REPO_DIR:/workspace/sharar" --volume "$HF_CACHE:/cache/huggingface:ro"
  --env HOME=/tmp --env HF_HUB_OFFLINE=1 --env PYTHONPATH=/workspace/sharar --env PYTHONDONTWRITEBYTECODE=1
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env TRITON_LIBCUDA_PATH=$DRIVER_LIB_DIR
  --env SHARAR_RUN_GPU=1 --env SHARAR_HF_CACHE=/cache/huggingface)
[[ -d "$FN_CACHE" ]] && args+=(--volume "$FN_CACHE:/cache/flash-next:ro" --env SHARAR_FLASHNEXT_CACHE=/cache/flash-next)
[[ -d "$RESULTS" ]] && args+=(--volume "$RESULTS:/results" --env SHARAR_RESULTS=/results)
# The served 27B profile: calibrated 3-bit decode copies (tests that need the exact path set their own config).
[[ -d "$W3" ]] && args+=(--volume "$W3:/w3:ro" --env SHARAR_W3_CHECKPOINT=/w3 --env SHARAR_W3_DECODE=1)
gb10_container_args
[[ $# -gt 0 ]] || set -- tests/gpu
exec "${args[@]}" --entrypoint "" "$IMAGE" python3 -m pytest -q -p no:cacheprovider "$@"
