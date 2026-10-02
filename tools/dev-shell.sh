#!/usr/bin/env bash
# Start a development container from the Sharar runtime image (SHARAR_DEV_IMAGE, else the
# config's SHARAR_IMAGE) with this repository mounted over the image's copy.
#
# Usage: tools/dev-shell.sh [--tty] [--serve PORT] [--] [command...]
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${SHARAR_CONFIG_FILE:-$REPO_DIR/config/gb10.env}"
if [[ -r "$CONFIG_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
fi
fail() { echo "dev-shell: $*" >&2; exit 1; }
IMAGE="${SHARAR_DEV_IMAGE:-${SHARAR_IMAGE:-sharar:qwen38-27b}}"
CACHE_DIR="${SHARAR_HF_CACHE:?set SHARAR_HF_CACHE in config/gb10.env}"
RESULTS_DIR="${SHARAR_RESULTS:?set SHARAR_RESULTS in config/gb10.env}"
DRIVER_LIB_DIR="/usr/lib/aarch64-linux-gnu"
NAME="${SHARAR_DEV_CONTAINER:-sharar-dev-$$}"

tty=false
serve_port=""
while [[ "${1:-}" == --* && "${1:-}" != "--" ]]; do
  case "$1" in
    --tty) tty=true ;;
    --serve) serve_port="$2"; shift ;;
    *) fail "unknown option $1" ;;
  esac
  shift
done
[[ "${1:-}" == "--" ]] && shift

[[ "$(uname -m)" == "aarch64" ]] || fail "host must be aarch64"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "missing image $IMAGE"
[[ -d "$CACHE_DIR" ]] || fail "missing model cache $CACHE_DIR"
mkdir -p "$RESULTS_DIR/home"

pythonpath="/workspace/sharar"

devices=(
  /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools
  /dev/nvidia-modeset /dev/dri/renderD128
)
network_args=(--network none)
if [[ -n "$serve_port" ]]; then
  # Serving: loopback-only publish; the server binds 0.0.0.0 inside the container.
  network_args=(--publish "127.0.0.1:$serve_port:$serve_port")
fi
args=(
  docker run --rm
  --name "$NAME"
  --entrypoint ""
  "${network_args[@]}"
  --security-opt no-new-privileges=true
  --cap-drop ALL
  --pids-limit 4096
  --shm-size 16g
  --user "$(id -u):$(id -g)"
  --workdir /workspace/sharar
  --volume "$REPO_DIR:/workspace/sharar"
  --volume "$CACHE_DIR:/cache/huggingface:ro"
  --volume "$RESULTS_DIR:/results"
  --env HF_HOME=/cache/huggingface
  --env HF_HUB_OFFLINE=1
  --env HOME=/results/home
  --env "PYTHONPATH=$pythonpath"
  --env PYTHONDONTWRITEBYTECODE=1
  --env TOKENIZERS_PARALLELISM=false
  --env TRITON_LIBCUDA_PATH=/usr/lib/aarch64-linux-gnu
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env SHARAR_RESULTS=/results
  --env SHARAR_HF_CACHE=/cache/huggingface
  --env SHARAR_REFERENCE_URL="${SHARAR_REFERENCE_URL:-http://127.0.0.1:30000}"
)
for device in "${devices[@]}"; do
  [[ -e "$device" ]] || fail "missing device $device"
  args+=(--device "$device")
done
add_lib() {
  local src; src="$(readlink -f "$DRIVER_LIB_DIR/$1")"
  [[ -f "$src" ]] || fail "missing driver library $1"
  args+=(--mount "type=bind,src=$src,dst=$DRIVER_LIB_DIR/$2,readonly")
}
add_lib libcuda.so.1 libcuda.so.1
add_lib libcuda.so.1 libcuda.so
add_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so.1
add_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so
add_lib libnvidia-ml.so.1 libnvidia-ml.so.1
add_lib libnvidia-ml.so.1 libnvidia-ml.so
add_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so.4
add_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so
add_lib libcudadebugger.so.1 libcudadebugger.so.1
for gpucomp in "$DRIVER_LIB_DIR"/libnvidia-gpucomp.so.*; do
  [[ -f "$gpucomp" ]] && args+=(--mount "type=bind,src=$gpucomp,dst=$gpucomp,readonly")
done
args+=(--mount type=bind,src=/usr/bin/nvidia-smi,dst=/usr/bin/nvidia-smi,readonly)

if [[ $# -eq 0 ]]; then
  exec "${args[@]}" --interactive --tty "$IMAGE" bash
fi
if $tty; then
  exec "${args[@]}" --interactive --tty "$IMAGE" "$@"
fi
exec "${args[@]}" "$IMAGE" "$@"
