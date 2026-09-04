#!/usr/bin/env bash
# Launch the frozen SGLang reference image with its own MTP speculative
# decoding (NEXTN, 3 steps, 4 draft tokens) on port 30002 for a like-for-like
# comparison with Sharar. Same model, same profile flags as the reference.
# Usage: tools/reference/run-sglang-spec.sh [--stop]
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/gb10.env"
IMAGE="${SHARAR_DEV_IMAGE:-sparklang-gb10-engine:qwen38-27b}"
CONTAINER="sglang-spec-qwen38"
PORT=30002
DRIVER_LIB_DIR="/usr/lib/aarch64-linux-gnu"
if [[ "${1:-}" == "--stop" ]]; then docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; echo stopped; exit 0; fi
args=(docker run --detach --name "$CONTAINER" --publish "127.0.0.1:$PORT:$PORT"
  --security-opt no-new-privileges=true --cap-drop ALL --pids-limit 2048 --shm-size 16g
  --user "$(id -u):$(id -g)"
  --volume "$SHARAR_HF_CACHE:/cache/huggingface"
  --env HF_HOME=/cache/huggingface --env HOME=/tmp/sglang-home --env TOKENIZERS_PARALLELISM=false
  --env TRITON_LIBCUDA_PATH=$DRIVER_LIB_DIR --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --mount type=bind,src=/usr/bin/nvidia-smi,dst=/usr/bin/nvidia-smi,readonly)
for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/dri/renderD128; do args+=(--device "$dev"); done
add_lib() { local src; src="$(readlink -f "$DRIVER_LIB_DIR/$1")"; args+=(--mount "type=bind,src=$src,dst=$DRIVER_LIB_DIR/$2,readonly"); }
add_lib libcuda.so.1 libcuda.so.1; add_lib libcuda.so.1 libcuda.so
add_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so.1; add_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so
add_lib libnvidia-ml.so.1 libnvidia-ml.so.1; add_lib libnvidia-ml.so.1 libnvidia-ml.so
add_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so.4; add_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so
add_lib libcudadebugger.so.1 libcudadebugger.so.1
for gpucomp in "$DRIVER_LIB_DIR"/libnvidia-gpucomp.so.*; do [[ -f "$gpucomp" ]] && args+=(--mount "type=bind,src=$gpucomp,dst=$gpucomp,readonly"); done
"${args[@]}" "$IMAGE" sglang serve \
  --model-path RadixArk/Qwen3.8-27B-NVFP4 --revision 319f741cce68d7914884900c138a1fbb70a42f30 \
  --trust-remote-code --language-model-only --host 0.0.0.0 --port $PORT \
  --context-length 32768 --max-total-tokens 32768 --kv-cache-dtype auto --mem-fraction-static 0.70 \
  --max-running-requests 4 --max-queued-requests 16 --chunked-prefill-size 2048 \
  --attention-backend flashinfer --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 20 --cuda-graph-max-bs-decode 4 --cuda-graph-max-bs-prefill 4 \
  --reasoning-parser auto --tool-call-parser auto \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  "$@"
echo "started $CONTAINER on http://127.0.0.1:$PORT (docker logs -f $CONTAINER)"
