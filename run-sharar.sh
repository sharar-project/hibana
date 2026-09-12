#!/usr/bin/env bash
# Production launcher for Sharar on one DGX Spark GB10.
# Usage: ./run-sharar.sh [--check | --stop]
# Reads config/gb10.env for SHARAR_HF_CACHE and SHARAR_RESULTS.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SHARAR_CONFIG_FILE:-$SCRIPT_DIR/config/gb10.env}"
[[ -r "$CONFIG_FILE" ]] && source "$CONFIG_FILE"
IMAGE="${SHARAR_IMAGE:-sharar:qwen38-27b}"
CONTAINER="${SHARAR_CONTAINER:-sharar-qwen38}"
PORT="${SHARAR_PORT:-30001}"
SPEC="${SHARAR_SPEC:-15}"   # draft tokens per round (0 = off); 15 is DFlash2's validated depth (16 verify rows)
PROPOSER="${SHARAR_PROPOSER:-dflash2}"
VERIFY_STATE="${SHARAR_VERIFY_STATE:-reconstruct}"
DRAFT_PRECISION="${SHARAR_DRAFT_PRECISION:-bf16}" # fp8 selects the measured optimized parallel draft
TRACES_DIR="${SHARAR_TRACES:-}"                    # opt-in request traces for draft adaptation (unset = off)
TRACES_MAX_MIB="${SHARAR_TRACES_MAX_MIB:-8192}"
DRAFT_REVISION="dedf8df68adfb1afeaf7b7480c0a0243108177b4"   # incoai/Qwen3.8-27B-DFlash2
CACHE_DIR="${SHARAR_HF_CACHE:?set SHARAR_HF_CACHE in config/gb10.env}"
RESULTS_DIR="${SHARAR_RESULTS:?set SHARAR_RESULTS in config/gb10.env}"
STATE_DIR="$RESULTS_DIR/runtime-state"   # JIT caches; persists across restarts
GUARD_UNIT="sharar-memory-guard.service"
GUARD_MIN_AVAILABLE_KIB="${SHARAR_MIN_AVAILABLE_KIB:-16777216}"
MODEL_REVISION="319f741cce68d7914884900c138a1fbb70a42f30"
DRIVER_LIB_DIR="/usr/lib/aarch64-linux-gnu"

fail() { echo "sharar launch failed: $*" >&2; exit 1; }
trim() { local v="$1"; v="${v#"${v%%[![:space:]]*}"}"; printf '%s' "${v%"${v##*[![:space:]]}"}"; }

if [[ "${1:-}" == "--stop" ]]; then
  systemctl --user stop "$GUARD_UNIT" 2>/dev/null || true
  docker stop --timeout 45 "$CONTAINER" >/dev/null 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true
  echo "stopped $CONTAINER"
  exit 0
fi

[[ "$DRAFT_PRECISION" == bf16 || "$DRAFT_PRECISION" == fp8 ]] || fail "SHARAR_DRAFT_PRECISION must be bf16 or fp8"
if [[ "$DRAFT_PRECISION" == fp8 && ( "$SPEC" == 0 || ( "$PROPOSER" != dflash2 && "$PROPOSER" != dspark ) ) ]]; then
  fail "FP8 draft precision requires enabled DFlash2 or DSpark speculation"
fi

# ---- preflight (same facts sharar check enforces inside) ----
[[ "$(uname -m)" == "aarch64" ]] || fail "host architecture must be aarch64"
command -v docker >/dev/null || fail "docker is not installed"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is not installed"
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader,nounits)
[[ ${#gpu_rows[@]} -eq 1 ]] || fail "exactly one NVIDIA GPU is required"
IFS=, read -r gpu_name gpu_cc gpu_driver <<<"${gpu_rows[0]}"
gpu_name="$(trim "$gpu_name")"; gpu_cc="$(trim "$gpu_cc")"; gpu_driver="$(trim "$gpu_driver")"
[[ "$gpu_name" == "NVIDIA GB10" && "$gpu_cc" == "12.1" ]] || fail "GPU must be NVIDIA GB10 SM 12.1 (found $gpu_name $gpu_cc)"
(( 10#${gpu_driver%%.*} >= 580 )) || fail "NVIDIA driver must be 580 or newer"
mem_total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo); mem_avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
(( mem_total >= 120000000 )) || fail "at least 120 GB host memory is required"
(( mem_avail >= 41943040 )) || fail "at least 40 GiB must be available before the measured streaming load (have $((mem_avail / 1048576)) GiB); runtime checks the prospective pool/draft budget separately"
snapshot_ok=false
for root in "$CACHE_DIR" "$CACHE_DIR/hub"; do
  s="$root/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/$MODEL_REVISION"
  [[ -f "$s/config.json" && -f "$s/model.safetensors.index.json" ]] && snapshot_ok=true
done
$snapshot_ok || fail "pinned model revision $MODEL_REVISION is incomplete in $CACHE_DIR"
if [[ "$PROPOSER" == dflash2 && "$SPEC" != 0 ]]; then
  draft_ok=false
  for root in "$CACHE_DIR/hub" "$CACHE_DIR"; do
    d="$root/models--incoai--Qwen3.8-27B-DFlash2/snapshots/$DRAFT_REVISION"
    [[ -f "$d/config.json" ]] && draft_ok=true
  done
  $draft_ok || fail "pinned DFlash2 draft revision $DRAFT_REVISION is missing in $CACHE_DIR (set SHARAR_PROPOSER=mtp to serve without it)"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image is missing: $IMAGE (build with docker/build.sh)"
[[ "$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "io.sharar.engine"}}')" == "sharar" ]] || fail "image is not a Sharar image"

cat <<EOT
Sharar preflight OK
  GPU: $gpu_name (SM $gpu_cc), driver $gpu_driver
  Model: RadixArk/Qwen3.8-27B-NVFP4@$MODEL_REVISION
  Host memory: $((mem_total / 1048576)) GiB total, $((mem_avail / 1048576)) GiB available; runtime floor $((GUARD_MIN_AVAILABLE_KIB / 1048576)) GiB
  Image: $IMAGE -> container $CONTAINER on http://127.0.0.1:$PORT (proposer: $PROPOSER, drafts: $SPEC, draft precision: $DRAFT_PRECISION, verify state: $VERIFY_STATE)
  Traces: ${TRACES_DIR:-off}${TRACES_DIR:+ (cap $TRACES_MAX_MIB MiB)}
EOT
[[ "${1:-}" == "--check" ]] && exit 0

docker container inspect "$CONTAINER" >/dev/null 2>&1 && fail "container '$CONTAINER' already exists (use --stop first)"
mkdir -p "$STATE_DIR"

args=(docker run --detach --name "$CONTAINER"
  --publish "127.0.0.1:$PORT:$PORT"
  --security-opt no-new-privileges=true --cap-drop ALL --pids-limit 2048 --shm-size 16g
  --log-opt max-size=50m --log-opt max-file=2
  --user "$(id -u):$(id -g)"
  --label io.sharar.target=gb10-sm121
  --volume "$CACHE_DIR:/cache/huggingface:ro"
  --volume "$STATE_DIR:/state"
  --env TRITON_LIBCUDA_PATH=$DRIVER_LIB_DIR
  --env SHARAR_SPEC=$SPEC
  --mount type=bind,src=/usr/bin/nvidia-smi,dst=/usr/bin/nvidia-smi,readonly)
for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/dri/renderD128; do
  [[ -e "$dev" ]] || fail "missing device $dev"
  args+=(--device "$dev")
done
add_lib() { local src; src="$(readlink -f "$DRIVER_LIB_DIR/$1")"; [[ -f "$src" ]] || fail "missing driver library $1"; args+=(--mount "type=bind,src=$src,dst=$DRIVER_LIB_DIR/$2,readonly"); }
add_lib libcuda.so.1 libcuda.so.1; add_lib libcuda.so.1 libcuda.so
add_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so.1; add_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so
add_lib libnvidia-ml.so.1 libnvidia-ml.so.1; add_lib libnvidia-ml.so.1 libnvidia-ml.so
add_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so.4; add_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so
add_lib libcudadebugger.so.1 libcudadebugger.so.1
for gpucomp in "$DRIVER_LIB_DIR"/libnvidia-gpucomp.so.*; do [[ -f "$gpucomp" ]] && args+=(--mount "type=bind,src=$gpucomp,dst=$gpucomp,readonly"); done

serve_args=(serve --host 0.0.0.0 --port "$PORT" --proposer "$PROPOSER" --verify-state "$VERIFY_STATE" --spec "$SPEC")
if [[ "$DRAFT_PRECISION" != bf16 ]]; then
  serve_args+=(--draft-precision "$DRAFT_PRECISION")
fi
if [[ -n "$TRACES_DIR" ]]; then
  [[ "$TRACES_MAX_MIB" =~ ^[1-9][0-9]*$ ]] || fail "SHARAR_TRACES_MAX_MIB must be a positive integer"
  mkdir -p "$TRACES_DIR"
  args+=(--volume "$TRACES_DIR:/traces")
  serve_args+=(--traces /traces --traces-max-mib "$TRACES_MAX_MIB")
fi
cid="$("${args[@]}" "$IMAGE" "${serve_args[@]}")"
mkdir -p "$RESULTS_DIR"
echo "$cid" > "$RESULTS_DIR/sharar-container-id.txt"
if ! systemd-run --user --quiet --collect --unit "$GUARD_UNIT" --description "Sharar unified-memory guard" \
     --property Restart=on-failure --property RestartSec=2s \
     --setenv "SHARAR_RESULTS=$RESULTS_DIR" --setenv "SHARAR_MIN_AVAILABLE_KIB=$GUARD_MIN_AVAILABLE_KIB" \
     "$SCRIPT_DIR/tools/memory-guard.sh" "$CONTAINER"; then
  docker stop --time 15 "$CONTAINER" >/dev/null; fail "could not start $GUARD_UNIT; container stopped"
fi
echo "Started $CONTAINER ($cid); logs: docker logs -f $CONTAINER; verify: SHARAR_URL=http://127.0.0.1:$PORT tools/verify-sharar.sh"
