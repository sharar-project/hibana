#!/usr/bin/env bash
# Production launcher for Sharar on one DGX Spark GB10.
# Usage: ./run-sharar.sh [--check | --stop]   (sharar-ctl serve 27b|flashnext drives it)
# Settings: the environment and config/gb10.env (optional), then the target's config/targets/<target>.env, then
# defaults (tools/launcher-config.sh). A model or draft missing from the cache is downloaded first.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SHARAR_CONFIG_FILE:-$SCRIPT_DIR/config/gb10.env}"
REQUESTED_TARGET="${SHARAR_TARGET:-}"   # a target named by the caller (sharar-ctl serve flashnext) wins over the config file
[[ -r "$CONFIG_FILE" ]] && source "$CONFIG_FILE"
[[ -n "$REQUESTED_TARGET" ]] && SHARAR_TARGET="$REQUESTED_TARGET"
source "$SCRIPT_DIR/tools/launcher-config.sh"
launcher_defaults || exit 1
# Serving profile: the default for every speed/fidelity knob below that the environment or the config file
# does not set explicitly (tools/serving-profile.sh).
source "$SCRIPT_DIR/tools/serving-profile.sh"
source "$SCRIPT_DIR/tools/image.sh"
PROFILE="${SHARAR_PROFILE:-fast}"
serving_profile "$PROFILE" "${SHARAR_TARGET:-}" || { echo "sharar launch failed: SHARAR_PROFILE must be exact, balanced, fast or throughput" >&2; exit 1; }
IMAGE="${SHARAR_IMAGE:-$SHARAR_DEFAULT_IMAGE}"   # tools/image.sh: the published nightly image unless set
CONTAINER="${SHARAR_CONTAINER:-sharar-qwen38}"
PORT="${SHARAR_PORT:-30001}"
SPEC="${SHARAR_SPEC:-15}"   # draft tokens per round (0 = off); 15 is DFlash2's validated depth (16 verify rows)
PROPOSER="${SHARAR_PROPOSER:-dflash2}"
VERIFY_STATE="${SHARAR_VERIFY_STATE:-reconstruct}"
DRAFT_PRECISION="${SHARAR_DRAFT_PRECISION:-fp8}"  # the measured serving profile (2.5 GB draft, and the 3-bit draft copy applies to it); bf16 is the publisher precision
TRACES_DIR="${SHARAR_TRACES:-}"                    # opt-in request traces for draft adaptation (unset = off)
DRAFT_DIR="${SHARAR_DRAFT:-}"                      # trained draft checkpoint directory; unset follows draft-runs/current; 'pinned' forces the publisher draft
TRACES_MAX_MIB="${SHARAR_TRACES_MAX_MIB:-32768}"
MAX_CONTEXT="${SHARAR_MAX_CONTEXT:-262144}"   # prompt plus output per request, up to the model's native 262144
MAX_BATCH="${SHARAR_MAX_BATCH:-$P_MAX_BATCH}"       # most requests decoded together, 1-16
SPEC_DEPTHS="${SHARAR_SPEC_DEPTHS:-$P_SPEC_DEPTHS}"  # verify depth per batch bucket, e.g. 8:7,16:3 (0: plain decode; empty: SHARAR_SPEC everywhere)
ACCEPT_DEFAULT="${SHARAR_ACCEPT_DEFAULT:-$P_ACCEPT}"      # relaxed acceptance for greedy requests that do not choose (K:M; off = exact)
ACCEPT_DEFAULT_SAMPLED="${SHARAR_ACCEPT_DEFAULT_SAMPLED:-$P_ACCEPT_SAMPLED}"  # relaxed acceptance for sampled requests too (1/0)
ACCEPT_DEFAULT_SAMPLED_THRESHOLD="${SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD:-$P_ACCEPT_SAMPLED_THRESHOLD}"  # their K:M (empty: ACCEPT_DEFAULT's)
DEVICE_ACCEPT="${SHARAR_DEVICE_ACCEPT:-$P_DEVICE_ACCEPT}"  # device-resident acceptance graphs (engine/accept_graphs.py; 1/0)
W3_DECODE="${SHARAR_W3_DECODE:-$P_W3}"   # calibrated 3-bit decode copies for calls of at most 64 rows (1/0; the profile decides)
W3_CHECKPOINT="${SHARAR_W3_CHECKPOINT:-auto}" # where they live: auto ($SHARAR_RESULTS/w3-checkpoints/<target>-current, fetched on first serve) or a directory
TARGET_NAME="${SHARAR_TARGET:-qwen38}"
if [[ "$PROPOSER" == dspark ]]; then   # the pinned parallel draft (sharar/proposers/parallel.py PINS)
  DRAFT_REPO="RadixArk/Qwen3.8-27B-DSpark"; DRAFT_REVISION="b9a5dbdf03bc999c6c73c426b19c2d9041cea393"
else
  DRAFT_REPO="incoai/Qwen3.8-27B-DFlash2"; DRAFT_REVISION="dedf8df68adfb1afeaf7b7480c0a0243108177b4"
fi
CACHE_DIR="$SHARAR_HF_CACHE"
RESULTS_DIR="$SHARAR_RESULTS"
STATE_DIR="$RESULTS_DIR/runtime-state"   # JIT caches; persists across restarts
GUARD_UNIT="sharar-memory-guard.service"
GUARD_MIN_AVAILABLE_KIB="${SHARAR_MIN_AVAILABLE_KIB:-16777216}"
# Pinned checkpoints (sharar/target.py and sharar/model/flashnext/config.py; tests/unit/test_launcher_config.py)
if [[ "$TARGET_NAME" == flashnext ]]; then
  MODEL_REPO="RadixArk/Qwen3.8-Flash-Next-NVFP4"; MODEL_REVISION="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
  MODEL_CACHE="$SHARAR_FLASHNEXT_CACHE"
else
  MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"; MODEL_REVISION="319f741cce68d7914884900c138a1fbb70a42f30"
  MODEL_CACHE="$CACHE_DIR"
fi
RUNS_DIR="${SHARAR_DRAFT_RUNS:-$RESULTS_DIR/draft-runs}"
if [[ "$DRAFT_DIR" == pinned ]]; then
  DRAFT_DIR=""
elif [[ -z "$DRAFT_DIR" && -L "$RUNS_DIR/current" ]]; then
  current="$(readlink -f "$RUNS_DIR/current")"
  # Follow the promoted checkpoint only when it is this proposer's architecture (trained drafts are DFlash2 today).
  want=$([[ "$PROPOSER" == dspark ]] && echo DSparkDraftModel || echo DFlash2DraftModel)
  [[ -f "$current/config.json" && -f "$current/model.safetensors" ]] && grep -q "\"$want\"" "$current/config.json" && DRAFT_DIR="$current"
fi
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
for v in W3_DECODE DEVICE_ACCEPT; do [[ "${!v}" == 0 || "${!v}" == 1 ]] || fail "SHARAR_$v must be 0 or 1"; done
[[ "$MAX_BATCH" =~ ^[0-9]+$ ]] && (( MAX_BATCH >= 1 && MAX_BATCH <= 16 )) || fail "SHARAR_MAX_BATCH must be in [1, 16]"
[[ -z "$SPEC_DEPTHS" || "$SPEC_DEPTHS" =~ ^[0-9]+:[0-9]+(,[0-9]+:[0-9]+)*$ ]] || fail "SHARAR_SPEC_DEPTHS must be BUCKET:DEPTH pairs, e.g. 8:1,16:0"
[[ "$W3_DECODE" == 1 ]] || W3_CHECKPOINT=""   # the checkpoint is where calibrated copies live; without decode copies it is unused
if [[ -n "$W3_CHECKPOINT" && "$W3_CHECKPOINT" != auto ]]; then
  [[ -f "$W3_CHECKPOINT/manifest.json" ]] || fail "SHARAR_W3_CHECKPOINT needs a completed manifest.json"
  W3_CHECKPOINT="$(realpath "$W3_CHECKPOINT")"
fi
if [[ "$DRAFT_PRECISION" == fp8 && ( "$SPEC" == 0 || ( "$PROPOSER" != dflash2 && "$PROPOSER" != dspark ) ) ]]; then
  fail "FP8 draft precision requires enabled DFlash2 or DSpark speculation"
fi

# ---- preflight (same facts sharar check enforces inside) ----
[[ "$(uname -m)" == "aarch64" ]] || fail "host architecture must be aarch64"
command -v docker >/dev/null || fail "docker is not installed (Docker Engine; the NVIDIA container toolkit is not needed)"
docker info >/dev/null 2>&1 || fail "$(id -un) cannot reach the Docker daemon: sudo usermod -aG docker $(id -un), then log in again"
# The memory guard runs as a user systemd unit beside the server (tools/memory-guard.sh).
{ command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; } \
  || fail "no user systemd session for the memory guard: log in with a full session, or sudo loginctl enable-linger $(id -un)"
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
# snapshot_complete CACHE REPO REVISION: the snapshot has config.json and every weight file its index names.
snapshot_complete() {
  local root
  for root in "$1" "$1/hub"; do
    python3 - "$root/models--${2//\//--}/snapshots/$3" <<'PY' && return 0
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
index = d / "model.safetensors.index.json"
files = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
sys.exit(0 if (d / "config.json").is_file() and all((d / f).is_file() for f in files) else 1)
PY
  done
  return 1
}
downloads=()   # "REPO REVISION CACHE" for each pinned snapshot the cache lacks
snapshot_complete "$MODEL_CACHE" "$MODEL_REPO" "$MODEL_REVISION" || downloads+=("$MODEL_REPO $MODEL_REVISION $MODEL_CACHE")
if [[ ( "$PROPOSER" == dflash2 || "$PROPOSER" == dspark ) && "$SPEC" != 0 && -z "$DRAFT_DIR" ]]; then
  snapshot_complete "$CACHE_DIR" "$DRAFT_REPO" "$DRAFT_REVISION" || downloads+=("$DRAFT_REPO $DRAFT_REVISION $CACHE_DIR")
fi
if [[ "$PROPOSER" == dspark && "$SPEC" != 0 && "$MAX_CONTEXT" -gt 32768 ]]; then
  # DSpark keeps draft features for the whole context (DFlash2 keeps a 2,048-token window): at 262K they do not fit.
  fail "DSpark needs SHARAR_MAX_CONTEXT of at most 32768 (it keeps draft features for the whole context; got $MAX_CONTEXT)"
fi
ensure_image "$IMAGE" $([[ "${1:-}" == --check ]] && echo --no-pull) || fail "no runtime image ($IMAGE)"
for d in "${downloads[@]}"; do
  read -r repo revision cache <<<"$d"
  if [[ "${1:-}" == --check ]]; then
    echo "sharar: $repo@${revision:0:7} is not in $cache; serving downloads it first" >&2
    continue
  fi
  mkdir -p "$cache/hub"
  docker run --rm $([[ -t 1 ]] && echo --tty) --user "$(id -u):$(id -g)" --volume "$cache:/models" \
    --volume "$SCRIPT_DIR/tools/fetch-model.py:/opt/fetch-model.py:ro" --env HOME=/tmp --env HF_HOME=/tmp/hf --env HF_HUB_OFFLINE=0 \
    ${HF_TOKEN:+--env HF_TOKEN} --entrypoint python3 "$IMAGE" /opt/fetch-model.py "$repo" "$revision" /models/hub "$cache" \
    || fail "could not download $repo@${revision:0:7} into $cache"
  snapshot_complete "$cache" "$repo" "$revision" || fail "$repo@${revision:0:7} is still incomplete in $cache after the download"
done
if [[ -n "$TRACES_DIR" || -n "$DRAFT_DIR" ]]; then
  docker run --rm --entrypoint python3 "$IMAGE" -m sharar serve --help 2>/dev/null | grep -q -- "--traces" \
    || fail "image $IMAGE predates trace recording and trained drafts; rebuild it with docker/build.sh"
fi
[[ "$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "io.sharar.engine"}}')" == "sharar" ]] || fail "image is not a Sharar image"
# Calibrated 3-bit copies (#72): auto serves $SHARAR_RESULTS/w3-checkpoints/<target>-current and, when that is
# missing, fetches the target's published checkpoint into it (weights/w3_checkpoint.py; not during --check).
# Without one Flash-Next serves its FP8 trunk and the 27B nearest-rounded copies, with a warning.
W3_WARNING=""
if [[ "$W3_CHECKPOINT" == auto ]]; then
  W3_ROOT="$RESULTS_DIR/w3-checkpoints"
  if [[ ! -f "$W3_ROOT/$TARGET_NAME-current/manifest.json" && "${1:-}" != "--check" ]]; then
    mkdir -p "$W3_ROOT"
    docker run --rm --user "$(id -u):$(id -g)" --volume "$W3_ROOT:/w3-checkpoints" --env HF_HOME=/tmp/hf --env HF_HUB_OFFLINE=0 \
      ${HF_TOKEN:+--env HF_TOKEN} --entrypoint python3 "$IMAGE" -m sharar.weights.w3_checkpoint fetch \
      --target "$TARGET_NAME" --root /w3-checkpoints >&2 || echo "sharar: no calibrated 3-bit checkpoint could be fetched for $TARGET_NAME" >&2
  fi
  W3_CHECKPOINT=""
  [[ -f "$W3_ROOT/$TARGET_NAME-current/manifest.json" ]] && W3_CHECKPOINT="$(realpath "$W3_ROOT/$TARGET_NAME-current")"
fi
if [[ "$W3_DECODE" == 1 && -z "$W3_CHECKPOINT" ]]; then
  if [[ "$TARGET_NAME" == flashnext ]]; then
    W3_DECODE=0
    W3_WARNING="no calibrated 3-bit checkpoint: serving the FP8 trunk, about 24% slower single-stream than $PROFILE with it"
  else
    W3_WARNING="no calibrated 3-bit checkpoint: nearest-rounded copies (about 2 GSM8K points and 8% tokens per round below calibrated)"
  fi
  W3_WARNING="$W3_WARNING; calibrate with docs/w3-calibration.md or set SHARAR_W3_CHECKPOINT"
fi

onoff() { [[ "$1" == 1 ]] && echo on || echo off; }
cat <<EOT
Sharar preflight OK
  GPU: $gpu_name (SM $gpu_cc), driver $gpu_driver
  Model: $MODEL_REPO@$MODEL_REVISION (${MODEL_CACHE})
  Host memory: $((mem_total / 1048576)) GiB total, $((mem_avail / 1048576)) GiB available; runtime floor $((GUARD_MIN_AVAILABLE_KIB / 1048576)) GiB
  Image: $IMAGE -> container $CONTAINER on http://127.0.0.1:$PORT (proposer: $PROPOSER, drafts: $SPEC, draft precision: $DRAFT_PRECISION, verify state: $VERIFY_STATE)
  Traces: ${TRACES_DIR:-off}${TRACES_DIR:+ (cap $TRACES_MAX_MIB MiB)}${TRACES_DIR:+, draft confidence $([[ "${SHARAR_TRACE_CONFIDENCE:-0}" == 1 ]] && echo on || echo off) (SHARAR_TRACE_CONFIDENCE)}
  Cold tier: ${SHARAR_COLD_TIER_GIB:-automatic (spare memory up to 24 GiB)}${SHARAR_COLD_TIER_GIB:+ GiB (SHARAR_COLD_TIER_GIB)}
  Device-resident acceptance: $(onoff "$DEVICE_ACCEPT") (SHARAR_DEVICE_ACCEPT; the fast profile turns it on for Flash-Next)
  Serving profile: $PROFILE (SHARAR_PROFILE: exact / balanced / fast / throughput; docs/operations.md#serving-profiles)
  Batch: up to $MAX_BATCH requests (SHARAR_MAX_BATCH), verify depth by batch ${SPEC_DEPTHS:-$SPEC everywhere} (SHARAR_SPEC_DEPTHS)
  Relaxed acceptance default: $ACCEPT_DEFAULT, sampled requests $(onoff "$ACCEPT_DEFAULT_SAMPLED")${ACCEPT_DEFAULT_SAMPLED_THRESHOLD:+ at $ACCEPT_DEFAULT_SAMPLED_THRESHOLD} (requests may set accept_top_k/accept_margin; 1 = exact)
  Three-bit decode copies: $(onoff "$W3_DECODE")$([[ "$W3_DECODE" == 1 ]] && echo ", ${W3_CHECKPOINT:-nearest rounding from the pinned serving weights}") (SHARAR_W3_DECODE, SHARAR_W3_CHECKPOINT)
  Draft: $([[ "$PROPOSER" == mtp ]] && echo "the model's own MTP head" || echo "${DRAFT_DIR:-pinned $DRAFT_REPO@${DRAFT_REVISION:0:7}}")
EOT
[[ -n "$W3_WARNING" ]] && echo "  WARNING: $W3_WARNING"
[[ "${1:-}" == "--check" ]] && exit 0

docker container inspect "$CONTAINER" >/dev/null 2>&1 && fail "container '$CONTAINER' already exists (use --stop first)"
mkdir -p "$STATE_DIR" "$CACHE_DIR"   # existing directories, so docker does not create a mount point as root

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
  --env SHARAR_W3_DECODE=$W3_DECODE --env SHARAR_PROFILE=$PROFILE
)
[[ -n "$W3_WARNING" ]] && args+=(--env "SHARAR_W3_WARNING=$W3_WARNING")   # reported in /get_server_info
source "$SCRIPT_DIR/tools/gb10-container.sh"
gb10_container_args

serve_args=(serve --host 0.0.0.0 --port "$PORT" --proposer "$PROPOSER" --verify-state "$VERIFY_STATE" --spec "$SPEC")
if [[ "$DRAFT_PRECISION" != bf16 ]]; then
  serve_args+=(--draft-precision "$DRAFT_PRECISION")
fi
if [[ "$ACCEPT_DEFAULT" != "2:1.0" ]]; then
  docker run --rm --entrypoint python3 "$IMAGE" -m sharar serve --help 2>/dev/null | grep -q -- "--accept-default" \
    || fail "image $IMAGE predates --accept-default; rebuild with docker/build.sh or unset SHARAR_ACCEPT_DEFAULT"
  serve_args+=(--accept-default "$ACCEPT_DEFAULT")
fi
if [[ "$ACCEPT_DEFAULT_SAMPLED" == 1 && "$ACCEPT_DEFAULT" != off ]]; then
  serve_help="$(docker run --rm --entrypoint python3 "$IMAGE" -m sharar serve --help 2>/dev/null)"
  grep -q -- "--accept-default-sampled" <<<"$serve_help" \
    || fail "image $IMAGE predates --accept-default-sampled; rebuild with docker/build.sh or unset SHARAR_ACCEPT_DEFAULT_SAMPLED"
  if [[ -z "$ACCEPT_DEFAULT_SAMPLED_THRESHOLD" ]]; then
    serve_args+=(--accept-default-sampled)
  elif grep -q -- "--accept-default-sampled-threshold" <<<"$serve_help"; then
    serve_args+=(--accept-default-sampled --accept-default-sampled-threshold "$ACCEPT_DEFAULT_SAMPLED_THRESHOLD")
  else  # an image older than the threshold: sampled requests stay exact rather than get the greedy threshold
    echo "  WARNING: image $IMAGE predates --accept-default-sampled-threshold; sampled requests stay exact (pull or rebuild the image)"
  fi
fi
if [[ "$MAX_CONTEXT" != 32768 ]]; then
  docker run --rm --entrypoint python3 "$IMAGE" -m sharar serve --help 2>/dev/null | grep -q -- "--context-length" \
    || fail "image $IMAGE predates --context-length; rebuild with docker/build.sh or set SHARAR_MAX_CONTEXT=32768"
  serve_args+=(--context-length "$MAX_CONTEXT")
fi
if [[ "$MAX_BATCH" != 4 || -n "$SPEC_DEPTHS" ]]; then
  docker run --rm --entrypoint python3 "$IMAGE" -m sharar serve --help 2>/dev/null | grep -q -- "--max-batch" \
    || fail "image $IMAGE predates --max-batch; rebuild with docker/build.sh or unset SHARAR_MAX_BATCH/SHARAR_SPEC_DEPTHS"
  serve_args+=(--max-batch "$MAX_BATCH")
  [[ -n "$SPEC_DEPTHS" && "$SPEC" != 0 ]] && serve_args+=(--spec-depths "$SPEC_DEPTHS")
fi
if [[ "$W3_DECODE" == 1 ]]; then
  # An older image reads the retired per-target settings and would silently drop copies (the draft's, Flash-Next's trunk).
  docker run --rm --entrypoint python3 "$IMAGE" -c "from sharar.weights.w3_checkpoint import fetch" 2>/dev/null \
    || fail "image $IMAGE predates one 3-bit switch for both targets (#72); rebuild with docker/build.sh or set SHARAR_W3_DECODE=0"
fi
if [[ -n "$W3_CHECKPOINT" ]]; then
  args+=(--volume "$W3_CHECKPOINT:/w3:ro" --env SHARAR_W3_CHECKPOINT=/w3)
fi
if [[ "$PROPOSER" == dflash2 || "$PROPOSER" == dspark ]] && [[ "$SPEC" != 0 ]]; then
  mkdir -p "$RUNS_DIR"   # promoted checkpoints; /reload_draft swaps them in without a restart
  args+=(--volume "$RUNS_DIR:/runs:ro")
fi
if [[ -n "$DRAFT_DIR" ]]; then
  [[ -f "$DRAFT_DIR/config.json" && -f "$DRAFT_DIR/model.safetensors" ]] || fail "SHARAR_DRAFT=$DRAFT_DIR needs config.json and model.safetensors"
  [[ "$PROPOSER" == dflash2 || "$PROPOSER" == dspark ]] && [[ "$SPEC" != 0 ]] || fail "SHARAR_DRAFT requires an enabled parallel draft"
  args+=(--volume "$DRAFT_DIR:/draft:ro")
  serve_args+=(--draft /draft)
fi
args+=(--env "SHARAR_TARGET=$TARGET_NAME")   # pinned serving target (sharar/target.py): qwen38 or flashnext
if [[ "$TARGET_NAME" == flashnext ]]; then
  args+=(--volume "$MODEL_CACHE:/cache/flash-next:ro" --env SHARAR_FLASHNEXT_CACHE=/cache/flash-next
         --env "SHARAR_FLASHNEXT_TRUNK=${SHARAR_FLASHNEXT_TRUNK:-fp8}")   # trunk precision at load: fp8 (default) or bf16
fi
args+=(--env "SHARAR_DEVICE_ACCEPT=$DEVICE_ACCEPT")   # device-resident acceptance graphs (engine/accept_graphs.py)
if [[ -n "${SHARAR_DRAFT_VOCAB:-}" ]]; then
  [[ "$SHARAR_DRAFT_VOCAB" =~ ^[0-9]+$ ]] || fail "SHARAR_DRAFT_VOCAB must be a row count (0 = the full lm_head)"
  args+=(--env "SHARAR_DRAFT_VOCAB=$SHARAR_DRAFT_VOCAB")   # MTP drafts over the N most likely ids (proposers/draft_vocab.py)
fi
if [[ -n "${SHARAR_FLASHNEXT_EXPERTS_SIM3:-}" && "${SHARAR_FLASHNEXT_EXPERTS_SIM3}" != 0 ]]; then
  [[ "$SHARAR_FLASHNEXT_EXPERTS_SIM3" == 16 || "$SHARAR_FLASHNEXT_EXPERTS_SIM3" == 32 ]] || fail "SHARAR_FLASHNEXT_EXPERTS_SIM3 must be 16 or 32"
  args+=(--env "SHARAR_FLASHNEXT_EXPERTS_SIM3=$SHARAR_FLASHNEXT_EXPERTS_SIM3")   # routed experts rounded to a 3-bit format (a quality screen, weights/expert_sim3.py)
fi
if [[ -n "${SHARAR_DRAFT_HEAD:-}" ]]; then
  [[ "$SHARAR_DRAFT_HEAD" == fp8 || "$SHARAR_DRAFT_HEAD" == w3 ]] || fail "SHARAR_DRAFT_HEAD must be fp8 or w3"
  args+=(--env "SHARAR_DRAFT_HEAD=$SHARAR_DRAFT_HEAD")   # MTP draft steps read a 3-bit copy of the lm_head (w3) or the target's own (fp8)
fi
if [[ -n "${SHARAR_QSA_SPARSE_GRAPHS:-}" ]]; then
  [[ "$SHARAR_QSA_SPARSE_GRAPHS" == 0 || "$SHARAR_QSA_SPARSE_GRAPHS" == 1 ]] || fail "SHARAR_QSA_SPARSE_GRAPHS must be 0 or 1"
  args+=(--env "SHARAR_QSA_SPARSE_GRAPHS=$SHARAR_QSA_SPARSE_GRAPHS")   # Flash-Next selecting graphs past the indexer budget (0: eager)
fi
if [[ -n "${SHARAR_COLD_TIER_GIB:-}" ]]; then
  [[ "$SHARAR_COLD_TIER_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "SHARAR_COLD_TIER_GIB must be a non-negative number of GiB"
  args+=(--env "SHARAR_COLD_TIER_GIB=$SHARAR_COLD_TIER_GIB")   # managed-memory budget for parked sessions (0 disables the tier)
fi
SESSION_DIR="${SHARAR_SESSION_DIR:-$RESULTS_DIR/sessions}"   # NVMe store for parked sessions (off disables)
if [[ "$SESSION_DIR" != off ]]; then
  [[ "${SHARAR_SESSION_GIB:-200}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "SHARAR_SESSION_GIB must be a number of GiB"
  mkdir -p "$SESSION_DIR"
  args+=(--volume "$SESSION_DIR:/sessions" --env SHARAR_SESSION_DIR=/sessions --env "SHARAR_SESSION_GIB=${SHARAR_SESSION_GIB:-200}"
        --env "SHARAR_SESSION_MIN_TOKENS=${SHARAR_SESSION_MIN_TOKENS:-4096}")
fi
if [[ -n "$TRACES_DIR" ]]; then
  [[ "$TRACES_MAX_MIB" =~ ^[1-9][0-9]*$ ]] || fail "SHARAR_TRACES_MAX_MIB must be a positive integer"
  mkdir -p "$TRACES_DIR"
  args+=(--volume "$TRACES_DIR:/traces")
  serve_args+=(--traces /traces --traces-max-mib "$TRACES_MAX_MIB")
  # The MTP draft's confidence per drafted position in the round records (#32, #43)
  [[ "${SHARAR_TRACE_CONFIDENCE:-0}" == 1 ]] && args+=(--env SHARAR_TRACE_CONFIDENCE=1)
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
