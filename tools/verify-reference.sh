#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER="${SHARAR_REFERENCE_CONTAINER:-sparklang-gb10-qwen38-nvfp4}"
BASE_URL="${SHARAR_REFERENCE_URL:-http://127.0.0.1:30000}"
MODEL="RadixArk/Qwen3.8-27B-NVFP4"
REVISION="319f741cce68d7914884900c138a1fbb70a42f30"
MIN_AVAILABLE_KIB="${SHARAR_MIN_AVAILABLE_KIB:-16777216}"

fail() {
  echo "GB10 verification failed: $*" >&2
  exit 1
}

[[ "$(uname -m)" == "aarch64" ]] || fail "host architecture is not aarch64"
[[ "$(docker inspect --format '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" == "running" ]] || \
  fail "container is not running: $CONTAINER"
[[ "$(docker inspect --format '{{index .Config.Labels "io.sparklang.target"}}' "$CONTAINER")" == "gb10-sm121" ]] || \
  fail "container target label is not gb10-sm121"
[[ "$(docker inspect --format '{{index .Config.Labels "io.sparklang.profile"}}' "$CONTAINER")" == "production" ]] || \
  fail "container is not using the production profile"
[[ "$(docker inspect --format '{{index .Config.Labels "io.sparklang.max-mamba-cache-size"}}' "$CONTAINER")" == "20" ]] || \
  fail "container does not use the 20-slot Mamba pool"

curl --fail --silent "$BASE_URL/health" >/dev/null || fail "health endpoint is not ready"
curl --fail --silent "$BASE_URL/get_server_info" | python3 -c '
import json, sys
d = json.load(sys.stdin)
expected = {
    "model_path": "RadixArk/Qwen3.8-27B-NVFP4",
    "revision": "319f741cce68d7914884900c138a1fbb70a42f30",
    "context_length": 32768,
    "max_total_tokens": 32768,
    "max_running_requests": 4,
    "chunked_prefill_size": 2048,
    "max_mamba_cache_size": 20,
    "mamba_ssm_dtype": "float32",
    "mem_fraction_static": 0.7,
}
wrong = {key: (d.get(key), value) for key, value in expected.items() if d.get(key) != value}
if wrong:
    raise SystemExit(f"server profile mismatch: {wrong}")
' || fail "server metadata does not match the pinned profile"

curl --fail --silent "$BASE_URL/v1/models" | python3 -c '
import json, sys
d = json.load(sys.stdin)
models = d.get("data", [])
if len(models) != 1 or models[0].get("id") != "RadixArk/Qwen3.8-27B-NVFP4":
    raise SystemExit(f"unexpected model list: {models}")
' || fail "model metadata is not the pinned target"
docker inspect --format '{{json .Config.Cmd}}' "$CONTAINER" | python3 -c '
import json, sys
command = json.load(sys.stdin)
expected = ["python3", "-m", "sparklang", "serve"]
if command[:4] != expected:
    raise SystemExit(f"container does not enter through SparkLang: {command}")
' || fail "container command does not use the SparkLang interface"
docker exec "$CONTAINER" python3 -m sparklang serve --print-command | python3 -c '
import shlex, sys
command = shlex.split(sys.stdin.read())
required = {
    "--language-model-only",
    "--model-path",
    "--revision",
    "--attention-backend",
    "--mamba-radix-cache-strategy",
}
missing = sorted(required - set(command))
if command[:2] != ["sglang", "serve"] or missing:
    raise SystemExit(f"invalid inherited runtime command; missing={missing}: {command}")
' || fail "SparkLang did not generate the pinned text runtime command"

resolved_line="$(docker logs "$CONTAINER" 2>&1 | awk '/max_total_num_tokens=/ {line=$0} END {print line}')"
[[ "$resolved_line" == *"max_total_num_tokens=32768"* ]] || fail "resolved token capacity is not 32768"
[[ "$resolved_line" == *"max_running_requests=4"* ]] || fail "resolved scheduler capacity is not 4"
[[ "$resolved_line" == *"context_len=32768"* ]] || fail "resolved context length is not 32768"
mamba_line="$(docker logs "$CONTAINER" 2>&1 | awk '/Mamba Cache is allocated/ {line=$0} END {print line}')"
[[ "$mamba_line" == *"max_mamba_cache_size: 20"* ]] || \
  fail "runtime did not allocate the 20-slot Mamba pool"

systemctl --user is-active --quiet sparklang-gb10-memory-guard.service || \
  fail "unified-memory guard is not active"
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= MIN_AVAILABLE_KIB )) || \
  fail "MemAvailable is below the runtime reserve"

echo "GB10 engine READY"
echo "  Container: $CONTAINER"
echo "  Model: $MODEL@$REVISION"
echo "  Resolved: 32768 tokens, concurrency 4, Mamba slots 20"
echo "  MemAvailable: $((available_kib / 1048576)) GiB"
