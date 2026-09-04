#!/usr/bin/env bash
# Verify a running Sharar server: health, model identity, profile, a decode.
set -Eeuo pipefail
BASE_URL="${SHARAR_URL:-http://127.0.0.1:30001}"
MODEL="RadixArk/Qwen3.8-27B-NVFP4"
fail() { echo "sharar verification failed: $*" >&2; exit 1; }
curl --fail --silent "$BASE_URL/health" >/dev/null || fail "health endpoint is not ready"
curl --fail --silent "$BASE_URL/v1/models" | python3 -c '
import json, sys
d = json.load(sys.stdin)["data"]
assert len(d) == 1 and d[0]["id"] == "RadixArk/Qwen3.8-27B-NVFP4", d
' || fail "model metadata is not the pinned target"
curl --fail --silent "$BASE_URL/get_server_info" | python3 -c '
import json, sys
d = json.load(sys.stdin)
expected = {"engine": "sharar", "model_path": "RadixArk/Qwen3.8-27B-NVFP4",
            "revision": "319f741cce68d7914884900c138a1fbb70a42f30", "context_length": 32768,
            "max_running_requests": 4, "kv_cache_dtype": "float8_e4m3fn", "mamba_ssm_dtype": "float32"}
wrong = {k: (d.get(k), v) for k, v in expected.items() if d.get(k) != v}
assert not wrong, wrong
' || fail "server profile mismatch"
out="$(curl --fail --silent "$BASE_URL/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with exactly SPARKLANG_OK and nothing else."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())')"
[[ "$out" == "SPARKLANG_OK" ]] || fail "chat canary returned '$out'"
echo "Sharar READY at $BASE_URL"
