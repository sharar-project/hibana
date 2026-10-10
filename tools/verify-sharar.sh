#!/usr/bin/env bash
# Verify a running Sharar server: health, pinned model identity, profile, a decode.
# SHARAR_URL (default http://127.0.0.1:30001); SHARAR_MAX_CONTEXT as served (default 262144).
set -Eeuo pipefail
BASE_URL="${SHARAR_URL:-http://127.0.0.1:30001}"
CONTEXT="${SHARAR_MAX_CONTEXT:-262144}"
fail() { echo "sharar verification failed: $*" >&2; exit 1; }
curl --fail --silent "$BASE_URL/health" >/dev/null || fail "health endpoint is not ready"
MODEL="$(curl --fail --silent "$BASE_URL/get_server_info" | CONTEXT="$CONTEXT" python3 -c '
import json, os, sys
d = json.load(sys.stdin)
pinned = {"qwen38": ("RadixArk/Qwen3.8-27B-NVFP4", "319f741cce68d7914884900c138a1fbb70a42f30"),
          "flashnext": ("RadixArk/Qwen3.8-Flash-Next-NVFP4", "7b719225242aacd3dbd3f9407468c2ee9a9d2594")}
model, revision = pinned.get(d.get("target"), (None, None))
expected = {"engine": "sharar", "model_path": model, "revision": revision,
            "context_length": int(os.environ["CONTEXT"]), "kv_cache_dtype": "float8_e4m3fn", "mamba_ssm_dtype": "float32"}
wrong = {k: (d.get(k), v) for k, v in expected.items() if d.get(k) != v}
assert not wrong, wrong
print(model)
')" || fail "server profile mismatch"
curl --fail --silent "$BASE_URL/v1/models" | MODEL="$MODEL" python3 -c '
import json, os, sys
d = json.load(sys.stdin)["data"]
assert len(d) == 1 and d[0]["id"] == os.environ["MODEL"], d
' || fail "model metadata is not the pinned target"
out="$(curl --fail --silent "$BASE_URL/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with exactly SHARAR_OK and nothing else."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())')"
[[ "$out" == "SHARAR_OK" ]] || fail "chat canary returned '$out'"
echo "Sharar READY at $BASE_URL ($MODEL)"
