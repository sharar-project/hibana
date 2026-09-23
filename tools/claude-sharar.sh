#!/usr/bin/env bash
# Run Claude Code against a Sharar server. Arguments go to `claude` unchanged.
#   tools/claude-sharar.sh                      # interactive session in the current directory
#   SHARAR_URL=http://127.0.0.1:30002 tools/claude-sharar.sh -p "fix the failing test"
# Claude Code only accepts model IDs containing "claude" or "anthropic"; Sharar
# serves any name. A separate config directory keeps your Anthropic login,
# settings and history untouched (first launch asks the onboarding questions).
set -euo pipefail
url="${SHARAR_URL:-http://127.0.0.1:30001}"
context="$(curl -fsS "$url/get_server_info" | python3 -c 'import json,sys; print(json.load(sys.stdin)["context_length"])')" \
  || { echo "no Sharar server at $url" >&2; exit 1; }
model="${SHARAR_CLAUDE_MODEL:-claude-sharar-qwen3.8}"
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDECODE
export ANTHROPIC_BASE_URL="$url" ANTHROPIC_AUTH_TOKEN=sharar
export ANTHROPIC_MODEL="$model" ANTHROPIC_DEFAULT_SONNET_MODEL="$model" ANTHROPIC_DEFAULT_OPUS_MODEL="$model" ANTHROPIC_DEFAULT_HAIKU_MODEL="$model"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$context" CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CONFIG_DIR="${SHARAR_CLAUDE_CONFIG:-$HOME/.claude-sharar}"
exec claude "$@"
