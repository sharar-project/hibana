#!/usr/bin/env bash
# Claude Code on the serving Sharar target; arguments go to `claude` unchanged. Kept for old instructions:
# this is `sharar-ctl launch claude` (SHARAR_URL still picks another server).
#   tools/claude-sharar.sh                      # interactive session in the current directory
#   tools/claude-sharar.sh -p "fix the failing test"
exec "$(dirname "$(readlink -f "$0")")/../sharar-ctl" launch claude "$@"
