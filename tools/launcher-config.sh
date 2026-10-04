# The launcher configuration for one target, shared by run-sharar.sh, run-sharar-train.sh, sharar-ctl, the GPU
# suite, the CI and the gates, so they all serve the same thing.
#
#   source tools/launcher-config.sh; launcher_defaults || exit   # after sourcing the user's config file
#   bash tools/launcher-config.sh print [TARGET]                 # the resolved settings as KEY=VALUE lines
#
# Order, first wins: the environment and the user's config file (config/gb10.env, or SHARAR_CONFIG_FILE), the
# target's settings (config/targets/<target>.env), then the defaults below. A config file is optional: without
# one, models live in the Hugging Face cache and results under ~/.local/share/sharar.
LAUNCHER_CONFIG_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# target_name NAME: the canonical target for a user-facing name (27b, qwen38, flashnext, flash-next).
target_name() {
  case "${1:-qwen38}" in
    27b|27B|qwen38|qwen3.8-27b) echo qwen38 ;;
    flashnext|flash-next|Flash-Next) echo flashnext ;;
    *) echo "unknown target '$1' (choose 27b or flashnext)" >&2; return 1 ;;
  esac
}

# launcher_defaults: normalize SHARAR_TARGET and fill every SHARAR_* setting still unset or empty.
launcher_defaults() {
  local k v file
  SHARAR_TARGET="$(target_name "${SHARAR_TARGET:-qwen38}")" || return 1
  file="$LAUNCHER_CONFIG_REPO/config/targets/$SHARAR_TARGET.env"
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^SHARAR_[A-Z0-9_]+$ && -z "${!k:-}" ]] && printf -v "$k" '%s' "$v"
  done < "$file"
  : "${SHARAR_HF_CACHE:=${HF_HOME:-$HOME/.cache/huggingface}}"
  : "${SHARAR_FLASHNEXT_CACHE:=$SHARAR_HF_CACHE}"
  : "${SHARAR_RESULTS:=${XDG_DATA_HOME:-$HOME/.local/share}/sharar}"
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" && "${1:-}" == print ]]; then
  set -Eeuo pipefail
  conf="${SHARAR_CONFIG_FILE:-$LAUNCHER_CONFIG_REPO/config/gb10.env}"
  # shellcheck disable=SC1090
  [[ -r "$conf" ]] && source "$conf"
  [[ -n "${2:-}" ]] && SHARAR_TARGET="$2"
  launcher_defaults
  for k in $(compgen -v SHARAR_); do printf '%s=%s\n' "$k" "${!k}"; done
fi
