#!/usr/bin/env bash
set -Eeuo pipefail

container="${1:-sparklang-gb10-qwen38-nvfp4}"
threshold_kib="${SHARAR_MIN_AVAILABLE_KIB:-16777216}"
state_home="${XDG_STATE_HOME:-${HOME}/.local/state}"
results_dir="${SHARAR_RESULTS:-$state_home/sharar/gb10}"
log_file="$results_dir/memory-guard.log"
low_samples=0
inspect_failures=0

mkdir -p "$results_dir"

while true; do
  if ! container_state="$(
    docker inspect --format '{{.State.Status}}' "$container" 2>>"$log_file"
  )"; then
    ((inspect_failures += 1))
    if (( inspect_failures >= 3 )); then
      printf '%s exiting=%s reason=container_inspect_failed attempts=%s\n' \
        "$(date --iso-8601=seconds)" "$container" "$inspect_failures" >>"$log_file"
      exit 0
    fi
    sleep 2
    continue
  fi
  inspect_failures=0

  if [[ "$container_state" != "running" && "$container_state" != "restarting" ]]; then
    printf '%s exiting=%s reason=container_state state=%s\n' \
      "$(date --iso-8601=seconds)" "$container" "$container_state" >>"$log_file"
    exit 0
  fi

  available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  printf '%s available_kib=%s threshold_kib=%s\n' \
    "$(date --iso-8601=seconds)" "$available_kib" "$threshold_kib" >>"$log_file"

  if (( available_kib < threshold_kib )); then
    ((low_samples += 1))
  else
    low_samples=0
  fi

  if (( low_samples >= 3 )); then
    printf '%s stopping=%s reason=low_available_memory\n' \
      "$(date --iso-8601=seconds)" "$container" >>"$log_file"
    docker stop --time 15 "$container" >>"$log_file" 2>&1
    exit 1
  fi

  sleep 2
done
