#!/usr/bin/env bash
# Stop a GPU workload before unified memory runs out: when host MemAvailable stays under the floor
# (SHARAR_MIN_AVAILABLE_KIB, default 16 GiB) for three samples two seconds apart, stop the workload and exit 1;
# exit 0 when the workload ends by itself. On the GB10 the GPU allocates from host memory, so running out pages the
# weights and stalls the whole machine.
#   tools/memory-guard.sh --unit NAME   a user systemd unit (run-sharar.sh's server, run-sharar-train.sh's trainer)
#   tools/memory-guard.sh --pid PID     a process and its process group (the CI's GPU suite)
set -Eeuo pipefail

kind="${1:?usage: memory-guard.sh --unit NAME | --pid PID}"
name="${2:?usage: memory-guard.sh --unit NAME | --pid PID}"
threshold_kib="${SHARAR_MIN_AVAILABLE_KIB:-16777216}"
state_home="${XDG_STATE_HOME:-${HOME}/.local/state}"
results_dir="${SHARAR_RESULTS:-$state_home/sharar/gb10}"
log_file="$results_dir/memory-guard.log"
low_samples=0

mkdir -p "$results_dir"

alive() {
  case "$kind" in
    --unit) systemctl --user is-active --quiet "$name" ;;
    --pid) kill -0 "$name" 2>/dev/null ;;
    *) echo "unknown workload kind $kind (--unit or --pid)" >&2; exit 2 ;;
  esac
}

stop() {
  case "$kind" in
    --unit) timeout 20 systemctl --user stop "$name" || systemctl --user kill --signal=SIGKILL "$name" ;;
    --pid) kill -TERM -- "-$(ps -o pgid= -p "$name" | tr -d ' ')" 2>/dev/null || kill -TERM "$name"
           sleep 15; kill -KILL "$name" 2>/dev/null || true ;;
  esac
}

while true; do
  if ! alive; then
    printf '%s exiting=%s reason=workload_ended\n' "$(date --iso-8601=seconds)" "$name" >>"$log_file"
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
    printf '%s stopping=%s reason=low_available_memory\n' "$(date --iso-8601=seconds)" "$name" >>"$log_file"
    stop >>"$log_file" 2>&1
    exit 1
  fi

  sleep 2
done
