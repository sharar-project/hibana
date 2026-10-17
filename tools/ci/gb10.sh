#!/usr/bin/env bash
# Nightly and weekly checks on the GB10 (.github/workflows/{nightly,weekly}-gb10.yml, self-hosted runner `gb10`).
#
#   Nightly:  tools/ci/gb10.sh [--skip-if-tested] venv gpu serve
#   Weekly:   tools/ci/gb10.sh venv evaluate
#   Cleanup:  tools/ci/gb10.sh --cleanup
#
# Stages (each needs the one before it to pass, so old code is never tested):
#   venv      the Python environment for this checkout: `uv sync --frozen --extra eval` into UV_PROJECT_ENVIRONMENT (default
#             ~/sharar-ci/venv, kept between runs; uv.lock decides what it holds)
#   gpu       tests/unit and tests/gpu from it (`uv run pytest`, SHARAR_RUN_GPU=1, under the memory guard)
#   serve     serve each target from it (`sharar serve <target>` in the foreground on port 30011, 32K context) and
#             check it (`sharar verify`)
#   evaluate  each target, fast profile (tools/evaluate.py): AIPerf speed (code, design, reasoning) and, so that a
#             speed gain from wrongly accepted drafts cannot pass, GSM8K (first 300) and HumanEval (164); paired with
#             that target's baseline. Fails on more than 5% lower TPS or a quality loss of at least 1 point (p < 0.05,
#             more items worse than better). The first complete run of a target becomes its baseline:
#             $SHARAR_RESULTS/ci/baseline-<target> (re-point the link to adopt a newer run).
#
# The machine is shared, so the checks never hold anyone up. They are skipped when, at the start, the shared GPU
# lock ($SHARAR_RESULTS/.gpu.lock) is held or awaited, another container with the GPU or a GPU process runs, or
# under 40 GiB of memory is available. While they run, a watchdog checks the same every 10 s (ignoring the
# checks' own processes); on any sign of someone else's work it stops the checks' processes and ends the run as
# "gave way" (exit 0). A skipped or interrupted nightly run is not recorded as tested.
#
# Settings come from ~/sharar-ci/gb10.env (SHARAR_CONFIG_FILE): SHARAR_HF_CACHE, SHARAR_RESULTS,
# optionally SHARAR_FLASHNEXT_CACHE. sharar-eval is the `eval` extra of the environment the venv stage syncs.
set -Euo pipefail
SELF="$(realpath "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")/../.."

ALL="venv gpu serve evaluate"
SERVE_PORT=30011
OUT="$PWD/ci-out"
MIN_FREE_GIB=40
MAX_SPEED_DROP=5      # percent
MIN_QUALITY_DROP=1.0  # points
MAX_P=0.05

export SHARAR_CONFIG_FILE="${SHARAR_CONFIG_FILE:-$HOME/sharar-ci/gb10.env}"
[[ -r "$SHARAR_CONFIG_FILE" ]] || { echo "missing config $SHARAR_CONFIG_FILE" >&2; exit 2; }
set -a; source "$SHARAR_CONFIG_FILE"; set +a
: "${SHARAR_HF_CACHE:?set SHARAR_HF_CACHE in $SHARAR_CONFIG_FILE}"
: "${SHARAR_RESULTS:?set SHARAR_RESULTS in $SHARAR_CONFIG_FILE}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/sharar-ci/venv}"
export PATH="$HOME/.local/bin:$PATH"   # uv
LOCK="$SHARAR_RESULTS/.gpu.lock"
SESSIONS="$OUT/sessions"   # the session ids of this run's processes (lock holders among them are ours)

# ---- whose work is on the machine ----
our_session() { local sid; sid="$(ps -o sid= -p "$1" 2>/dev/null | tr -d ' ')"; [[ -n "$sid" ]] && grep -qxF "$sid" "$SESSIONS" 2>/dev/null; }
lock_pids() {  # pids /proc/locks names for the GPU lock (holders, or waiters with `waiting`)
  [[ -e "$LOCK" ]] || return 0
  awk -v ino=":$(stat -c %i "$LOCK")" -v want="${1:-}" '$0 ~ /FLOCK/ && (want == "waiting") == ($2 == "->") {
    for (i = 1; i <= NF; i++) if ($i ~ ino"$") print $(i - 1) }' /proc/locks | sort -u
}
# Takes the GPU lock on fd 8 without waiting. /proc/locks names the short-lived flock(1) process as the holder,
# so its pid is recorded as this run's.
take_lock() {
  exec 8>>"$LOCK"
  flock -n 8 || { echo "the GPU lock was taken"; return 1; }
  lock_pids >>"$OUT/lock-pids"
}

# Prints why the machine is someone else's right now (nothing when it is free). `start` adds the memory floor.
foreign_work() {
  local why=() name pid
  # Containers with the GPU device (these checks start none).
  while read -r name; do
    [[ -n "$name" ]] || continue
    docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}{{len .HostConfig.DeviceRequests}}' "$name" 2>/dev/null \
      | grep -qE '/dev/nvidia|[1-9]$' && why+=("container $name has the GPU")
  done < <(command -v docker >/dev/null && docker ps --format '{{.Names}}')
  # Detached Sharar services (someone serving or training with --detach).
  while read -r name; do
    [[ -n "$name" ]] && why+=("service $name runs")
  done < <(systemctl --user list-units --state=active --plain --no-legend 'sharar-*.service' 2>/dev/null | awk '{print $1}')
  # GPU processes outside these checks' sessions.
  while IFS=, read -r pid _; do
    pid="${pid// /}"; [[ "$pid" =~ ^[0-9]+$ ]] || continue
    our_session "$pid" && continue
    why+=("GPU process $pid ($(ps -o comm= -p "$pid" 2>/dev/null || echo '?'))")
  done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
  # The shared GPU lock, held or awaited by anyone else.
  while read -r pid; do
    our_session "$pid" || grep -qxF "$pid" "$OUT/lock-pids" 2>/dev/null || why+=("process $pid holds the GPU lock")
  done < <(lock_pids)
  while read -r pid; do
    our_session "$pid" || why+=("process $pid waits for the GPU lock")
  done < <(lock_pids waiting)
  if [[ "${1:-}" == start ]]; then
    local avail; avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
    (( avail >= MIN_FREE_GIB )) || why+=("only $avail GiB of memory available")
  fi
  (( ${#why[@]} )) && { local IFS=';'; echo "${why[*]}"; }
  return 0
}

stop_ours() {  # every process these checks started (each stage runs in a session of its own)
  local sid
  while read -r sid; do
    [[ -n "$sid" && "$sid" != "$$" ]] && pkill -TERM -s "$sid" 2>/dev/null
  done < <(tail -n +2 "$SESSIONS" 2>/dev/null)
  return 0
}

# Model files left in the page cache by earlier loads keep MemFree low, and a Flash-Next load then stalls (CUDA
# on the GB10 sees MemFree): drop them before each stage that loads a model.
evict_models() {
  python3 -c '
import os, sys
for root in filter(None, sys.argv[1:]):
    for d, _, files in os.walk(root):
        for f in files:
            try:
                fd = os.open(os.path.join(d, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass
' "$SHARAR_HF_CACHE" "${SHARAR_FLASHNEXT_CACHE:-}"
}

# ---- stages (each runs in its own session: `gb10.sh --stage NAME`) ----
stage_venv() {
  uv sync --frozen --quiet --extra eval
}

# tests/unit too: some unit tests need the model or the GPU and only run here. Test temp files go in ci-out/ on
# the runner's disk: tests/unit/test_ple_rows.py relies on RWF_NOWAIT, which a tmpfs /tmp may lack.
stage_gpu() {
  local pid rc=0
  take_lock || return 1
  evict_models
  SHARAR_RUN_GPU=1 uv run --frozen pytest -q -p no:cacheprovider tests/unit tests/gpu --basetemp=ci-out/pytest-tmp &
  pid=$!
  uv run --frozen python -m sharar.launch.guard --pid "$pid" 8>&- &
  wait "$pid" || rc=$?
  return "$rc"
}

# Each target in the foreground on SERVE_PORT, 32K context, the pinned draft, nothing recorded; `sharar verify`, then
# SIGTERM (as Ctrl-C).
stage_serve() {
  local target pid waited rc=0 log
  take_lock || return 1
  for target in 27b flashnext; do
    evict_models
    log="$OUT/serve-$target.log"
    SHARAR_SSD=0 SHARAR_DRAFT=pinned SHARAR_TRACES=off uv run --frozen sharar serve "$target" --port "$SERVE_PORT" --context-length 32768 >"$log" 2>&1 8>&- &
    pid=$!
    waited=0
    until curl -fs "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; do
      if ! kill -0 "$pid" 2>/dev/null || (( waited >= 900 )); then
        echo "$target: not healthy"; tail -n 100 "$log"; rc=1; break
      fi
      sleep 5; waited=$(( waited + 5 ))
    done
    if (( waited < 900 )) && curl -fs "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; then
      if uv run --frozen sharar verify "$target" --url "http://127.0.0.1:$SERVE_PORT"; then
        echo "$target: verified (healthy after ${waited}s)" >>"$OUT/notes-serve"
      else
        echo "$target: verification failed"; tail -n 100 "$log"; rc=1
      fi
    fi
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
  done
  return "$rc"
}

# tools/evaluate.py takes the GPU lock itself for its serving window.
stage_evaluate() {
  local target name dir base rc=0 verdict paired
  for target in qwen38 flashnext; do
    name="ci-weekly-$target-$(date -u +%Y%m%dT%H%M)"
    base="$SHARAR_RESULTS/ci/baseline-$target"
    paired=(); [[ -e "$base" ]] && paired=(--baseline "$(realpath "$base")")
    evict_models
    python3 tools/evaluate.py --target "$target" --profile fast --tasks gsm8k,humaneval --limit 300 --speed --workload all \
      --name "$name" "${paired[@]}" || { echo "$target: evaluate did not complete"; rc=1; continue; }
    dir="$(ls -d "$SHARAR_RESULTS"/reviews/*-evaluate-"$name" | tail -1)"
    echo "$dir" >>"$OUT/evaluate-dirs"
    verdict="$(python3 - "$dir" "$MIN_QUALITY_DROP" "$MAX_P" "$MAX_SPEED_DROP" <<'EOF'
import json, sys
from pathlib import Path
run, min_drop, max_p, max_slow = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
bad, notes = [], []
compare = run / "compare.json"
if compare.is_file():
    for task, r in json.loads(compare.read_text())["tasks"].items():
        if r.get("delta") is None:
            continue
        if -r["delta"] >= min_drop and r["losses"] > r["wins"] and r["p"] < max_p:
            bad.append(f"{task} {r['delta']:+.1f} points (p {r['p']:.3f})")
    notes.append("quality paired with the baseline")
else:
    notes.append("no quality baseline yet")
speed = json.loads((run / "speed" / "summary.json").read_text())
comparison = speed.get("comparison") or {}
for case, c in comparison.items():
    for key in ("decode_tps_change_pct", "output_tps_change_pct"):
        if c.get(key) is not None and c[key] < -max_slow:
            bad.append(f"{case} {key.split('_')[0]} TPS {c[key]:+.1f}%")
notes.append("speed paired with the baseline" if comparison else "no speed baseline yet")
print("; ".join(bad) if bad else ", ".join(notes))
sys.exit(1 if bad else 0)
EOF
)" || { echo "$target: regression: $verdict" >>"$OUT/notes-evaluate"; rc=1; continue; }
    echo "$target: $verdict ($dir)" >>"$OUT/notes-evaluate"
    if [[ ! -e "$base" ]]; then
      mkdir -p "$(dirname "$base")" && ln -s "$dir" "$base" && echo "$target: this run is now the baseline" >>"$OUT/notes-evaluate"
    fi
  done
  return "$rc"
}

if [[ "${1:-}" == --stage ]]; then   # one stage in its own session (see run_stage)
  ps -o sid= -p $$ | tr -d ' ' >>"$SESSIONS"
  "stage_$2"
  exit $?
fi

# ---- the run ----
if [[ "${1:-}" == --cleanup ]]; then stop_ours; exit 0; fi
skip_tested=0
if [[ "${1:-}" == --skip-if-tested ]]; then skip_tested=1; shift; fi
stages="${*:-$ALL}"
for s in $stages; do
  [[ " $ALL " == *" $s "* ]] || { echo "unknown stage '$s' (choose from: $ALL)" >&2; exit 2; }
done
wants() { [[ " $stages " == *" $1 "* ]]; }

commit="$(git rev-parse HEAD)"
TESTED="$SHARAR_RESULTS/ci/nightly-tested-commit"
if (( skip_tested )) && [[ "$(cat "$TESTED" 2>/dev/null || true)" == "$commit" ]]; then
  echo "main is unchanged since the last nightly run (${commit:0:7}); skipped" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
  exit 0
fi

export SHARAR_PROFILE=fast   # what the lab serves; fixed here so a new default cannot shift the baselines
rm -rf "$OUT"; mkdir -p "$OUT" "$SHARAR_RESULTS"
ps -o sid= -p $$ | tr -d ' ' >"$SESSIONS"
watchdog_pid=""
finish() { [[ -n "$watchdog_pid" ]] && kill "$watchdog_pid" 2>/dev/null; stop_ours; }
trap finish EXIT
trap 'exit 130' INT TERM

busy="$(foreign_work start)"

# Gives way: on someone else's work, stop this run's containers and stage, and leave the reason for the summary.
watchdog() {
  local why
  while sleep 10; do
    why="$(foreign_work)"
    [[ -n "$why" ]] || continue
    echo "$why" >"$OUT/gave-way"
    [[ -f "$OUT/stage.pid" ]] && pkill -TERM -s "$(cat "$OUT/stage.pid")" 2>/dev/null
    stop_ours
    return 0
  done
}

rows=() failed=0 completed=" "
run_stage() {
  local start=$SECONDS rc=0 pid took note
  setsid bash "$SELF" --stage "$1" &
  pid=$!
  echo "$pid" >"$OUT/stage.pid"
  wait "$pid" || rc=$?
  rm -f "$OUT/stage.pid"
  took="$(( (SECONDS - start) / 60 ))m $(( (SECONDS - start) % 60 ))s"
  note="$(cat "$OUT/notes-$1" 2>/dev/null | paste -sd ';' | sed 's/;/; /g')"
  if [[ -f "$OUT/gave-way" ]]; then rows+=("| $1 | gave way | $took; $(cat "$OUT/gave-way") |"); return 1; fi
  completed+="$1 "
  if (( rc == 0 )); then rows+=("| $1 | passed | $took${note:+; $note} |"); return 0; fi
  rows+=("| $1 | **failed** | $took${note:+; $note} |"); failed=1; return 1
}

depends() { case $1 in gpu|serve|evaluate) echo venv;; esac; }
[[ -n "$busy" ]] || { watchdog & watchdog_pid=$!; }
missing=" "
for s in $ALL; do
  wants "$s" || continue
  dep="$(depends "$s")"
  if [[ -n "$busy" ]]; then rows+=("| $s | skipped | machine in use |")
  elif [[ -f "$OUT/gave-way" ]]; then rows+=("| $s | skipped | gave way to other work |")
  elif d="$(for x in $dep; do wants "$x" && [[ "$missing" == *" $x "* ]] && echo "$x"; done | head -1)"; [[ -n "$d" ]]; then
    rows+=("| $s | skipped | $d did not pass |")
  elif run_stage "$s"; then continue
  fi
  missing+="$s "
done
if [[ -n "$busy" ]]; then echo "::warning::GB10 in use ($busy); checks skipped"; fi
if [[ -f "$OUT/gave-way" ]]; then echo "::warning::GB10 checks gave way to other work ($(cat "$OUT/gave-way"))"; fi
# A nightly commit counts as tested only when every stage ran to the end.
if wants gpu && [[ "$completed" == *" gpu "* && ( "$completed" == *" serve "* || ! " $stages " == *" serve "* ) ]]; then
  mkdir -p "$(dirname "$TESTED")" && echo "$commit" >"$TESTED"
fi

{
  echo "## GB10 checks, $(date -u '+%Y-%m-%d %H:%M UTC'), ${commit:0:7}, $(hostname), profile $SHARAR_PROFILE"
  [[ -z "$busy" ]] || printf '\nSkipped: the machine is in use (%s).\n' "$busy"
  echo
  echo "| Stage | Result | Time and notes |"
  echo "|---|---|---|"
  printf '%s\n' "${rows[@]}"
  while read -r f; do
    [[ -f "$f/SUMMARY.md" ]] && { echo; sed 's/^# /### /; s/^## /#### /' "$f/SUMMARY.md"; }
  done < <(cat "$OUT/evaluate-dirs" 2>/dev/null)
} >"$OUT/summary.md"
cat "$OUT/summary.md"
[[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && cat "$OUT/summary.md" >>"$GITHUB_STEP_SUMMARY"
exit "$failed"
