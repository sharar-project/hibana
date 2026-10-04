#!/usr/bin/env bash
# Nightly and weekly checks on the GB10 (.github/workflows/{nightly,weekly}-gb10.yml, self-hosted runner `gb10`).
#
#   Nightly:  tools/ci/gb10.sh [--skip-if-tested] image gpu serve [publish]
#   Weekly:   tools/ci/gb10.sh image evaluate
#   Cleanup:  tools/ci/gb10.sh --cleanup
#
# Stages (each needs the one before it to pass, so old code is never tested):
#   image     build this checkout into sharar:ci (the previous sharar:ci is removed)
#   gpu       tests/unit and tests/gpu in that image (tools/gpu-tests.sh)
#   serve     serve each target from that image (run-sharar.sh, 32K context) and check it (tools/verify-sharar.sh)
#   publish   push that image to ghcr.io/abdelfattah-lab/sharar as main-<commit> and latest (the launchers' default,
#             tools/image.sh); only after gpu and serve passed in the same run, and the workflow asks for it only on
#             main. Credentials: GHCR_TOKEN (and GHCR_USER) from the workflow, used through a private docker config so
#             the runner's own `docker login` is untouched
#   evaluate  each target, fast profile (tools/evaluate.py): AIPerf speed (code, design, reasoning) and, so that a
#             speed gain from wrongly accepted drafts cannot pass, GSM8K (first 300) and HumanEval (164); paired with
#             that target's baseline. Fails on more than 5% lower TPS or a quality loss of at least 1 point (p < 0.05,
#             more items worse than better). The first complete run of a target becomes its baseline:
#             $SHARAR_RESULTS/ci/baseline-<target> (re-point the link to adopt a newer run).
#
# The machine is shared, so the checks never hold anyone up. They are skipped when, at the start, the shared GPU
# lock ($SHARAR_RESULTS/.gpu.lock) is held or awaited, another container with the GPU or a GPU process runs, or
# under 40 GiB of memory is available. While they run, a watchdog checks the same every 10 s (ignoring the
# checks' own containers and processes); on any sign of someone else's work it stops the checks' containers and
# ends the run as "gave way" (exit 0). A skipped or interrupted nightly run is not recorded as tested.
#
# Settings come from ~/sharar-ci/gb10.env (SHARAR_CONFIG_FILE): SHARAR_HF_CACHE, SHARAR_RESULTS, SHARAR_AIPERF,
# optionally SHARAR_FLASHNEXT_CACHE and SHARAR_EVALS (the sharar-eval checkout, default ~/sharar-eval).
set -Euo pipefail
SELF="$(realpath "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")/../.."

ALL="image gpu serve publish evaluate"
IMAGE=sharar:ci
REGISTRY=ghcr.io/abdelfattah-lab/sharar
SERVE_CONTAINER=sharar-ci
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
EVALS="${SHARAR_EVALS:-$HOME/sharar-eval}"
LOCK="$SHARAR_RESULTS/.gpu.lock"
OURS="$OUT/containers"     # the containers these checks started, one name per line
SESSIONS="$OUT/sessions"   # the session ids of this run's processes (lock holders among them are ours)

# ---- whose work is on the machine ----
ours() { grep -qxF "$1" "$OURS" 2>/dev/null; }
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
  local why=() name id ids="" pid cg
  # Containers with the GPU device that these checks did not start.
  while read -r name; do
    [[ -n "$name" ]] || continue
    ours "$name" && { ids+=" $(docker inspect -f '{{.Id}}' "$name" 2>/dev/null)"; continue; }
    docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}{{len .HostConfig.DeviceRequests}}' "$name" 2>/dev/null \
      | grep -qE '/dev/nvidia|[1-9]$' && why+=("container $name has the GPU")
  done < <(docker ps --format '{{.Names}}')
  # GPU processes outside these checks' containers.
  while IFS=, read -r pid _; do
    pid="${pid// /}"; [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cg="$(cat "/proc/$pid/cgroup" 2>/dev/null || true)"
    for id in $ids; do [[ "$cg" == *"$id"* ]] && continue 2; done
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

stop_ours() {
  local name
  while read -r name; do
    [[ -n "$name" ]] && docker container inspect "$name" >/dev/null 2>&1 && docker rm -f "$name" >/dev/null 2>&1
  done < <(cat "$OURS" 2>/dev/null)
  systemctl --user stop sharar-memory-guard.service >/dev/null 2>&1 || true   # run-sharar.sh's guard of the served container
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
stage_image() {
  local old new
  old="$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
  bash docker/build.sh "$IMAGE" || return 1
  new="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
  [[ -n "$old" && "$old" != "$new" ]] && docker rmi "$old" >/dev/null 2>&1   # untagged by the build; kept if in use
  return 0
}

# tests/unit too: some unit tests need the model or the GPU and only run here. Test temp files go in ci-out/ on
# the host disk: the container's /tmp lacks RWF_NOWAIT, which tests/unit/test_ple_rows.py relies on.
stage_gpu() {
  local pid rc=0
  take_lock || return 1
  evict_models
  SHARAR_TEST_IMAGE="$IMAGE" bash tools/gpu-tests.sh tests/unit tests/gpu --basetemp=ci-out/pytest-tmp &
  pid=$!
  echo "sharar-gpu-tests-$pid" >>"$OURS"   # gpu-tests.sh names its container after its process id
  guard "sharar-gpu-tests-$pid" 8>&- &
  wait "$pid" || rc=$?
  return "$rc"
}

guard() {  # tools/memory-guard.sh on a container once it exists (run-sharar.sh starts its own)
  local _
  for _ in $(seq 120); do
    docker container inspect "$1" >/dev/null 2>&1 && exec bash tools/memory-guard.sh "$1"
    sleep 1
  done
}

# One target's launcher config with the CI image, container, port and context; no draft-runs, no saved sessions.
serve_config() {
  local base=$SHARAR_CONFIG_FILE conf="$OUT/serve-$1.env"
  [[ "$1" == flashnext ]] && base=config/flashnext.env
  { grep -vE '^(SHARAR_IMAGE|SHARAR_CONTAINER|SHARAR_PORT|SHARAR_MAX_CONTEXT|SHARAR_DRAFT|SHARAR_SESSION_DIR|SHARAR_HF_CACHE|SHARAR_RESULTS|SHARAR_FLASHNEXT_CACHE)=' "$base"
    echo "SHARAR_HF_CACHE=$SHARAR_HF_CACHE"; echo "SHARAR_RESULTS=$SHARAR_RESULTS"
    [[ -n "${SHARAR_FLASHNEXT_CACHE:-}" ]] && echo "SHARAR_FLASHNEXT_CACHE=$SHARAR_FLASHNEXT_CACHE"
    echo "SHARAR_IMAGE=$IMAGE"; echo "SHARAR_CONTAINER=$SERVE_CONTAINER"; echo "SHARAR_PORT=$SERVE_PORT"
    echo "SHARAR_MAX_CONTEXT=32768"; echo "SHARAR_DRAFT=pinned"; echo "SHARAR_SESSION_DIR=off"; } >"$conf"
  echo "$conf"
}

stage_serve() {
  local target conf waited rc=0
  take_lock || return 1
  echo "$SERVE_CONTAINER" >>"$OURS"
  for target in qwen38 flashnext; do
    conf="$(serve_config "$target")"
    evict_models
    SHARAR_CONFIG_FILE="$conf" bash run-sharar.sh 8>&- || { echo "$target: launch failed"; rc=1; continue; }
    waited=0
    until curl -fs "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; do
      if [[ "$(docker container inspect -f '{{.State.Running}}' "$SERVE_CONTAINER" 2>/dev/null)" != true ]] || (( waited >= 900 )); then
        echo "$target: not healthy"; docker logs --tail 100 "$SERVE_CONTAINER" 2>&1; rc=1; break
      fi
      sleep 5; waited=$(( waited + 5 ))
    done
    if (( waited < 900 )) && curl -fs "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; then
      if SHARAR_URL="http://127.0.0.1:$SERVE_PORT" SHARAR_MAX_CONTEXT=32768 bash tools/verify-sharar.sh; then
        echo "$target: verified (healthy after ${waited}s)" >>"$OUT/notes-serve"
      else
        echo "$target: verification failed"; docker logs --tail 100 "$SERVE_CONTAINER" 2>&1; rc=1
      fi
    fi
    SHARAR_CONFIG_FILE="$conf" bash run-sharar.sh --stop 8>&- || docker rm -f "$SERVE_CONTAINER" >/dev/null 2>&1
  done
  return "$rc"
}

# The image must be this checkout's build (docker/build.sh labels it with the commit); the GPU is not used.
stage_publish() {
  local commit rev tag
  commit="$(git rev-parse HEAD)"
  rev="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE")"
  [[ "$rev" == "$commit" ]] || { echo "$IMAGE is from ${rev:-an unlabeled build}, not $commit"; return 1; }
  [[ -n "${GHCR_TOKEN:-}" ]] || { echo "GHCR_TOKEN is not set"; return 1; }
  export DOCKER_CONFIG="$OUT/docker-config"
  mkdir -p "$DOCKER_CONFIG"
  printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "${GHCR_USER:-${GITHUB_ACTOR:-sharar-ci}}" --password-stdin >/dev/null || return 1
  for tag in "main-${commit:0:7}" latest; do
    docker tag "$IMAGE" "$REGISTRY:$tag" && docker push -q "$REGISTRY:$tag" >/dev/null || { echo "push of $tag failed"; return 1; }
  done
  docker rmi "$REGISTRY:main-${commit:0:7}" >/dev/null 2>&1   # keep only :latest tagged locally
  echo "pushed $REGISTRY:main-${commit:0:7} and :latest" >>"$OUT/notes-publish"
  docker logout ghcr.io >/dev/null 2>&1; rm -rf "$DOCKER_CONFIG"
}

# tools/evaluate.py takes the GPU lock itself for its serving window.
stage_evaluate() {
  local target name dir base rc=0 verdict paired
  echo sharar-evaluate >>"$OURS"
  for target in qwen38 flashnext; do
    name="ci-weekly-$target-$(date -u +%Y%m%dT%H%M)"
    base="$SHARAR_RESULTS/ci/baseline-$target"
    paired=(); [[ -e "$base" ]] && paired=(--baseline "$(realpath "$base")")
    evict_models
    python3 tools/evaluate.py --target "$target" --profile fast --image "$IMAGE" --tasks gsm8k,humaneval --limit 300 --speed --workload all \
      --name "$name" --evals "$EVALS" "${paired[@]}" || { echo "$target: evaluate did not complete"; rc=1; continue; }
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
: >"$OURS"; ps -o sid= -p $$ | tr -d ' ' >"$SESSIONS"
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

depends() { case $1 in gpu|serve|evaluate) echo image;; publish) echo image gpu serve;; esac; }
[[ -n "$busy" ]] || { watchdog & watchdog_pid=$!; }
missing=" "
for s in $ALL; do
  wants "$s" || continue
  dep="$(depends "$s")"
  if [[ -n "$busy" ]]; then rows+=("| $s | skipped | machine in use |")
  elif [[ -f "$OUT/gave-way" ]]; then rows+=("| $s | skipped | gave way to other work |")
  elif [[ "$s" == publish ]] && ! { wants gpu && wants serve; }; then rows+=("| $s | skipped | needs gpu and serve in the same run |")
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
