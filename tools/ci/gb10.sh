#!/usr/bin/env bash
# Nightly and weekly checks on the Spark.
#
#   Nightly:  tools/ci/gb10.sh [--skip-if-tested] image gpu serve
#   Weekly:   tools/ci/gb10.sh image serve bench accuracy
#   Cleanup:  tools/ci/gb10.sh --cleanup
#
# Stages:
#   image     build the code into the sharar:ci image
#   gpu       run the unit and GPU tests in that image
#   serve     start a server from that image and check that it answers
#   bench     speed test, fails if more than 5% slower than the baseline
#   accuracy  GSM8K, fails if clearly worse than the baseline
#
# If someone else is using the Spark, every stage is skipped.
# Settings come from ~/sharar-ci/gb10.env. Weekly results are saved in $SHARAR_RESULTS/ci/weekly/,
# and the first run that passes becomes the baseline. To use a newer run as the baseline:
#   ln -sfn "$SHARAR_RESULTS/ci/weekly/<time>" "$SHARAR_RESULTS/ci/baseline"
set -Euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ALL="image gpu serve bench accuracy"
IMAGE=sharar:ci
SERVE_CONTAINER=sharar-ci
URL=http://127.0.0.1:30011
OUT="$PWD/ci-out"
MAX_SPEED_DROP=5      # percent, the same limit as bench/compare_contracts.py
GSM8K_MIN_DROP=1.0    # points
GSM8K_MAX_P=0.05

export SHARAR_CONFIG_FILE="${SHARAR_CONFIG_FILE:-$HOME/sharar-ci/gb10.env}"
[[ -r "$SHARAR_CONFIG_FILE" ]] || { echo "missing config $SHARAR_CONFIG_FILE" >&2; exit 2; }
set -a; source "$SHARAR_CONFIG_FILE"; set +a   # run-sharar.sh needs these paths, even to stop a server
: "${SHARAR_HF_CACHE:?set SHARAR_HF_CACHE in $SHARAR_CONFIG_FILE}"
: "${SHARAR_RESULTS:?set SHARAR_RESULTS in $SHARAR_CONFIG_FILE}"

cleanup() {
  local gpu; gpu="$(cat "$OUT/gpu-container" 2>/dev/null || true)"
  if [[ -n "$gpu" ]] && docker container inspect "$gpu" >/dev/null 2>&1; then docker stop --time 30 "$gpu" >/dev/null; fi
  if docker container inspect "$SERVE_CONTAINER" >/dev/null 2>&1; then SHARAR_CONTAINER="$SERVE_CONTAINER" bash run-sharar.sh --stop; fi
  return 0
}
if [[ "${1:-}" == --cleanup ]]; then cleanup; exit 0; fi
skip_tested=0
if [[ "${1:-}" == --skip-if-tested ]]; then skip_tested=1; shift; fi

stages="${*:-$ALL}"
for s in $stages; do
  [[ " $ALL " == *" $s "* ]] || { echo "unknown stage '$s' (choose from: $ALL)" >&2; exit 2; }
done
wants() { [[ " $stages " == *" $1 "* ]]; }

commit="$(git rev-parse HEAD)"
TESTED="$SHARAR_RESULTS/ci/nightly-tested-commit"   # the last commit the nightly tests ran on
if (( skip_tested )) && [[ "$(cat "$TESTED" 2>/dev/null || true)" == "$commit" ]]; then
  echo "main is unchanged since the last nightly run (${commit:0:7}); skipped" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
  exit 0
fi

export SHARAR_PROFILE=fast   # what the lab serves; set here so a new default can't shift the baseline
BASELINE="$SHARAR_RESULTS/ci/baseline"
rm -rf "$OUT"; mkdir -p "$OUT"
trap cleanup EXIT
trap 'exit 130' INT TERM

# Skip everything if someone else is using the Spark: the lab's GPU lock is taken,
# a process is on the GPU, or less than 40 GB of memory is free.
busy=""
mkdir -p "$SHARAR_RESULTS"
exec 9>>"$SHARAR_RESULTS/.gpu.lock"   # held until the script exits
flock -n 9 || busy="another Sharar tool holds the GPU lock"
apps="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr '\n' ' ' | sed 's/ *$//' || true)"
[[ -z "${apps// }" ]] || busy="${busy:+$busy; }GPU in use by $apps"
avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
(( avail >= 40 )) || busy="${busy:+$busy; }only $avail GiB of memory free"

# Runs one stage and adds its row to the results table.
rows=() failed=0 note="" ran=" "
run_stage() {
  local start=$SECONDS rc=0
  note=""
  ran+="$1 "
  "stage_$1" || rc=$?
  local took="$(( (SECONDS - start) / 60 ))m $(( (SECONDS - start) % 60 ))s"
  if (( rc == 0 )); then rows+=("| $1 | passed | $took ${note} |")
  else rows+=("| $1 | **failed** | $took ${note} |"); failed=1; fi
  return "$rc"
}

stage_image() { bash docker/build.sh "$IMAGE"; }

# Starts the memory guard once the container is up.
guard() {
  local _
  for _ in $(seq 120); do
    docker container inspect "$1" >/dev/null 2>&1 && exec bash tools/memory-guard.sh "$1"
    sleep 1
  done
}

# tests/unit is included because some unit tests need the model or the GPU and only run here.
stage_gpu() {
  local pid gpid rc=0
  SHARAR_TEST_IMAGE="$IMAGE" bash tools/gpu-tests.sh tests/unit tests/gpu &
  pid=$!
  echo "sharar-gpu-tests-$pid" >"$OUT/gpu-container"   # gpu-tests.sh names its container after its process ID
  guard "sharar-gpu-tests-$pid" 9>&- &   # keep the lock away from the guard, so a leftover guard can't hold it
  gpid=$!
  wait "$pid" || rc=$?
  kill "$gpid" 2>/dev/null; wait "$gpid" 2>/dev/null
  return "$rc"
}

# 32K context:  what verify-sharar.sh expects
# pinned draft: the published drafter, not one the lab trained later
# no sessions:  the benchmark clears the cache, which would also delete the lab's saved sessions
stage_serve() {
  local waited=0
  SHARAR_IMAGE="$IMAGE" SHARAR_CONTAINER="$SERVE_CONTAINER" SHARAR_PORT=30011 \
    SHARAR_MAX_CONTEXT=32768 SHARAR_DRAFT=pinned SHARAR_SESSION_DIR=off bash run-sharar.sh || return 1
  until curl -fs "$URL/health" >/dev/null 2>&1; do
    if [[ "$(docker container inspect -f '{{.State.Running}}' "$SERVE_CONTAINER" 2>/dev/null)" != true ]] || (( waited >= 900 )); then
      echo "server did not become healthy"; docker logs --tail 200 "$SERVE_CONTAINER"; return 1
    fi
    sleep 5; waited=$(( waited + 5 ))
  done
  note="(healthy after ${waited}s)"
  SHARAR_URL="$URL" bash tools/verify-sharar.sh || { docker logs --tail 200 "$SERVE_CONTAINER"; return 1; }
}

# The benchmark runs outside the container, so it gets its own Python with CPU-only PyTorch.
# --accept omit makes it use the server's own (fast) settings.
stage_bench() {
  local base=() msg rc
  python3 -m venv .venv-bench \
    && .venv-bench/bin/pip install -q -c constraints/cpu.txt torch --index-url https://download.pytorch.org/whl/cpu \
    && .venv-bench/bin/pip install -q -c constraints/cpu.txt -e . \
    || { note="(could not install the benchmark client)"; return 1; }
  if [[ -f "$BASELINE/bench/summary.json" ]]; then
    base=(--inputs "$BASELINE/bench/frozen-inputs.json" --baseline "$BASELINE/bench")
  fi
  .venv-bench/bin/python -m sharar benchmark --url "$URL" --workload all --accept omit --output "$OUT/bench" ${base[@]+"${base[@]}"} \
    || { note="(benchmark did not complete; see the log)"; return 1; }
  msg="$(python3 - "$OUT/bench/summary.json" "$MAX_SPEED_DROP" <<'EOF'
import json, sys
comparison, limit = json.load(open(sys.argv[1])).get("comparison"), float(sys.argv[2])
if not comparison:
    print("(no baseline yet)"); sys.exit(0)
slow = [f"{case} {key.split('_')[0]} {change:+.1f}%" for case, c in comparison.items()
        for key, change in c.items() if key.endswith("_change_pct") and change < -limit]
print(f"(slower than the baseline by more than {limit:g}%: {', '.join(slow)})" if slow
      else f"(within {limit:g}% of the baseline)")
sys.exit(1 if slow else 0)
EOF
)"; rc=$?
  note="$msg"
  return "$rc"
}

# All 1,319 GSM8K questions, compared one by one with the baseline run.
stage_accuracy() {
  local msg rc
  python3 tools/eval_gsm8k.py --base-url "$URL" --n 1319 --out "$OUT/gsm8k.json" \
    || { note="(GSM8K run did not complete; see the log)"; return 1; }
  msg="$(python3 - "$OUT/gsm8k.json" "$BASELINE/gsm8k.json" "$GSM8K_MIN_DROP" "$GSM8K_MAX_P" <<'EOF'
import json, math, os, sys
run, min_drop, max_p = json.load(open(sys.argv[1])), float(sys.argv[3]), float(sys.argv[4])
right = lambda r: r["pred"] is not None and r["pred"] == r["gold"]   # same scoring as eval_gsm8k.py
errors = sum(1 for r in run["results"] if r["text"].startswith("ERROR"))
msg, bad = f"GSM8K {run['correct']}/{run['n']} = {100 * run['correct'] / run['n']:.1f}%", errors > 0
if errors:
    msg += f", {errors} request errors"
if not os.path.isfile(sys.argv[2]):
    msg += ", no baseline yet"
else:
    base = json.load(open(sys.argv[2]))
    if [r["gold"] for r in base["results"]] != [r["gold"] for r in run["results"]]:
        print(f"({msg}; the baseline has different questions)"); sys.exit(1)
    worse = sum(1 for b, r in zip(base["results"], run["results"]) if right(b) and not right(r))
    better = sum(1 for b, r in zip(base["results"], run["results"]) if right(r) and not right(b))
    n = worse + better
    # exact McNemar test, the same formula as the old bench/accuracy_eval.py
    p = min(1.0, sum(math.comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(worse - n / 2)) / 2 ** n) if n else 1.0
    drop = 100 * (worse - better) / run["n"]
    msg += f", {-drop:+.1f} points vs the baseline ({worse} worse, {better} better, p = {p:.3f})"
    # fail only if the drop is at least 1 point, mostly in one direction, and not likely to be chance
    bad = bad or (drop >= min_drop and worse > better and p < max_p)
print(f"({msg})")
sys.exit(1 if bad else 0)
EOF
)"; rc=$?
  note="$msg"
  return "$rc"
}

# Run the stages in order. A stage is skipped if the one it needs didn't pass, so old code never gets tested.
depends() { case $1 in gpu|serve) echo image;; bench|accuracy) echo serve;; esac; }
missing=" "
for s in $ALL; do
  wants "$s" || continue
  dep="$(depends "$s")"
  if [[ -n "$busy" ]]; then rows+=("| $s | skipped | Spark busy |")
  elif [[ -n "$dep" && "$missing" == *" $dep "* ]]; then rows+=("| $s | skipped | $dep did not pass |")
  elif run_stage "$s"; then continue
  fi
  missing+="$s "
done
[[ -z "$busy" ]] || echo "::warning::Spark busy ($busy); checks skipped"
if [[ "$ran" == *" gpu "* ]]; then mkdir -p "$(dirname "$TESTED")" && echo "$commit" >"$TESTED"; fi

# Weekly runs are kept, and the first one that fully passes becomes the baseline.
keep="" new_baseline=0
if { wants bench || wants accuracy; } && [[ -z "$busy" ]]; then
  keep="$SHARAR_RESULTS/ci/weekly/$(date -u +%Y%m%dT%H%M%SZ)"
  [[ ! -e "$BASELINE" && "$missing" == " " ]] && new_baseline=1
fi

# The results table, printed and shown on the GitHub run page.
{
  echo "## GB10 checks, $(date -u '+%Y-%m-%d %H:%M UTC'), ${commit:0:7}, profile $SHARAR_PROFILE"
  [[ -z "$busy" ]] || printf '\nSkipped: %s.\n' "$busy"
  echo
  echo "| Stage | Result | Time and notes |"
  echo "|---|---|---|"
  printf '%s\n' "${rows[@]}"
  if [[ -f "$OUT/bench/summary.json" ]]; then
    echo
    python3 - "$OUT/bench/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
comparison = s.get("comparison", {})
print("| Workload | End-to-end TPS | Decode TPS | TTFT (s) | vs baseline (end-to-end, decode) |")
print("|---|---:|---:|---:|---|")
for name, c in s["cases"].items():
    d = comparison.get(name)
    vs = f"{d['output_tps_change_pct']:+.1f}%, {d['decode_tps_change_pct']:+.1f}%" if d else "no baseline"
    print(f"| {name} | {c['output_tps']:.2f} | {c['decode_tps']:.2f} | {c['ttft_s']:.3f} | {vs} |")
EOF
  fi
  (( new_baseline )) && printf '\nThis run is now the baseline for later comparisons.\n'
} >"$OUT/summary.md"
cat "$OUT/summary.md"
[[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && cat "$OUT/summary.md" >>"$GITHUB_STEP_SUMMARY"
if [[ -n "$keep" ]]; then
  mkdir -p "$(dirname "$keep")" && cp -R "$OUT" "$keep" && echo "kept in $keep"
  (( new_baseline )) && ln -s "$keep" "$BASELINE"
fi
exit "$failed"
