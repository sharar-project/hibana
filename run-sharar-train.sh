#!/usr/bin/env bash
# Train the DFlash2 drafter on this machine's request traces while nothing is served.
# Usage: ./run-sharar-train.sh start [--keep-serving] | stop [--no-serve] | status
# Reads config/gb10.env for SHARAR_HF_CACHE, SHARAR_RESULTS and SHARAR_TRACES.
#
# start: stops the serving container (the GPU has no spare bandwidth), then trains
#        until `stop`, from the current promoted checkpoint when one exists, on every
#        trace in SHARAR_TRACES with the newest requests held out (about 20K output
#        tokens). Promotions update $SHARAR_RESULTS/draft-runs/current.
# stop:  asks the trainer to finish (final held-out evaluation, promotion), then
#        relaunches serving, which follows draft-runs/current by default.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SHARAR_CONFIG_FILE:-$SCRIPT_DIR/config/gb10.env}"
[[ -r "$CONFIG_FILE" ]] && source "$CONFIG_FILE"
IMAGE="${SHARAR_IMAGE:-sharar:qwen38-27b}"
CONTAINER="${SHARAR_TRAIN_CONTAINER:-sharar-train}"
SERVING_CONTAINER="${SHARAR_CONTAINER:-sharar-qwen38}"
CACHE_DIR="${SHARAR_HF_CACHE:?set SHARAR_HF_CACHE in config/gb10.env}"
RESULTS_DIR="${SHARAR_RESULTS:?set SHARAR_RESULTS in config/gb10.env}"
TRACES_DIR="${SHARAR_TRACES:?set SHARAR_TRACES in config/gb10.env to the trace directory serving writes}"
RUNS_DIR="${SHARAR_DRAFT_RUNS:-$RESULTS_DIR/draft-runs}"
STATE_DIR="$RESULTS_DIR/runtime-state"
STOP_GRACE_S="${SHARAR_TRAIN_STOP_GRACE_S:-1200}"   # a final held-out evaluation takes a few minutes
EVAL_EVERY="${SHARAR_TRAIN_EVAL_EVERY:-100}"
HOLDOUT_TOKENS="${SHARAR_TRAIN_HOLDOUT_TOKENS:-20000}"
LR="${SHARAR_TRAIN_LR:-1e-4}"
TRAIN_PARAMS="${SHARAR_TRAIN_PARAMS:-attn}"     # attn | all | selector
KL_ALPHA="${SHARAR_TRAIN_KL_ALPHA:-1.0}"
CONTINUE_LR="${SHARAR_TRAIN_CONTINUE_LR:-3e-5}"  # used when continuing from a promoted checkpoint
ON_POLICY="${SHARAR_TRAIN_ON_POLICY:-0}"
REGRESSION_DIR="${SHARAR_TRAIN_REGRESSION:-$RESULTS_DIR/draft-regression}"  # optional fixed general trace set
DRIVER_LIB_DIR="/usr/lib/aarch64-linux-gnu"

fail() { echo "sharar-train: $*" >&2; exit 1; }
running() { [[ "$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null || echo false)" == true ]]; }

case "${1:-}" in
  start)
    shift
    keep_serving=false
    [[ "${1:-}" == "--keep-serving" ]] && keep_serving=true
    running "$CONTAINER" && fail "training container $CONTAINER is already running (use: $0 status)"
    [[ -d "$TRACES_DIR" ]] || fail "trace directory $TRACES_DIR does not exist; serve with SHARAR_TRACES first"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image is missing: $IMAGE (build with docker/build.sh)"
    # The image must carry this checkout's trainer: an older image fails at argument parsing after serving was stopped.
    docker run --rm --entrypoint python3 "$IMAGE" -m sharar train --help 2>/dev/null | grep -q -- "--holdout-tokens" \
      || fail "image $IMAGE predates the current trainer; rebuild it with docker/build.sh"
    if running "$SERVING_CONTAINER"; then
      $keep_serving && fail "serving container $SERVING_CONTAINER is running; stop it or pass --keep-serving only on a second machine"
      "$SCRIPT_DIR/run-sharar.sh" --stop
    fi
    mkdir -p "$RUNS_DIR" "$STATE_DIR"
    run="run-$(date -u +%Y%m%d-%H%M%S)"
    args=(docker run --detach --name "$CONTAINER"
      --security-opt no-new-privileges=true --cap-drop ALL --pids-limit 2048 --shm-size 16g
      --log-opt max-size=50m --log-opt max-file=2 --stop-timeout "$STOP_GRACE_S"
      --user "$(id -u):$(id -g)"
      --label io.sharar.target=gb10-sm121
      --volume "$CACHE_DIR:/cache/huggingface:ro"
      --volume "$STATE_DIR:/state"
      --volume "$TRACES_DIR:/traces:ro"
      --volume "$RUNS_DIR:/runs"
      --env TRITON_LIBCUDA_PATH=$DRIVER_LIB_DIR)
    source "$SCRIPT_DIR/tools/gb10-container.sh"
    gb10_container_args
    train_args=(train --traces /traces --output "/runs/$run" --steps 0
            --promote-to /runs/current --continue-from /runs/current --continue-lr "$CONTINUE_LR"
            --eval-every "$EVAL_EVERY" --holdout-tokens "$HOLDOUT_TOKENS" --lr "$LR"
            --train-params "$TRAIN_PARAMS" --kl-alpha "$KL_ALPHA")
    [[ "$ON_POLICY" == 1 ]] && train_args+=(--on-policy)
    if [[ -d "$REGRESSION_DIR" ]]; then
      args+=(--volume "$REGRESSION_DIR:/regression:ro")
      train_args+=(--regression-traces /regression)
    fi
    cid="$("${args[@]}" "$IMAGE" "${train_args[@]}")"
    echo "Started $CONTAINER ($cid): run $RUNS_DIR/$run; logs: docker logs -f $CONTAINER; stop with: $0 stop"
    ;;
  stop)
    shift
    running "$CONTAINER" || fail "training container $CONTAINER is not running"
    echo "stopping $CONTAINER: the trainer finishes its step, evaluates once more and promotes (up to ${STOP_GRACE_S}s)"
    docker stop --timeout "$STOP_GRACE_S" "$CONTAINER" >/dev/null
    docker logs --tail 5 "$CONTAINER" 2>&1 | sed 's/^/  /'
    docker rm "$CONTAINER" >/dev/null 2>&1 || true
    if [[ -L "$RUNS_DIR/current" ]]; then
      echo "current draft: $(readlink -f "$RUNS_DIR/current")"
    else
      echo "no promoted draft yet; serving keeps the pinned draft"
    fi
    if [[ "${1:-}" != "--no-serve" ]]; then
      if running "$SERVING_CONTAINER"; then
        # a server kept running (second machine): swap the promoted draft in place
        "$SCRIPT_DIR/sharar-ctl" draft reload
      else
        "$SCRIPT_DIR/run-sharar.sh"
      fi
    fi
    ;;
  status)
    if running "$CONTAINER"; then
      echo "training: running ($(docker inspect --format '{{.State.StartedAt}}' "$CONTAINER"))"
      docker logs --tail 3 "$CONTAINER" 2>&1 | sed 's/^/  /'
    else
      echo "training: not running"
    fi
    if [[ -L "$RUNS_DIR/current" ]]; then
      target="$(readlink -f "$RUNS_DIR/current")"
      echo "current draft: $target"
      [[ -f "$target/manifest.json" ]] && python3 -c "import json,sys; m=json.load(open(sys.argv[1])); e=m.get('eval',{}); print(f\"  step {m.get('step')}: held-out tokens/round {e.get('tokens_per_round')}\")" "$target/manifest.json"
    else
      echo "current draft: pinned publisher draft (nothing promoted yet)"
    fi
    running "$SERVING_CONTAINER" && echo "serving: running" || echo "serving: stopped"
    ;;
  *) echo "usage: $0 start [--keep-serving] | stop [--no-serve] | status" >&2; exit 2 ;;
esac
