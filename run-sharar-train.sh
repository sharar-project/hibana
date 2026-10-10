#!/usr/bin/env bash
# Train the DFlash2 drafter on this machine's request traces while nothing is served: `sharar train` from this
# checkout's Python environment as the user systemd service SHARAR_TRAIN_SERVICE (default sharar-train).
# Usage: ./run-sharar-train.sh start [--keep-serving] | stop [--no-serve] | status
# Reads config/gb10.env for SHARAR_HF_CACHE, SHARAR_RESULTS and SHARAR_TRACES.
#
# start: stops serving (the GPU has no spare bandwidth), then trains
#        until `stop`, from the current promoted checkpoint when one exists, on every
#        trace in SHARAR_TRACES with the newest requests held out (about 20K output
#        tokens). Promotions update $SHARAR_RESULTS/draft-runs/current.
# stop:  asks the trainer to finish (final held-out evaluation, promotion), then
#        relaunches serving, which follows draft-runs/current by default.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SHARAR_CONFIG_FILE:-$SCRIPT_DIR/config/gb10.env}"
REQUESTED_TARGET="${SHARAR_TARGET:-}"   # a target named by the caller (sharar-ctl serve flashnext) wins over the config file
[[ -r "$CONFIG_FILE" ]] && source "$CONFIG_FILE"
[[ -n "$REQUESTED_TARGET" ]] && SHARAR_TARGET="$REQUESTED_TARGET"
source "$SCRIPT_DIR/tools/launcher-config.sh"
launcher_defaults || exit 1
VENV="${SHARAR_VENV:-$SCRIPT_DIR/.venv}"
SERVICE="${SHARAR_TRAIN_SERVICE:-sharar-train}"
SERVING_SERVICE="${SHARAR_SERVICE:-sharar-qwen38}"
CACHE_DIR="$SHARAR_HF_CACHE"
RESULTS_DIR="$SHARAR_RESULTS"
TRACES_DIR="${SHARAR_TRACES:?set SHARAR_TRACES in config/gb10.env to the trace directory serving writes}"
RUNS_DIR="${SHARAR_DRAFT_RUNS:-$RESULTS_DIR/draft-runs}"
LOG="$RESULTS_DIR/logs/$SERVICE.log"
STOP_GRACE_S="${SHARAR_TRAIN_STOP_GRACE_S:-1200}"   # a final held-out evaluation takes a few minutes
EVAL_EVERY="${SHARAR_TRAIN_EVAL_EVERY:-100}"
HOLDOUT_TOKENS="${SHARAR_TRAIN_HOLDOUT_TOKENS:-20000}"
LR="${SHARAR_TRAIN_LR:-1e-4}"
TRAIN_PARAMS="${SHARAR_TRAIN_PARAMS:-attn}"     # attn | all | selector
KL_ALPHA="${SHARAR_TRAIN_KL_ALPHA:-1.0}"
CONTINUE_LR="${SHARAR_TRAIN_CONTINUE_LR:-3e-5}"  # used when continuing from a promoted checkpoint
ON_POLICY="${SHARAR_TRAIN_ON_POLICY:-0}"
REGRESSION_DIR="${SHARAR_TRAIN_REGRESSION:-$RESULTS_DIR/draft-regression}"  # optional fixed general trace set

fail() { echo "sharar-train: $*" >&2; exit 1; }
running() { systemctl --user is-active --quiet "$1"; }

case "${1:-}" in
  start)
    shift
    keep_serving=false
    [[ "${1:-}" == "--keep-serving" ]] && keep_serving=true
    running "$SERVICE" && fail "training ($SERVICE) is already running (use: $0 status)"
    [[ "$SHARAR_TARGET" == qwen38 ]] || fail "training adapts the 27B's DFlash2 drafter; Flash-Next's MTP head is not trainable here yet (#40)"
    [[ -d "$TRACES_DIR" ]] || fail "trace directory $TRACES_DIR does not exist; serve with SHARAR_TRACES first"
    bash "$SCRIPT_DIR/tools/setup-venv.sh" --check || bash "$SCRIPT_DIR/tools/setup-venv.sh" >&2 \
      || fail "could not build the Python environment (tools/setup-venv.sh)"
    if running "$SERVING_SERVICE"; then
      $keep_serving && fail "$SERVING_SERVICE is serving; stop it or pass --keep-serving only on a second machine"
      "$SCRIPT_DIR/run-sharar.sh" --stop
    fi
    mkdir -p "$RUNS_DIR" "$(dirname "$LOG")"
    run="run-$(date -u +%Y%m%d-%H%M%S)"
    train_args=(train --traces "$TRACES_DIR" --output "$RUNS_DIR/$run" --steps 0
            --promote-to "$RUNS_DIR/current" --continue-from "$RUNS_DIR/current" --continue-lr "$CONTINUE_LR"
            --eval-every "$EVAL_EVERY" --holdout-tokens "$HOLDOUT_TOKENS" --lr "$LR"
            --train-params "$TRAIN_PARAMS" --kl-alpha "$KL_ALPHA")
    [[ "$ON_POLICY" == 1 ]] && train_args+=(--on-policy)
    [[ -d "$REGRESSION_DIR" ]] && train_args+=(--regression-traces "$REGRESSION_DIR")
    [[ -f "$LOG" ]] && mv -f "$LOG" "$LOG.1"
    systemctl --user reset-failed "$SERVICE" 2>/dev/null || true
    systemd-run --user --quiet --collect --unit "$SERVICE" --description "Sharar drafter training" \
      --working-directory "$SCRIPT_DIR" --property KillSignal=SIGTERM --property "TimeoutStopSec=$STOP_GRACE_S" \
      --property "StandardOutput=append:$LOG" --property "StandardError=append:$LOG" \
      --setenv "PATH=$VENV/bin:/usr/local/cuda/bin:/usr/bin:/bin" --setenv CUDA_HOME=/usr/local/cuda \
      --setenv "PYTHONPATH=$SCRIPT_DIR" --setenv HF_HUB_OFFLINE=1 --setenv TOKENIZERS_PARALLELISM=false \
      --setenv PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --setenv "SHARAR_HF_CACHE=$CACHE_DIR" \
      --setenv "SHARAR_RESULTS=$RESULTS_DIR" --setenv SHARAR_TARGET=qwen38 \
      "$SCRIPT_DIR/tools/lxcfs-run.sh" "$VENV/bin/python" -u -m sharar "${train_args[@]}" || fail "could not start $SERVICE"
    echo "Started $SERVICE: run $RUNS_DIR/$run; logs: $LOG; stop with: $0 stop"
    ;;
  stop)
    shift
    running "$SERVICE" || fail "training ($SERVICE) is not running"
    echo "stopping $SERVICE: the trainer finishes its step, evaluates once more and promotes (up to ${STOP_GRACE_S}s)"
    systemctl --user stop "$SERVICE"
    tail -n 5 "$LOG" 2>/dev/null | sed 's/^/  /'
    if [[ -L "$RUNS_DIR/current" ]]; then
      echo "current draft: $(readlink -f "$RUNS_DIR/current")"
    else
      echo "no promoted draft yet; serving keeps the pinned draft"
    fi
    if [[ "${1:-}" != "--no-serve" ]]; then
      if running "$SERVING_SERVICE"; then
        # a server kept running (second machine): swap the promoted draft in place
        "$SCRIPT_DIR/sharar-ctl" draft reload
      else
        "$SCRIPT_DIR/run-sharar.sh"
      fi
    fi
    ;;
  status)
    if running "$SERVICE"; then
      echo "training: running (since $(systemctl --user show -p ActiveEnterTimestamp --value "$SERVICE"))"
      tail -n 3 "$LOG" 2>/dev/null | sed 's/^/  /'
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
    running "$SERVING_SERVICE" && echo "serving: running" || echo "serving: stopped"
    ;;
  *) echo "usage: $0 start [--keep-serving] | stop [--no-serve] | status" >&2; exit 2 ;;
esac
