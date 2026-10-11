"""What `sharar serve`, `sharar train`, `stop`, `status` and `logs` do: settings, preflight, then the engine.

`sharar serve` runs the server in this process, in the foreground (Ctrl-C drains and stops it), with the memory
guard as a thread; `--detach` runs the same command as a user systemd service; `--tp 2` supervises one rank here and
one on the peer (launch/tp.py). Nothing here imports torch before the engine starts.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

from sharar.launch import ctl, guard, preflight, service
from sharar.launch.settings import Settings, resolve

TRAIN_SERVICE = "sharar-train"


def engine_runtime_env() -> dict[str, str]:
    """What the engine process needs besides its settings: offline model loads and the CUDA toolkit for its first-use
    kernel builds (nvcc, and this environment's ninja)."""
    venv_bin = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "/usr/bin:/bin")
    return {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"), "PATH": f"{venv_bin}:/usr/local/cuda/bin:{path}"}


def install_env(env: dict[str, str]) -> None:
    """This process's environment for the engine: the resolved settings, and none of the caller's that they replace."""
    for key in ("SHARAR_DRAFT", "SHARAR_TRACES", "SHARAR_W3_CHECKPOINT", "SHARAR_SESSION_DIR", "SHARAR_W3_WARNING"):
        if key not in env:
            os.environ.pop(key, None)
    os.environ.update(env)


def serve_overrides(args) -> dict[str, str]:
    """The serve command's flags as SHARAR_* settings."""
    o = {}
    for flag, key in (("port", "SHARAR_PORT"), ("host", "SHARAR_HOST"), ("context_length", "SHARAR_MAX_CONTEXT"),
                      ("max_batch", "SHARAR_MAX_BATCH"), ("proposer", "SHARAR_PROPOSER"), ("spec", "SHARAR_SPEC"),
                      ("draft_precision", "SHARAR_DRAFT_PRECISION"), ("spec_depths", "SHARAR_SPEC_DEPTHS"),
                      ("accept_default", "SHARAR_ACCEPT_DEFAULT"), ("service", "SHARAR_SERVICE")):
        value = getattr(args, flag, None)
        if value is not None:
            o[key] = str(value)
    if getattr(args, "pinned", False):
        o["SHARAR_DRAFT"] = "pinned"
    elif getattr(args, "draft", None):
        o["SHARAR_DRAFT"] = str(Path(args.draft).expanduser().resolve())
    if getattr(args, "no_sessions", False):
        o["SHARAR_SESSION_DIR"] = "off"
    return o


def settings_for(args, **kw) -> Settings:
    return resolve(getattr(args, "target", None), getattr(args, "profile", None), serve_overrides(args), **kw)


def serve(args) -> int:
    try:
        s = settings_for(args, traces=not args.no_traces, tp=args.tp)
    except ValueError as exc:
        sys.exit(f"sharar serve: {exc}")
    if args.detach:
        argv = [a for a in sys.argv[1:] if a not in ("--detach", "-d")]
        try:
            preflight.check_host()
            preflight.ensure_snapshots(s)  # downloads show here, not in the service's log
            preflight.ensure_w3(s)
            service.start(s.service, [*argv, "--no-fetch"], s.log_file(), f"Sharar {s.target}")
        except (preflight.PreflightError, RuntimeError) as exc:
            sys.exit(f"sharar serve: {exc}")
        print(f"started {s.service}; log {s.log_file()}")
        return service.follow_startup(s.service, s.log_file(), s.url)
    if ctl.server_info(str(s.port)) is not None:
        sys.exit(f"sharar serve: something already serves {s.url} (sharar stop, or stop the server in its terminal)")
    try:
        host = preflight.check_host()
        missing = preflight.ensure_snapshots(s, fetch=not args.no_fetch)
        if missing:
            raise preflight.PreflightError(f"missing from the model cache: {', '.join(missing)}")
        w3 = preflight.ensure_w3(s, fetch=not args.no_fetch)
    except preflight.PreflightError as exc:
        sys.exit(f"sharar serve: {exc}")
    print(preflight.summary(s, host, w3, args.tp), flush=True)
    for d in (s.traces, s.sessions):
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
    if s.parallel_draft:
        s.draft_runs.mkdir(parents=True, exist_ok=True)
    if args.tp > 1:
        from sharar.launch import tp
        return tp.serve(s, w3)
    install_env({**s.engine_env(w3), **engine_runtime_env()})
    guard.start_thread()
    from sharar.cli import engine_parser, run_engine
    parser = engine_parser()
    return run_engine(parser.parse_args(s.engine_args()), parser)


def stop(args) -> int:
    targets = [args.target] if args.target else ["27b", "flashnext"]
    stopped = False
    for name in targets:
        s = resolve(name)
        if service.stop(s.service):
            print(f"stopped {s.service}")
            stopped = True
        elif ctl.server_info(str(s.port)) is not None and args.target:
            print(f"{s.url} is served in the foreground: stop it with Ctrl-C in its terminal")
            stopped = True
    if not stopped:
        print("nothing to stop (no detached server)")
    return 0


def serving(target: str | None = None) -> Settings | None:
    """The named target, else whichever target answers on its port (the 27B first)."""
    if target:
        return resolve(target)
    for name in ("27b", "flashnext"):
        s = resolve(name)
        if ctl.server_info(str(s.port)) is not None:
            return s
    return None


def status(args) -> int:
    any_up = False
    for name in ([args.target] if args.target else ["27b", "flashnext"]):
        s = resolve(name)
        info = ctl.server_info(str(s.port))
        how = "a detached service" if service.running(s.service) else "the foreground"
        if info is None:
            if service.running(s.service):
                print(f"{s.target}: starting ({s.service}; sharar logs -f)")
                any_up = True
            continue
        any_up = True
        cfg, spec, stats = info.get("config") or {}, info.get("speculative") or {}, info.get("spec_stats") or {}
        print(f"{s.target}: serving on {s.url} from {how}, profile {cfg.get('profile')}, {info.get('tp_size', 1)} Spark(s), "
              f"context {info.get('context_length')}, up to {info.get('max_running_requests')} requests")
        print(f"  draft {spec.get('draft') or 'the target MTP head'}, precision {spec.get('draft_precision')}")
        if stats.get("rounds"):
            print(f"  {stats['rounds']} speculative rounds: {stats['tokens_per_round']:.2f} tokens per round, {stats.get('round_ms', 0):.1f} ms each")
        traces = (info.get("scheduler") or {}).get("traces") or info.get("traces")
        print(f"  traces: {traces['records']} recorded, {traces['bytes'] / 2**20:.0f} MiB" if traces else "  traces: off")
    if not any_up:
        print("nothing is serving (sharar serve 27b, or flashnext)")
    print("training: " + ("running (sharar train stop finishes it)" if service.running(TRAIN_SERVICE) else "not running"))
    s27 = resolve("27b")
    draft = ctl.current_draft(s27)
    print(f"27B draft: {draft if draft else 'pinned publisher draft (nothing promoted yet)'}")
    return 0


def logs(args) -> int:
    log = resolve("27b").log_file(TRAIN_SERVICE) if args.train else (serving(args.target) or resolve(args.target)).log_file()
    if not log.is_file():
        sys.exit(f"no log at {log} (logs are kept for detached servers and training; a foreground server prints to its terminal)")
    return subprocess.call(["tail", "-n", str(args.lines), *(["-F"] if args.follow else []), str(log)])


def train(args, parser) -> int:
    """`sharar train`: the 27B's DFlash2 drafter on the recorded traces; stop, status, or a (foreground) run."""
    s = resolve("27b", overrides=serve_overrides(args))
    if args.action == "stop":
        if service.stop(TRAIN_SERVICE):
            print(f"stopped training; current draft: {ctl.current_draft(s) or 'pinned'} (sharar serve 27b serves it)")
        else:
            print("training is not running as a service (a foreground run stops with Ctrl-C)")
        return 0
    if args.action == "status":
        print("training: " + ("running" if service.running(TRAIN_SERVICE) else "not running"))
        current = ctl.current_draft(s)
        print(f"current draft: {current or 'pinned publisher draft (nothing promoted yet)'}")
        return 0
    if s.target != "qwen38":
        sys.exit("training adapts the 27B's DFlash2 drafter; Flash-Next's MTP head is not trainable yet (#40)")
    runs = s.draft_runs
    args.traces = args.traces or str(s.traces or s.results / "traces")
    if not Path(args.traces).is_dir():
        sys.exit(f"no traces at {args.traces}: serve with traces on (the default) first")
    args.output = args.output or str(runs / f"run-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    args.promote_to = None if args.promote_to == "none" else (args.promote_to or str(runs / "current"))
    args.continue_from = None if args.continue_from == "none" else (args.continue_from or str(runs / "current"))
    regression = s.results / "draft-regression"
    if args.regression_traces is None and regression.is_dir():
        args.regression_traces = str(regression)
    up = ctl.server_info(str(s.port)) is not None
    if up and not args.keep_serving:
        if service.running(s.service):
            service.stop(s.service)
            print(f"stopped {s.service}: training and serving do not share the GPU")
        else:
            sys.exit(f"the 27B is serving on {s.url} in the foreground: stop it first (training and serving do not share the GPU)")
    if args.detach:
        argv = [a for a in sys.argv[1:] if a not in ("--detach", "-d")]
        try:
            service.start(TRAIN_SERVICE, [*argv, "--output", args.output], s.log_file(TRAIN_SERVICE), "Sharar drafter training",
                          stop_seconds=int(s.get("SHARAR_TRAIN_STOP_GRACE_S", "1200")))
        except RuntimeError as exc:
            sys.exit(f"sharar train: {exc}")
        print(f"training run {args.output}; log {s.log_file(TRAIN_SERVICE)}; sharar train stop finishes it (final evaluation, promotion)")
        return 0
    runs.mkdir(parents=True, exist_ok=True)
    install_env({**s.engine_env(None), **engine_runtime_env()})
    guard.start_thread()
    print(f"training run {args.output} on {args.traces}; Ctrl-C finishes it (final evaluation and promotion)", flush=True)
    from sharar.cli import run_train
    code = run_train(args, parser)
    print(f"current draft: {ctl.current_draft(s) or 'pinned'}; serve it with sharar serve 27b")
    return code
