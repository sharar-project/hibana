"""What `sharar serve`, `stop`, `status` and `logs` do: settings, preflight, then the engine.

`sharar serve` runs the server in this process, in the foreground (Ctrl-C drains and stops it), with the memory
guard as a thread; `--detach` runs the same command as a user systemd service; `--tp 2` supervises one rank here and
one on the peer (launch/tp.py). Nothing here imports torch before the engine starts.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sharar.launch import ctl, guard, preflight, service
from sharar.launch.settings import Settings, resolve


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
    if getattr(args, "draft", None):
        o["SHARAR_DRAFT"] = str(Path(args.draft).expanduser().resolve())
    if getattr(args, "no_sessions", False):
        o["SHARAR_SESSION_DIR"] = "off"
    return o


def settings_for(args, **kw) -> Settings:
    return resolve(getattr(args, "target", None), getattr(args, "profile", None), serve_overrides(args), **kw)


def serve(args) -> int:
    try:
        s = settings_for(args, traces=args.traces, tp=args.tp)
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
    return 0


def logs(args) -> int:
    log = (serving(args.target) or resolve(args.target)).log_file()
    if not log.is_file():
        sys.exit(f"no log at {log} (logs are kept for detached servers; a foreground server prints to its terminal)")
    return subprocess.call(["tail", "-n", str(args.lines), *(["-F"] if args.follow else []), str(log)])

