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
from sharar.launch.settings import SERVE_HINT, SERVED, Settings, config_file, read_config_file, resolve


def engine_runtime_env() -> dict[str, str]:
    """What the engine process needs besides its settings: offline model loads and the CUDA toolkit for its first-use
    kernel builds (nvcc, and this environment's ninja)."""
    venv_bin = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "/usr/bin:/bin")
    return {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"), "PATH": f"{venv_bin}:/usr/local/cuda/bin:{path}"}


def install_env(env: dict[str, str]) -> None:
    """This process's environment for the engine: the resolved settings, and none of the caller's that they replace."""
    for key in ("SHARAR_DRAFT", "SHARAR_TRACES", "SHARAR_W3_CHECKPOINT", "SHARAR_SSD_DIR", "SHARAR_W3_WARNING"):
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
    return o


def settings_for(args, **kw) -> Settings:
    return resolve(getattr(args, "target", None), getattr(args, "profile", None), serve_overrides(args), **kw)


def serve(args) -> int:
    if not args.target and not (os.environ.get("SHARAR_TARGET") or read_config_file(config_file()).get("SHARAR_TARGET")):
        # serving takes the GPU and may start a download of 26-135 GB: only on a model someone named
        sys.exit("sharar serve: name a model: 27b, flashnext, nemotron or bolt (or set SHARAR_TARGET)")
    try:
        s = settings_for(args, traces=args.traces, tp=args.tp)
    except ValueError as exc:
        sys.exit(f"sharar serve: {exc}")
    if args.detach:
        argv = [a for a in sys.argv[1:] if a not in ("--detach", "-d")]
        try:
            preflight.check_host(guard.floor_kib(s.values))
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
        host = preflight.check_host(guard.floor_kib(s.values))
        missing = preflight.ensure_snapshots(s, fetch=not args.no_fetch)
        if missing:
            raise preflight.PreflightError(f"missing from the model cache: {', '.join(missing)}")
        w3 = preflight.ensure_w3(s, fetch=not args.no_fetch)
    except preflight.PreflightError as exc:
        sys.exit(f"sharar serve: {exc}")
    print(preflight.summary(s, host, w3, args.tp), flush=True)
    for d in (s.traces, s.ssd):
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
    if args.tp > 1:
        from sharar.launch import tp
        return tp.serve(s, w3)
    install_env({**s.engine_env(w3), **engine_runtime_env()})
    guard.start_thread(floor=guard.guard_kib(s.values))
    from sharar.cli import engine_parser, run_engine
    parser = engine_parser()
    return run_engine(parser.parse_args(s.engine_args()), parser)


def stop(args) -> int:
    targets = [args.target] if args.target else list(SERVED)
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
    for name in SERVED:
        s = resolve(name)
        if ctl.server_info(str(s.port)) is not None:
            return s
    return None


def status(args) -> int:
    any_up = False
    for name in ([args.target] if args.target else SERVED):
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
        print(f"nothing is serving ({SERVE_HINT})")
    return 0


def cache(args) -> int:
    """`sharar cache`: the serving engine's GPU prefix cache and SSD tier; with nothing serving, the SSD store on disk."""
    s = serving(args.target)
    info = ctl.server_info(str(s.port)) if s is not None else None
    if info is None:
        s = s or resolve("27b")
        directory = s.ssd or Path(s.get("SHARAR_SSD_DIR") or s.results / "ssd" / s.target)
        d = ctl.stored_on_disk(directory)
        who = f"{s.target} is not serving" if args.target else "nothing is serving"
        print(f"{who}; SSD tier {'on' if s.ssd else 'off'} in the settings")
        print(f"  SSD store {directory}: {d['prefixes']:,} prefixes, {d['blocks']:,} blocks, {d['bytes'] / 2**30:.1f} GiB")
        return 0
    cfg, cache = info.get("config") or {}, (info.get("scheduler") or {}).get("cache")
    print(f"{s.target}: serving on {s.url}")
    if cache is None:
        print("  no prefix cache")
        return 0
    print(f"  GPU: {cache['cached_tokens']:,} of {info.get('kv_pages', 0):,} KV tokens cached; "
          f"{cache['hits']:,} hits, {cache['misses']:,} misses, {cache['evictions']:,} evictions")
    if "locks" in cache:  # the cache policy's own counters (cache/radix.py `stats`)
        locks, snaps, drafts = cache["locks"], cache.get("snapshots", {}), cache.get("draft_resumes")
        evicted = sum(n for by_spare in cache.get("evicted", {}).values() for n in by_spare.values())
        print(f"  locks: {locks.get('total', 0)} ({locks.get('request', 0)} requests, {locks.get('restore', 0)} SSD "
              f"restores); snapshots: {snaps.get('not_spare', 0):,} + {snaps.get('spare', 0):,} spare, {evicted:,} "
              f"evicted; {cache.get('pruned', 0):,} dead-KV nodes pruned, {cache.get('full_reprefills', 0):,} full "
              f"re-prefills, {cache.get('branching_resumes', 0):,} resumes at a branching point"
              + (f"; drafter resumes {drafts['hits']:,} hits, {drafts['fallbacks']:,} fallbacks" if drafts else ""))
    ssd = cache.get("ssd")
    if ssd is None:
        print("  SSD: " + ("on in the settings but not running (the server log says why)" if cfg.get("ssd") else "off (SHARAR_SSD=1 turns it on)"))
        return 0
    print(f"  SSD: {ssd['prefixes']:,} prefixes, {ssd['used_gib']:.1f} of {ssd['budget_gib']:.0f} GiB in {cfg.get('ssd_dir')}; "
          f"{ssd['restores']:,} restores ({ssd['restored_tokens']:,} tokens)")
    return 0


def logs(args) -> int:
    log = (serving(args.target) or resolve(args.target)).log_file()
    if not log.is_file():
        sys.exit(f"no log at {log} (logs are kept for detached servers; a foreground server prints to its terminal)")
    return subprocess.call(["tail", "-n", str(args.lines), *(["-F"] if args.follow else []), str(log)])

