#!/usr/bin/env python3
"""Evaluate one Sharar build with sharar-eval: serve it, run quality suites and/or the speed benchmark, record both versions.

    tools/evaluate.py --target flashnext --profile fast --suite quick
    tools/evaluate.py --target flashnext --profile fast --suite hard --speed
    tools/evaluate.py --target qwen38 --profile exact --tasks gpqa,mmlupro --limit 50 --name gpqa50
    tools/evaluate.py --target flashnext --profile fast --suite quick --baseline $SHARAR_RESULTS/reviews/<date>-evaluate-<name>

- **What is served.** `sharar serve <target> --<profile>` with the `--env SHARAR_KEY=VALUE` overrides in its
  environment (and the user's config file), from this checkout's engine and Python environment (`uv sync --frozen`).
  The run records the commit; uncommitted engine changes are refused unless --allow-dirty (and then recorded as a
  hash). To evaluate another revision, run this tool from a worktree of it.
- **What runs.** sharar-eval (`--evals`, default ../sharar-eval, with its .venv) against the server:
  - `--suite quick|full|hard|long|agentic|all`, or `--tasks a,b` with `--limit N` (`evals.suite`);
  - `--speed` adds the AIPerf benchmark (`evals.speed`, `--workload code|design|reasoning|all`).
    Speed requests carry no acceptance fields, so both measure the served profile as a client sees it.
- **Where.** One GPU window under $SHARAR_RESULTS/.gpu.lock, on --port (30200-30299), the server in the foreground
  of this tool; it stops afterwards. Output goes to
  $SHARAR_RESULTS/reviews/<date>-evaluate-<name>/:
  - manifest.json: the engine (commit, uncommitted changes), sharar-eval commit, config;
  - served/: the serve command, /get_server_info, the server's log;
  - suite/: the sharar-eval run directory;
  - speed/: the AIPerf run;
  - SUMMARY.md.
- **Resume.** A rerun with the same --name resumes: the suite from its shards, and speed is skipped once complete.
- **`--baseline DIR`.** An earlier evaluate output directory. The suite is paired with it item by item
  (`evals.suite compare`), and speed reuses its frozen inputs and reports the TPS change.

This reuses the lossy gate's served-arm machinery (tools/lossy-gate.py: settings resolution,
GPU window, memory-floor watch, suite runner) without its in-process configuration check: there is
no in-process arm to compare with, so /get_server_info is recorded instead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_gate():
    loader = SourceFileLoader("lossy_gate", str(REPO / "tools" / "lossy-gate.py"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("lossy_gate", loader))
    loader.exec_module(module)
    return module


gate = load_gate()
log = gate.log


class Served(gate.Served):
    """A served build: `sharar serve` as a user would run it (3-bit checkpoint fetched), recorded rather than checked
    against an in-process arm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command = [a for a in self.command if a != "--no-fetch"]

    def start(self) -> None:
        with (self.dir / "server.log").open("a") as logf:
            self.proc = subprocess.Popen(self.command, env=self.env, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1800
        while not gate.health(self.url):
            if not self.running() or time.monotonic() > deadline or gate.available_gib() < gate.FLOOR_GIB:
                self.stop()
                raise RuntimeError(f"server did not come up; see {self.dir / 'server.log'}")
            time.sleep(5)
        info = gate.get_json(f"{self.url}/get_server_info")
        (self.dir / "server_info.json").write_text(json.dumps(info, indent=1) + "\n")
        log(f"serving on {self.url}: {info.get('target')} profile {(info.get('config') or {}).get('profile')}")


def evals_cmd(python: str, *args: str) -> list[str]:
    return [python, "-m", *args]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("flashnext", "qwen38"), required=True)
    ap.add_argument("--profile", choices=gate.PROFILE_NAMES, default="fast")
    ap.add_argument("--env", action="append", default=[], help="SHARAR_KEY=VALUE override (repeatable)")
    ap.add_argument("--allow-dirty", action="store_true", help="evaluate with uncommitted engine changes (recorded as a hash)")
    ap.add_argument("--suite", choices=("quick", "full", "hard", "long", "agentic", "all"), default=None)
    ap.add_argument("--tasks", default=None, help="comma-separated sharar-eval tasks instead of (or within) --suite")
    ap.add_argument("--limit", type=int, default=0, help="the first N items of each task")
    ap.add_argument("--skip", action="append", default=[], help="leave a task out (repeatable)")
    ap.add_argument("--speed", action="store_true", help="also run the AIPerf speed benchmark (evals.speed)")
    ap.add_argument("--workload", default="code", help="evals.speed workload: code, design, reasoning or all")
    ap.add_argument("--name", default=None, help="output name (<date>-evaluate-<name>; default: target-profile)")
    ap.add_argument("--baseline", type=Path, default=None, help="an earlier evaluate output directory to pair with")
    ap.add_argument("--port", type=int, default=30200, help="30200-30299")
    ap.add_argument("--evals", type=Path, default=Path(os.environ.get("SHARAR_EVALS") or REPO.parent / "sharar-eval"))
    ap.add_argument("--python", default=os.environ.get("SHARAR_EVALS_PYTHON"), help="sharar-eval's Python (default: its .venv)")
    ap.add_argument("--hold-hours", type=float, default=48, help="longest GPU window")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved run and exit")
    args = ap.parse_args(argv)
    if not (args.suite or args.tasks or args.speed):
        ap.error("nothing to run: give --suite, --tasks and/or --speed")
    if not 30200 <= args.port <= 30299:
        ap.error("--port must be in 30200-30299")
    name = args.name or f"{args.target}-{args.profile}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        ap.error("--name: letters, digits, . _ -")
    if not (args.evals / "evals" / "suite.py").is_file():
        raise SystemExit(f"{args.evals}: no sharar-eval checkout; clone abdelfattah-lab/sharar-eval there or set SHARAR_EVALS")
    python = args.python or str(args.evals / ".venv" / "bin" / "python")
    if not Path(python).is_file():
        raise SystemExit(f"{python} not found: run `uv sync` in {args.evals}")
    if args.baseline and not (args.baseline / "manifest.json").is_file():
        raise SystemExit(f"{args.baseline}: not an evaluate output directory")

    conf_path = gate.launch.config_file()
    overrides = gate.parse_env(args.env)
    try:
        settings = gate.resolve(args.target, args.profile, overrides)
    except ValueError as exc:
        raise SystemExit(f"evaluate: {exc}")
    results = settings.results
    gate.LOCK = results / ".gpu.lock"
    env = gate.evals_env(results, settings.hf_cache)  # sharar-eval's datasets and tokenizer come from this configuration
    engine = gate.engine_identity()
    if "+dirty" in engine and not args.allow_dirty:
        raise SystemExit("the engine has uncommitted changes; commit them or pass --allow-dirty")
    spec = {"target": args.target, "profile": args.profile, "env": overrides, "engine": engine}
    existing = sorted((results / "reviews").glob(f"*-evaluate-{name}"))
    run_dir = None
    for d in reversed(existing):
        m = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
        if {k: m.get(k) for k in spec} == spec:
            run_dir = d
            break
    if run_dir is None:
        if existing:
            raise SystemExit(f"{existing[-1]} holds another configuration; choose another --name")
        run_dir = results / "reviews" / f"{dt.date.today().isoformat()}-evaluate-{name}"
    suite_args = [*(["--suite", args.suite] if args.suite else []), *(["--tasks", args.tasks] if args.tasks else []),
                  *(["--limit", str(args.limit)] if args.limit else []), *[a for t in args.skip for a in ("--skip", t)]]
    manifest = {**spec, "name": name, "config_file": str(conf_path) if conf_path else None, "port": args.port,
                "launcher_source": gate.git("rev-parse", "HEAD"), "dirty": bool(gate.git("status", "--porcelain", "--", "sharar", "tools")),
                "evals": str(args.evals), "evals_commit": gate.git("rev-parse", "HEAD", cwd=args.evals),
                "suite": suite_args, "speed": args.workload if args.speed else None,
                "baseline": str(args.baseline) if args.baseline else None}
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "engine": engine, **manifest}, indent=1))
        return 0
    gate.ensure_venv()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"{name}: {run_dir}")

    served = Served("evaluate", run_dir / "served", args.target, args.profile, overrides, args.port, python, args.evals, {})
    speed_dir = run_dir / "speed"
    speed_done = (speed_dir / "summary.json").is_file() and json.loads((speed_dir / "summary.json").read_text()).get("completed")
    with gate.gpu_window(f"evaluate {name}"):
        served.start()
        try:
            with (served.dir / "evals.log").open("a") as logf:
                if suite_args:
                    cmd = evals_cmd(python, "evals.suite", "run", "--url", served.url, "--label", name, "--out", str(run_dir / "suite"),
                                    "--engine", engine, *suite_args)
                    log(f"evals.suite {' '.join(suite_args)}")
                    proc = subprocess.Popen(cmd, cwd=args.evals, stdout=logf, stderr=subprocess.STDOUT, env=env)
                    if gate.watch(proc, time.monotonic() + 3600 * args.hold_hours, None, "suite", proc.kill) != 0:
                        raise RuntimeError(f"evals.suite failed; see {served.dir / 'evals.log'} (rerun with --name {name} to resume)")
                if args.speed and not speed_done:
                    if speed_dir.exists():
                        speed_dir.rename(run_dir / f"speed-incomplete-{int(time.time())}")
                    cmd = evals_cmd(python, "evals.speed", "--url", served.url, "--workload", args.workload, "--accept", "omit",
                                    "--output", str(speed_dir))
                    if args.baseline and (args.baseline / "speed" / "summary.json").is_file():
                        cmd += ["--inputs", str(args.baseline / "speed" / "frozen-inputs.json"), "--baseline", str(args.baseline / "speed")]
                    log(f"evals.speed --workload {args.workload}")
                    proc = subprocess.Popen(cmd, cwd=args.evals, stdout=logf, stderr=subprocess.STDOUT, env=env)
                    if gate.watch(proc, time.monotonic() + 3600 * args.hold_hours, None, "speed", proc.kill) != 0:
                        raise RuntimeError(f"evals.speed failed; see {served.dir / 'evals.log'}")
        finally:
            served.stop()

    lines = [f"# Evaluate {name}", "", f"Engine `{engine}`; profile `{args.profile}` + `{overrides}`; "
             f"sharar-eval {manifest['evals_commit'][:10]}; config `{conf_path.name if conf_path else 'none'}`.", ""]
    if suite_args:
        if args.baseline:
            out = subprocess.run(evals_cmd(python, "evals.suite", "compare", str(args.baseline / "suite"), str(run_dir / "suite"),
                                           "--json", str(run_dir / "compare.json")), cwd=args.evals, capture_output=True, text=True, env=env)
        else:
            out = subprocess.run(evals_cmd(python, "evals.suite", "show", str(run_dir / "suite"), "--json", str(run_dir / "scores.json")),
                                 cwd=args.evals, capture_output=True, text=True, env=env)
        lines += ["## Quality (sharar-eval `evals.suite`)", "", out.stdout.strip() or out.stderr.strip(), ""]
    if args.speed:
        summary = json.loads((speed_dir / "summary.json").read_text())
        lines += ["## Speed (sharar-eval `evals.speed`, AIPerf, cold prefix, concurrency 1)", "",
                  "| workload | end-to-end TPS | decode TPS | TTFT s | latency s | vs baseline |", "|---|---:|---:|---:|---:|---|"]
        for case, row in summary["cases"].items():
            change = (summary.get("comparison") or {}).get(case)
            vs = f"decode {change['decode_tps_change_pct']:+.2f}%, end-to-end {change['output_tps_change_pct']:+.2f}%" if change else ""
            lines.append(f"| {case} | {row['output_tps']:.2f} | {row['decode_tps']:.2f} | {row['ttft_s']:.3f} | {row['latency_s']:.3f} | {vs} |")
        lines.append("")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    log(f"summary: {run_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
