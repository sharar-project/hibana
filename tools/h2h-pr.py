#!/usr/bin/env python3
"""Speed of a branch against a base, served, as every speed PR reports it: sharar-eval's h2h on `fast` and
`throughput`, the two checkouts alternated, one Markdown table to paste.

    tools/h2h-pr.py --target flashnext --pr my-branch
    tools/h2h-pr.py --target 27b --pr origin/perf/x --base origin/main --repeats 2 --modes greedy,t07,t10
    tools/h2h-pr.py --target nemotron --pr HEAD --name knobs --arm exact=exact \
        --arm relaxed=exact:SHARAR_ACCEPT_DEFAULT=3:2.0,SHARAR_ACCEPT_DEFAULT_SAMPLED=1 --arm fast=fast

Each arm is a git worktree of its ref under the run directory, served by this checkout's Python environment
(`uv sync`; the two refs must share `uv.lock`) with `sharar serve <target> --fast|--throughput` on --port, the
user's config file, no traces, and the SSD tier on as served but in a fresh directory per server (removed after), so
no arm or repeat restores another's prefixes. Per arm and profile, under the shared GPU lock:

- `fast`: 1 request = h2h `decode` (the `--categories` prompts of `evals/workloads/h2h_v1.json`, one at a time; the
  median of their decode tok/s), and `--fast` stream counts above one as h2h's closed-loop `--streams`;
- `throughput`: the `--throughput` stream counts (default 8, 16, 32, 64, 128) as closed-loop `--streams`: N streams
  each send their next request as one ends, tokens counted over --duration-s after --warmup-s (aggregate tok/s).

Prompts: chat, code, math, prose, structured by default (18 of the 22: multilingual is left out; `--categories` picks
others), in turn by category; numbered per request under `--streams`, so none share a prefix; thinking off, up to
1,024 output tokens with natural stops; --modes picks the sampling (greedy, t07 = T 0.7, t10 = T 1.0, both top-p
0.95, top-k 20), each with the profile's acceptance default.
Order: base then PR, `--repeats` times; the table averages the repeats. Results, logs and the table (report.md) go
to $SHARAR_RESULTS/reviews/<date>-h2h-<name>/; a rerun with the same --name skips what exists. The arms' worktrees
are removed once the report is written.

sharar-eval runs from this environment's `eval` extra (pinned in uv.lock), or from a checkout given with --evals /
SHARAR_EVALS for unreleased sharar-eval changes, as `tools/evaluate.py` does.

**Configurations of one checkout** (`--arm NAME=PROFILE[:KEY=VALUE,...]`, repeatable): instead of base and PR, every
arm serves the --pr ref (one worktree) with `sharar serve <target> --<PROFILE>` and its SHARAR_* settings in the
environment, in the order given, each at the `--fast` request counts (`--throughput` counts for a `throughput` arm).
The table reports each arm against the first: a knob's increment is an arm that adds it to the one before. This is
how a target main does not serve (Nemotron) is measured, and how a profile's knobs are each measured on its own.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import importlib.util
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = Path(os.environ.get("UV_PROJECT_ENVIRONMENT") or REPO / ".venv")
sys.path.insert(0, str(REPO))


def load_gate():
    loader = SourceFileLoader("lossy_gate", str(REPO / "tools" / "lossy-gate.py"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("lossy_gate", loader))
    loader.exec_module(module)
    return module


gate = load_gate()
log, available_gib, health = gate.log, gate.available_gib, gate.health


def git(*args, cwd=REPO) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


@contextlib.contextmanager
def gpu_lock(lock: Path, what: str):
    """The shared GPU lock every Sharar tool takes, held while at least 40 GiB are free (another server has left)."""
    with open(lock, "a") as f:
        log(f"waiting for the GPU lock ({what})")
        fcntl.flock(f, fcntl.LOCK_EX)
        while available_gib() < 40 or health(URL):
            log(f"{available_gib():.0f} GiB available{', a server answers on the port' if health(URL) else ''}; waiting")
            time.sleep(30)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


URL = ""


class Server:
    """`sharar serve <target> --<profile>` from one checkout, in its own process group; stopped with SIGTERM (Ctrl-C)."""

    def __init__(self, tree: Path, target: str, profile: str, port: int, logfile: Path, env: dict, tp: int = 1):
        self.cmd = [str(VENV / "bin" / "python"), "-u", "-m", "sharar", "serve", target, f"--{profile}", "--port", str(port),
                    "--no-fetch", "--tp", str(tp)]
        self.tree, self.logfile, self.env = tree, logfile, env
        self.proc = None

    def __enter__(self):
        with self.logfile.open("a") as f:
            self.proc = subprocess.Popen(self.cmd, cwd=self.tree, env=self.env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1200
        while not health(URL):
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.__exit__()
                raise RuntimeError(f"server did not come up; see {self.logfile}")
            time.sleep(5)
        with urllib.request.urlopen(URL + "/get_server_info", timeout=30) as r:
            self.info = json.load(r)
        return self

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        while health(URL):
            time.sleep(2)


def h2h(ev: dict, engine: str, out: Path, args: list[str], logfile: Path, env: dict) -> None:
    cmd = [ev["python"], "-m", "evals.h2h", "--url", URL, "--engine", engine, "--out", str(out), *args]
    with logfile.open("a") as f:
        r = subprocess.run(cmd, cwd=ev["cwd"], env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"h2h failed ({' '.join(args)}); see {logfile}")


def collect(run: Path) -> dict:
    """(arm, profile, requests, mode) -> one value per run file: the median decode tok/s over the prompts for one
    request, the aggregate tok/s for a stream count."""
    values: dict = {}
    for path in sorted(run.glob("*-*-r*.jsonl")):
        arm, profile, _ = path.stem.split("-")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for mode in {r["mode"] for r in rows}:
            tps = [r["tps"] for r in rows if r["workload"] == "decode" and r["mode"] == mode and r.get("tps")]
            if tps:
                values.setdefault((arm, profile, 1, mode), []).append(statistics.median(tps))
        for r in rows:
            if r["workload"] == "streams":
                values.setdefault((arm, profile, r["streams"], r["mode"]), []).append(r["aggregate_tps"])
    return values


def table(run: Path, plan: dict, modes: list[str]) -> str:
    """base against PR per profile, request count and sampling mode (the mean over the repeats)."""
    values = collect(run)
    mean = lambda key: statistics.mean(values[key]) if values.get(key) else None  # noqa: E731
    lines = ["| profile | requests | sampling | base tok/s | PR tok/s | PR / base |", "|---|---:|---|---:|---:|---:|"]
    for profile, counts in plan.items():
        for n in counts:
            for mode in modes:
                a, b = mean(("base", profile, n, mode)), mean(("pr", profile, n, mode))
                cells = [f"{a:.1f}" if a else "", f"{b:.1f}" if b else "", f"x{b / a:.3f}" if a and b else ""]
                lines.append(f"| {profile} | {n} | {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_arms(items: list[str], ap) -> dict[str, tuple[str, dict]]:
    """--arm NAME=PROFILE[:KEY=VALUE,...] -> {name: (profile, {SHARAR_KEY: value})}, in the order given. A value may
    hold ':' and ',' (3:2.0, SHARAR_SPEC_DEPTHS=2:4,4:4): pairs split only where a ',' precedes the next SHARAR_."""
    import re
    arms: dict[str, tuple[str, dict]] = {}
    for item in items:
        name, _, spec = item.partition("=")
        profile, _, settings = spec.partition(":")
        if not re.fullmatch(r"[A-Za-z0-9_.]+", name) or name in arms:
            ap.error(f"--arm {item!r}: NAME is letters, digits, _ and ., once each")
        if profile not in gate.PROFILE_NAMES:
            ap.error(f"--arm {item!r}: PROFILE must be one of {', '.join(gate.PROFILE_NAMES)}")
        values = dict(kv.split("=", 1) for kv in re.split(r",(?=SHARAR_)", settings) if kv) if settings else {}
        if any(not k.startswith("SHARAR_") for k in values):
            ap.error(f"--arm {item!r}: settings are SHARAR_KEY=VALUE pairs")
        arms[name] = (profile, values)
    return arms


def run_arms(args, arms: dict, target: str, refs: dict, labels: dict, run: Path, tree: Path, env: dict, ev: dict,
             results: Path, plan: dict) -> int:
    """Every --arm configuration of the one worktree in turn, `--repeats` times; the table reports each against the first."""
    (run / "manifest.json").write_text(json.dumps({"target": target, "refs": refs, "labels": labels, "modes": args.modes,
                                                  "arms": {n: {"profile": p, "env": e} for n, (p, e) in arms.items()},
                                                  "fast": plan.get("fast"), "throughput": plan.get("throughput"),
                                                  "repeats": args.repeats, "warmup_s": args.warmup_s, "duration_s": args.duration_s,
                                                  "categories": args.categories, "evals": ev["label"], "evals_commit": ev["commit"]}, indent=1) + "\n")
    evals_env = {**os.environ, "SHARAR_RESULTS": str(results)}
    lock = results / ".gpu.lock"
    counts_of = lambda profile: plan.get("throughput" if profile == "throughput" else "fast", [])  # noqa: E731
    for rep in range(args.repeats):
        for arm, (profile, values) in arms.items():
            stem = f"{arm}-{profile}-r{rep}"
            if (run / f"{stem}.done").exists() or not counts_of(profile):
                continue
            (run / f"{stem}.jsonl").unlink(missing_ok=True)
            ssd = run / f"{stem}.ssd"
            shutil.rmtree(ssd, ignore_errors=True)
            with gpu_lock(lock, f"{arm} r{rep}"), Server(tree, target, profile, args.port, run / f"{stem}.server.log",
                                                        {**env, **values, "SHARAR_SSD_DIR": str(ssd)}) as s:
                (run / f"{stem}.server.json").write_text(json.dumps(s.info, indent=1) + "\n")
                log(f"{arm} ({profile} {values}) r{rep}: serving")
                serve_h2h(args, ev, arm, run / f"{stem}.jsonl", counts_of(profile), run / f"{stem}.h2h.log", evals_env)
            shutil.rmtree(ssd, ignore_errors=True)
            (run / f"{stem}.done").touch()
            log(f"{arm} r{rep}: done")
    values = collect(run)
    mean = lambda key: statistics.mean(values[key]) if values.get(key) else None  # noqa: E731
    first = next(iter(arms))
    lines = ["| arm | profile | settings | requests | sampling | tok/s | vs " + first + " |", "|---|---|---|---:|---|---:|---:|"]
    for arm, (profile, settings) in arms.items():
        for n in counts_of(profile):
            for mode in args.modes.split(","):
                v, b = mean((arm, profile, n, mode)), mean((first, arms[first][0], n, mode))
                shown = ", ".join(f"{k.removeprefix('SHARAR_')}={x}" for k, x in settings.items()) or "(profile)"
                lines.append(f"| {arm} | {profile} | {shown} | {n} | {mode} | {f'{v:.1f}' if v else ''} | "
                             f"{f'x{v / b:.3f}' if v and b else ''} |")
    md = (f"**h2h, {target}, one Spark, configurations of `{labels['pr']}`** ({args.repeats} repeat{'s' if args.repeats > 1 else ''}, "
          f"arms in order). tok/s: 1 request = median decode tok/s over h2h's {args.categories} prompts (one at a time); more = "
          f"aggregate tok/s of that many closed-loop streams over {args.duration_s:.0f} s after {args.warmup_s:.0f} s. Thinking off, "
          f"up to 1,024 tokens, natural stops; greedy / t07 = T 0.7 (top-p 0.95, top-k 20).\n\n" + "\n".join(lines) + "\n")
    (run / "report.md").write_text(md)
    git("worktree", "remove", "--force", str(tree))
    print(md)
    log(f"report: {run / 'report.md'}")
    return 0


def serve_h2h(args, ev: dict, arm: str, out: Path, counts: list[int], hlog: Path, evals_env: dict) -> None:
    """h2h against the server on --port: `decode` for one request, closed-loop `--streams` for more."""
    common = ["--modes", args.modes, "--thinking", "off", "--only", args.categories]
    if 1 in counts:
        h2h(ev, arm, out, ["--workloads", "decode", *common], hlog, evals_env)
    streams = [n for n in counts if n > 1]
    if streams:
        h2h(ev, arm, out, ["--workloads", "conc", "--streams", ",".join(map(str, streams)),
                           "--warmup-s", str(args.warmup_s), "--duration-s", str(args.duration_s), *common], hlog, evals_env)


# h2h_v1's prompt categories measured by default: all but multilingual (--categories adds it back)
CATEGORIES = ("chat", "code", "math", "prose", "structured")


def main() -> int:
    global URL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--target", choices=("flashnext", "27b", "qwen38", "nemotron"), required=True)
    ap.add_argument("--pr", default=None, help="the branch, commit or ref to measure (with --arm: the one served; default HEAD)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PROFILE[:KEY=VALUE,...]",
                    help="a configuration of the --pr checkout to measure instead of base and PR (repeatable; see above)")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--name", help="run directory name (default: the PR ref's short commit)")
    ap.add_argument("--fast", default="1,4", help="request counts under --fast (1: h2h decode; more: closed-loop streams)")
    ap.add_argument("--throughput", default="8,16,32,64,128", help="stream counts under --throughput ('' skips the profile)")
    ap.add_argument("--modes", default="greedy,t07", help="h2h sampling modes: greedy, t07, t10")
    ap.add_argument("--categories", default=",".join(CATEGORIES),
                    help="h2h prompt categories (default: all but multilingual; add it back with ...,multilingual)")
    ap.add_argument("--repeats", type=int, default=1, help="base then PR, this many times; the table averages them")
    ap.add_argument("--warmup-s", type=float, default=30.0)
    ap.add_argument("--duration-s", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--tp", type=int, choices=(1, 2), default=1, help="2: serve both arms on two Sparks")
    ap.add_argument("--evals", type=Path, default=Path(os.environ["SHARAR_EVALS"]) if os.environ.get("SHARAR_EVALS") else None,
                    help="a sharar-eval checkout to run instead of the eval extra (default: SHARAR_EVALS, else the extra)")
    ap.add_argument("--python", default=os.environ.get("SHARAR_EVALS_PYTHON"),
                    help="sharar-eval's Python (default: this environment's, or the checkout's .venv)")
    args = ap.parse_args()
    arms = parse_arms(args.arm, ap)
    if not arms and args.pr is None:
        ap.error("--pr is required (or give --arm configurations of one checkout)")
    if arms:
        args.pr = args.pr or "HEAD"
    launch = gate.launch
    target = launch.target_name(args.target)
    URL = f"http://127.0.0.1:{args.port}"
    ev = gate.sharar_eval(args.evals, args.python)
    usage = subprocess.run([ev["python"], "-m", "evals.h2h", "--help"], cwd=ev["cwd"], capture_output=True, text=True).stdout
    if "--streams" not in usage or "--only" not in usage:
        raise SystemExit(f"sharar-eval ({ev['label']}, {ev['commit'][:12]}): its h2h has no --streams or --only; update it")
    refs = {"pr": git("rev-parse", args.pr)} if arms else {"base": git("rev-parse", args.base), "pr": git("rev-parse", args.pr)}
    if not arms and git("show", f"{refs['base']}:uv.lock") != git("show", f"{refs['pr']}:uv.lock"):
        raise SystemExit("the two refs pin different dependencies (uv.lock); both arms run this checkout's environment")
    conf = launch.config_file()
    env = {**os.environ, "SHARAR_TRACES": "off", **({"SHARAR_CONFIG_FILE": str(conf)} if conf else {})}
    results = launch.resolve(target, "fast", {}, env=env, traces=False).results
    run = results / "reviews" / f"{dt.date.today():%Y-%m-%d}-h2h-{args.name or refs['pr'][:12]}"
    trees = run / "trees"
    trees.mkdir(parents=True, exist_ok=True)
    if args.tp == 2:  # `sharar serve --tp 2` mirrors each arm's checkout to the same path on the peer
        from sharar.launch import tp
        tp.ssh(tp.peer(env), f"mkdir -p {trees}", check=True)
    labels = {"pr": f"{args.pr} {refs['pr'][:9]}"} if arms else {"base": f"{args.base} {refs['base'][:9]}", "pr": f"{args.pr} {refs['pr'][:9]}"}
    for arm, sha in refs.items():
        tree = trees / arm
        if not tree.exists():
            git("worktree", "add", "--detach", str(tree), sha)
        elif git("rev-parse", "HEAD", cwd=tree) != sha:
            raise SystemExit(f"{tree} holds another commit: use a new --name")
    plan = {"fast": [int(v) for v in args.fast.split(",") if v], "throughput": [int(v) for v in args.throughput.split(",") if v]}
    plan = {p: c for p, c in plan.items() if c}
    if arms:
        return run_arms(args, arms, target, refs, labels, run, trees / "pr", env, ev, results, plan)
    (run / "manifest.json").write_text(json.dumps({"target": target, "refs": refs, "labels": labels, "plan": plan, "modes": args.modes,
                                                  "repeats": args.repeats, "warmup_s": args.warmup_s, "duration_s": args.duration_s,
                                                  "categories": args.categories, "evals": ev["label"], "evals_commit": ev["commit"]}, indent=1) + "\n")
    evals_env = {**os.environ, "SHARAR_RESULTS": str(results)}
    lock = results / ".gpu.lock"
    for rep in range(args.repeats):
        for arm in ("base", "pr"):
            for profile, counts in plan.items():
                out = run / f"{arm}-{profile}-r{rep}.jsonl"
                done = run / f"{arm}-{profile}-r{rep}.done"
                if done.exists():
                    continue
                out.unlink(missing_ok=True)
                ssd = run / f"{arm}-{profile}-r{rep}.ssd"
                shutil.rmtree(ssd, ignore_errors=True)
                with gpu_lock(lock, f"{arm} {profile} r{rep}"), Server(trees / arm, target, profile, args.port, run / f"{arm}-{profile}-r{rep}.server.log",
                                                                       {**env, "SHARAR_SSD_DIR": str(ssd)}, args.tp) as s:
                    (run / f"{arm}-{profile}-r{rep}.server.json").write_text(json.dumps(s.info, indent=1) + "\n")
                    log(f"{arm} ({labels[arm]}) {profile} r{rep}: serving")
                    serve_h2h(args, ev, arm, out, counts, run / f"{arm}-{profile}-r{rep}.h2h.log", evals_env)
                shutil.rmtree(ssd, ignore_errors=True)
                done.touch()
                log(f"{arm} {profile} r{rep}: done")
    md = (f"**h2h, {target}, {'two Sparks' if args.tp == 2 else 'one Spark'}** ({args.repeats} repeat{'s' if args.repeats > 1 else ''}, base then PR): "
          f"base `{labels['base']}`, PR `{labels['pr']}`. tok/s: 1 request = median decode tok/s over h2h's {args.categories} "
          f"prompts (one at a time); more = aggregate tok/s of that many closed-loop streams over {args.duration_s:.0f} s after "
          f"{args.warmup_s:.0f} s. Thinking off, up to 1,024 tokens, natural stops; greedy / t07 = T 0.7 (top-p 0.95, top-k 20).\n\n"
          + table(run, plan, args.modes.split(",")) + "\n")
    (run / "report.md").write_text(md)
    for arm in refs:
        git("worktree", "remove", "--force", str(trees / arm))
    print(md)
    log(f"report: {run / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
