#!/usr/bin/env python3
"""The lossy-change gate (#31): one scorecard, paired against a baseline profile, for a lossy speed option.

    tools/lossy-gate.py --target flashnext --name fast --profile fast
    tools/lossy-gate.py --target flashnext --name recal --profile fast --env SHARAR_W3_CHECKPOINT=/path/to/candidate
    tools/lossy-gate.py --target qwen38 --name fewer-experts --profile exact --env SHARAR_X=1 --through full
    tools/lossy-gate.py --target flashnext --name experts3 --profile fast --env SHARAR_W3_CHECKPOINT=/path --battery hard
    tools/lossy-gate.py --target flashnext --name kv-fp4 --profile fast --env SHARAR_X=1 --battery long

A candidate is a serving profile (`--profile`, default exact) plus `--env KEY=VALUE` overrides, resolved for the
target exactly as `sharar serve <target> --<profile>` resolves them with those settings in its environment
(sharar/launch/settings.py: the user's config file, the target's settings, the profile's defaults); the baseline is
`--baseline-profile` (default exact) with `--baseline-env`. Both arms run this checkout's engine from its Python
environment (`uv sync --frozen`; the scorecard records the commit and any uncommitted engine changes). The run is
unattended and resumable: every step writes its own file and a rerun with the same name skips what exists. Do not
wrap the tool in flock: it takes $SHARAR_RESULTS/.gpu.lock itself for each GPU phase (at most --hold-minutes), and
waits while another model server or GPU process runs.

Layers, cheapest first:
  1  Distribution distance (in-process, tools/gate/lossy_gate.py): the baseline's own greedy
     continuations of the 28 frozen prompts of tools/gate/workloads/lossy_v1.json (code, math, prose,
     multilingual, structured, short chat), scored 64 rows per forward (the decode path) as
     argmax agreement and truncated KL(baseline || candidate); the floor is the baseline scored at
     32 rows per forward against itself. Per domain.
  2  Speculation health (same process): tokens per verify round, round cost and decode tok/s of
     each prompt, greedy with the profile's acceptance rule, paired per prompt; the floor is a
     second baseline pass. The speed delta of the verdict is the decode tok/s ratio here.
  3  Task battery (served over HTTP: `sharar serve` in the foreground on --port, and
     sharar-eval's `evals.suite` runs with the server's own acceptance default): paired per item,
     McNemar. Screen: the `quick` suite (GSM8K first 300, HumanEval 164, MGSM 24 per language, plus
     long-context needles at 16K and 64K as a smoke test). Full, when the screen is close: the `full`
     suite (GSM8K 1319, HumanEval 164, MBPP 500, MGSM 2750). `--battery` adds suites after the screen or
     full stage: `hard` (MMLU-Pro, GPQA Diamond, IFEval, WMT24++; 4-6 hours per arm), `long` (RULER v2 at
     32K and 128K, 120 items each; about 3 hours per arm), `all` (both) or `agentic` (Terminal-Bench 2.1 through
     Harbor, the 78 tasks valid on the GB10 twice; about 18 hours per arm, in one GPU window).

Verdict rules (also in CONTRIBUTING.md, "Judge a lossy option"). Intervals are 95%: bootstrap over
64-position blocks (layer 1) or prompts (layer 2), normal approximation of the paired difference
(layer 3); "pooled" adds the discordant pairs of every task of the stage.
  exact     layer 1 within its floor in every domain (excess KL interval reaches 0, argmax
            agreement not below the floor's) and layer-2 tokens per round within its floor
            (interval includes 1); a lossless change. Layer 3 is skipped (--through screen forces it).
  rejected  any of: no measurable speed gain, the speed interval's lower bound at or below 1.035 (a
            lossy option must pay for itself, beyond the layer's 3.5% noise; layer 3 is skipped); a
            task or the pooled battery significantly worse (McNemar p < 0.05, more losses than wins;
            with --battery this includes MMLU-Pro, GPQA, IFEval and each RULER v2 length (paired sign
            test), while WMT24++ chrF++ is reported and decides nothing); pooled accuracy delta below -2.0 points. Pooled is the
            stage's tasks only.
  balanced  not rejected, the full stage ran, pooled delta interval lower bound >= -1.0 point and
            every task's delta >= -1.0 point.
  fast      not rejected otherwise (the loss is not significant and at most 2.0 points).
The screen escalates to the full stage (--through auto) unless it rejects the candidate or its
pooled interval already excludes a loss of more than 1.0 point. Layer 1 decides nothing beyond
`exact`; it locates the damage (which domain) and is the cheap early warning.

Outputs: $SHARAR_RESULTS/reviews/<date>-lossy-gate-<name>/ (manifest.json, arm configs, logs,
scorecard.json, SUMMARY.md); the baseline arm is cached under
$SHARAR_RESULTS/reviews/lossy-gate-baselines/<target>-<profile>-<key>/ and reused by every
candidate with the same baseline, engine, prompts and eval code.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # this checkout's sharar (the launcher resolver is standard-library only)
from sharar.launch import lxcfs, preflight  # noqa: E402
from sharar.launch import settings as launch  # noqa: E402
WORKLOAD = REPO / "tools" / "gate" / "workloads" / "lossy_v1.json"
WORKER = REPO / "tools" / "gate" / "lossy_gate.py"
LOCK: Path | None = None  # $SHARAR_RESULTS/.gpu.lock, set by main()
VENV = Path(os.environ.get("UV_PROJECT_ENVIRONMENT") or REPO / ".venv")  # the project's environment (uv sync)
PROTOCOL = 3  # bump when a step's meaning changes (it keys the baseline cache); 3: arms resolved by sharar.launch
CHUNK, FLOOR_CHUNK = 64, 32
FLOOR_GIB = 16.0  # the memory guard's floor (sharar/launch/guard.py)
BLOCK = 64  # layer-1 bootstrap unit: consecutive continuation positions
PROFILE_NAMES = launch.PROFILE_NAMES
# EngineConfig fields the served arm must report exactly as the in-process arm installed them.
CHECKED = ("target", "profile", "w3_decode", "w3_checkpoint", "stable_numerics", "flashnext_trunk",
           "flashnext_mixers", "flashnext_mtp_trunk", "trunk_keep_bf16", "draft_vocab", "draft_head",
           "draft_head_rescore", "device_accept", "accept_default", "accept_default_sampled",
           "accept_default_sampled_threshold", "qsa_sparse_graphs", "flashnext_experts")

# Layer-3 stages: the sharar-eval suite each runs and the tasks its verdict reads (the suite's GSM8K-300 is the first
# three shards of the full set, so the full stage reuses them).
STAGES = {"screen": ("quick", ["gsm8k", "humaneval", "mgsm-pl24", "needles"]), "full": ("full", ["gsm8k", "humaneval", "mbpp", "mgsm", "needles"]),
          "hard": ("hard", ["mmlupro", "gpqa", "ifeval", "wmt"]), "long": ("long", ["ruler2-32k", "ruler2-128k"]),
          "agentic": ("agentic", ["tbench"])}
BATTERIES = {"quick": [], "hard": ["hard"], "long": ["long"], "agentic": ["agentic"], "all": ["hard", "long"]}
AGENTIC_HOLD_MINUTES = 36 * 60  # one window: a Terminal-Bench task cut off by a window restarts from scratch
BALANCED_MARGIN, FAST_MARGIN = 1.0, 2.0  # accuracy points
# Layer-2 speed resolution (Flash-Next, 2026-09-27): a warm baseline repeat reads x1.035 and a second exact
# process x0.986, so a gain must clear 3.5% to be told from the instrument's own noise.
SPEED_NOISE = 0.035


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- configuration

def engine_identity() -> str:
    """The engine both arms run: this checkout's commit, plus a hash of any uncommitted engine changes."""
    commit = git("rev-parse", "HEAD")
    paths = ["sharar", "pyproject.toml", "constraints"]
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "HEAD", "--", *paths], capture_output=True).stdout
    diff += subprocess.run(["git", "-C", str(REPO), "ls-files", "--others", "--exclude-standard", "--", *paths], capture_output=True).stdout
    return commit + (f"+dirty-{sha256(diff)[:12]}" if diff else "")


def ensure_venv() -> None:
    """The project's Python environment, as uv.lock pins it (`uv sync --frozen`; quick when it is current)."""
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    subprocess.run([uv, "sync", "--frozen", "--quiet"], cwd=REPO, env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(VENV)}, check=True)


def gate_env() -> dict:
    """The arms' environment: this process's, without its SHARAR_* settings (the arms read the config file, then their
    own profile and overrides), with the user's config file named explicitly."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHARAR_")}
    conf = launch.config_file()
    if conf is not None:
        env["SHARAR_CONFIG_FILE"] = str(conf)
    return env


def resolve(target: str, profile: str, overrides: dict) -> launch.Settings:
    """One arm's settings, as `sharar serve <target> --<profile>` resolves them with the overrides in its environment."""
    return launch.resolve(target, profile, overrides, env=gate_env(), traces=False)


def accept_rule(text: str) -> str:
    """A K:M (or off) as `sharar engine` installs it."""
    if text == "off":
        return "1:0.0"
    k, m = text.split(":")
    return f"{int(k)}:{float(m)}"


# Settings the in-process worker leaves out: what only the server uses (batching, sessions, traces, the context it plans).
SERVER_ONLY = ("SHARAR_MAX_BATCH", "SHARAR_SPEC_DEPTHS", "SHARAR_SESSION_DIR", "SHARAR_SESSION_GIB", "SHARAR_SESSION_MIN_TOKENS",
               "SHARAR_TRACES", "SHARAR_TRACES_MAX_MIB", "SHARAR_MAX_CONTEXT", "SHARAR_COLD_TIER_GIB")


def arm_plan(s: launch.Settings) -> dict:
    """What the arm runs: its engine environment and the executor built as `sharar engine` builds it (the gate does not
    fetch 3-bit checkpoints: an arm serves what <target>-current holds)."""
    w3 = preflight.ensure_w3(s, fetch=False)
    cenv = {k: v for k, v in s.engine_env(w3).items() if k not in SERVER_ONLY}
    spec = s.spec
    accept = accept_rule(s.get("SHARAR_ACCEPT_DEFAULT", "2:1.0")) if spec else "1:0.0"
    cenv["SHARAR_SPEC"], cenv["SHARAR_ACCEPT_DEFAULT"] = str(spec), accept
    sampled_on = s.get("SHARAR_ACCEPT_DEFAULT_SAMPLED") == "1" and accept != "1:0.0"
    # sampled requests' own K:M, as `sharar engine` records it: only while they are relaxed (an exact one turns them off)
    threshold = s.get("SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD")
    threshold = accept_rule(threshold) if threshold and sampled_on else ""
    if threshold == "1:0.0":
        sampled_on, threshold = False, ""
    cenv["SHARAR_ACCEPT_DEFAULT_SAMPLED"] = "1" if sampled_on else "0"
    cenv["SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD"] = threshold
    return {"target": s.target, "env": cenv, "proposer": s.proposer, "spec": spec,
            "verify_state": s.get("SHARAR_VERIFY_STATE", "reconstruct"), "draft_precision": s.get("SHARAR_DRAFT_PRECISION", "bf16"),
            "draft": str(s.draft_dir) if s.draft_dir else None, "accept": accept}


def served_command(target: str, profile: str, port: int) -> list[str]:
    """A served arm: `sharar serve` in the foreground, on our port, recording nothing."""
    return [str(VENV / "bin" / "python"), "-u", "-m", "sharar", "serve", target, f"--{profile}", "--port", str(port),
            "--no-traces", "--no-sessions", "--no-fetch"]


def served_env(overrides: dict) -> dict:
    env = {**gate_env(), **overrides}
    env.setdefault("SHARAR_COLD_TIER_GIB", "0")  # host memory for Terminal-Bench's task containers; caching only, same in every stage
    return env


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args, cwd=REPO) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- GPU windows

def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable not found")


def other_gpu_work() -> list[str]:
    """Model servers and GPU processes that are not this tool's: Sharar services, model containers, GPU processes."""
    busy = []
    units = subprocess.run(["systemctl", "--user", "list-units", "--state=active", "--plain", "--no-legend", "sharar-*.service"],
                           capture_output=True, text=True).stdout.split()
    busy += [u for u in units if u.endswith(".service")]
    out = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"], capture_output=True, text=True).stdout if shutil.which("docker") else ""
    rows = [line.split("\t") for line in out.splitlines() if "\t" in line]
    busy += [f"{n} ({i})" for n, i in rows if i.startswith("sharar") or n.startswith("sharar") or "sglang" in i or "vllm" in i]
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    busy += [f"GPU process {line.strip()}" for line in out.splitlines() if line.strip()]
    return busy


@contextlib.contextmanager
def gpu_window(what: str, lock: Path | None = None):
    """Hold the shared GPU lock for one phase; start only when no other model server or GPU process runs."""
    lock = lock or LOCK
    while True:
        f = open(lock, "a")
        log(f"waiting for the GPU lock ({what})")
        fcntl.flock(f, fcntl.LOCK_EX)
        busy = other_gpu_work()
        if not busy and available_gib() >= 40:
            break
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
        log(f"GPU busy ({', '.join(busy) or f'{available_gib():.0f} GiB available'}); retry in 2 minutes")
        time.sleep(120)
    log(f"GPU lock held: {what}")
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
        log(f"GPU lock released: {what}")


def watch(proc: subprocess.Popen, deadline: float, trace, label: str, stop) -> int:
    """Wait for proc under the host memory floor and a deadline; stop() on breach."""
    while proc.poll() is None:
        avail = available_gib()
        if trace:
            trace.write(json.dumps({"phase": label, "t": round(time.time(), 1), "available_gib": round(avail, 2)}) + "\n")
            trace.flush()
        if avail < FLOOR_GIB or time.monotonic() > deadline:
            log(f"{label}: {'memory floor' if avail < FLOOR_GIB else 'time limit'} reached; stopping")
            stop()
            proc.wait(timeout=120)
            return -1
        time.sleep(2)
    return proc.returncode


def tool_command(plan: dict, results: Path, hf_cache: str, out: Path, base: Path | None) -> tuple[list[str], dict]:
    """The in-process worker (tools/gate/lossy_gate.py) on this checkout's engine, and its environment."""
    env = {"PATH": f"{VENV}/bin:/usr/local/cuda/bin:/usr/bin:/bin", "HOME": os.environ["HOME"], "CUDA_HOME": "/usr/local/cuda",
           "PYTHONPATH": f"{REPO / 'tools'}:{REPO}", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "SHARAR_HF_CACHE": hf_cache, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SHARAR_RESULTS": str(results), **plan["env"]}
    cmd = [str(VENV / "bin" / "python"), "-m", "gate.lossy_gate", "--prompts", str(WORKLOAD),
           "--out", str(out), "--accept", plan["accept"], "--proposer", plan["proposer"], "--spec", str(plan["spec"]),
           "--verify-state", plan["verify_state"], "--draft-precision", plan["draft_precision"], "--chunk", str(CHUNK)]
    if plan["draft"]:
        cmd += ["--draft", plan["draft"]]
    if base is None:
        cmd += ["--passes", "2", "--floor-chunk", str(FLOOR_CHUNK)]
    else:
        cmd += ["--base", str(base)]
    return lxcfs.wrap(cmd), env


def run_inprocess(label: str, command: tuple[list[str], dict], out: Path, done: Path, minutes: float) -> None:
    cmd, env = command
    if done.exists():
        log(f"{label}: in-process layers already done ({done})")
        return
    out.mkdir(parents=True, exist_ok=True)
    with gpu_window(f"{label} in-process layers 1-2"):
        if done.exists():  # another gate run finished this (shared baseline) while we waited
            log(f"{label}: in-process layers already done ({done})")
            return
        (out / "command.json").write_text(json.dumps({"command": cmd, "env": env}, indent=1) + "\n")
        with (out / "worker.log").open("a") as logf, (out / "memory.jsonl").open("a") as trace:
            proc = subprocess.Popen(cmd, env=env, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            rc = watch(proc, time.monotonic() + 60 * minutes, trace, label, lambda: os.killpg(proc.pid, signal.SIGTERM))
    if rc != 0 or not done.exists():
        raise SystemExit(f"{label}: in-process worker failed (exit {rc}); see {out / 'worker.log'}")
    log(f"{label}: in-process layers done")


# ---------------------------------------------------------------- served layer 3

def health(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


class Served:
    """One served arm: `sharar serve <target> --<profile>` in the foreground with the arm's overrides, its configuration
    checked against the in-process arm, eval shards run against it, then stopped (SIGTERM, as Ctrl-C)."""

    def __init__(self, label, arm_dir: Path, target: str, profile: str, overrides: dict, port: int, python: str, evals: Path,
                 expect: dict):
        self.label, self.dir, self.port, self.python, self.evals, self.expect = label, arm_dir, port, python, evals, expect
        self.url = f"http://127.0.0.1:{port}"
        self.command, self.env = served_command(target, profile, port), served_env(overrides)
        self.proc: subprocess.Popen | None = None
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "served.json").write_text(json.dumps({"command": self.command, "overrides": overrides}, indent=1) + "\n")

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        with (self.dir / "server.log").open("a") as logf:
            self.proc = subprocess.Popen(self.command, env=self.env, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1200
        while not health(self.url):
            if not self.running() or time.monotonic() > deadline or available_gib() < FLOOR_GIB:
                self.stop()
                raise RuntimeError(f"{self.label}: server did not come up; see {self.dir / 'server.log'}")
            time.sleep(5)
        info = get_json(f"{self.url}/get_server_info")
        (self.dir / "server_info.json").write_text(json.dumps(info, indent=1) + "\n")
        cfg = info.get("config", {})
        bad = {k: (cfg.get(k), v) for k, v in self.expect["config"].items() if cfg.get(k) != v}
        spec = info.get("speculative") or {}
        if spec.get("num_draft_tokens") != self.expect["spec"] or spec.get("draft_precision") != self.expect["draft_precision"]:
            bad["speculative"] = (spec, (self.expect["spec"], self.expect["draft_precision"]))
        if info.get("w3_target_identity") != self.expect["w3_identity"]:
            bad["w3_target_identity"] = (info.get("w3_target_identity"), self.expect["w3_identity"])
        if bad:
            self.stop()
            raise RuntimeError(f"{self.label}: served configuration differs from the in-process arm: {bad}")
        log(f"{self.label}: serving on {self.url}, configuration matches the in-process arm")

    def stop(self) -> None:
        if self.running():
            os.killpg(self.proc.pid, signal.SIGTERM)  # the server drains, as on Ctrl-C
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()


def run_suite(label: str, served: Served, suite: str, l3: Path, skip: list[str], minutes: float, engine: str) -> None:
    """Run one sharar-eval suite on a served arm in GPU windows of at most `minutes` each; the suite keeps
    100-item shards, so a window that ends mid-suite loses at most one shard and the next resumes."""
    done = l3 / f".{suite}{''.join('-no-' + t for t in skip)}.done"
    while not done.exists():
        with gpu_window(f"{label} {suite} suite"):
            if done.exists():  # another gate run filled the shared baseline meanwhile
                return
            before = len(list(l3.glob("*/s*.json")))
            served.start()
            try:
                cmd = [served.python, "-m", "evals.suite", "run", "--url", served.url, "--label", label, "--suite", suite,
                       "--out", str(l3), "--engine", engine, *[a for t in skip for a in ("--skip", t)]]
                log(f"{label}: {suite} suite")
                with (served.dir / "evals.log").open("a") as logf:
                    proc = subprocess.Popen(cmd, cwd=served.evals, stdout=logf, stderr=subprocess.STDOUT)
                    rc = watch(proc, time.monotonic() + 60 * minutes, None, f"{label} {suite}", proc.kill)
            finally:
                served.stop()
            if rc == 0:
                done.write_text("")
            elif rc != -1 or len(list(l3.glob("*/s*.json"))) == before:
                raise RuntimeError(f"{label}: {suite} suite failed (exit {rc}); see {served.dir / 'evals.log'}")


def suite_compare(python: str, evals: Path, base: Path, cand: Path, tasks: list[str]) -> dict:
    """sharar-eval's paired scorecard over `tasks` (the needles count only when both arms ran them)."""
    out = cand / f"compare-{'-'.join(tasks)}.json"
    r = subprocess.run([python, "-m", "evals.suite", "compare", str(base), str(cand), "--tasks", ",".join(tasks),
                        "--json", str(out)], cwd=evals, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"evals.suite compare failed: {r.stderr.strip()[-500:]}")
    return json.loads(out.read_text())


# ---------------------------------------------------------------- statistics

def ci_mean(values: list[float], groups: list | None = None, resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and a 95% bootstrap interval, resampling whole groups (blocks or prompts) when given."""
    if not values:
        return (math.nan, math.nan, math.nan)
    if groups is None:
        groups = list(range(len(values)))
    sums: dict = {}
    for v, g in zip(values, groups):
        s = sums.setdefault(g, [0.0, 0])
        s[0] += v
        s[1] += 1
    units = list(sums.values())
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        tot = cnt = 0.0
        for _ in units:
            s = units[rng.randrange(len(units))]
            tot += s[0]
            cnt += s[1]
        means.append(tot / cnt)
    means.sort()
    return (sum(values) / len(values), means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1])


def layer1(dist: dict, floor: dict) -> dict:
    """Per domain and pooled: candidate and floor KL and argmax agreement over the same positions, and the excess with intervals."""
    fl = {s["id"]: s for s in floor["sequences"]}
    per: dict[str, dict] = {}
    for s in dist["sequences"]:
        f = fl[s["id"]]
        assert f["n"] == s["n"], s["id"]
        for dom in (s["domain"], "all"):
            d = per.setdefault(dom, {"kl": [], "kl_f": [], "agree": [], "agree_f": [], "g": []})
            d["kl"] += s["kl"]; d["kl_f"] += f["kl"]; d["agree"] += s["agree"]; d["agree_f"] += f["agree"]
            d["g"] += [f"{s['id']}/{i // BLOCK}" for i in range(s["n"])]
    out = {}
    for dom, d in per.items():
        n = len(d["kl"])
        excess = ci_mean([a - b for a, b in zip(d["kl"], d["kl_f"])], d["g"])
        dagree = ci_mean([a - b for a, b in zip(d["agree"], d["agree_f"])], d["g"])
        # within the floor: neither measure shows the candidate worse than rounding noise
        out[dom] = {"n": n, "kl": sum(d["kl"]) / n, "kl_floor": sum(d["kl_f"]) / n, "kl_excess": excess,
                    "argmax": sum(d["agree"]) / n, "argmax_floor": sum(d["agree_f"]) / n, "argmax_delta": dagree,
                    "within_floor": excess[1] <= 0 and dagree[2] >= 0}
    return out


def paired_log(base: list[dict], cand: list[dict], key: str, domain: str | None = None) -> dict:
    logs = [math.log(c[key] / b[key]) for b, c in zip(base, cand)
            if b.get(key) and c.get(key) and (domain is None or b["domain"] == domain)]
    if not logs:
        return {"n": 0}
    m, lo, hi = ci_mean(logs)
    return {"n": len(logs), "ratio": math.exp(m), "ci95": [math.exp(lo), math.exp(hi)]}


def layer2(base_spec: dict, cand_spec: dict) -> dict:
    a, b = base_spec["passes"][0], base_spec["passes"][1] if len(base_spec["passes"]) > 1 else None
    c = cand_spec["passes"][0]
    assert [r["id"] for r in a] == [r["id"] for r in c]
    out = {k: paired_log(a, c, k) for k in ("decode_tps", "tok_per_round", "round_ms")}
    out["domains"] = {dom: paired_log(a, c, "tok_per_round", dom) for dom in sorted({r["domain"] for r in a})}
    out["floor"] = {k: paired_log(a, b, k) for k in ("decode_tps", "tok_per_round", "round_ms")} if b else None
    out["relaxed_share"] = sum(r["relaxed"] for r in c) / max(1, sum(r["output_tokens"] for r in c))
    out["base_tok_per_round"] = sum(r["output_tokens"] - 1 for r in a) / max(1, sum(r["rounds"] for r in a))
    out["cand_tok_per_round"] = sum(r["output_tokens"] - 1 for r in c) / max(1, sum(r["rounds"] for r in c))
    t = out["tok_per_round"]
    out["within_floor"] = bool(t.get("n")) and t["ci95"][0] <= 1 <= t["ci95"][1]
    return out


# ---------------------------------------------------------------- verdict

def binary(card: dict | None) -> dict:
    """The pass/fail tasks of a sharar-eval scorecard."""
    return {t: r for t, r in (card or {}).get("tasks", {}).items() if r["kind"] == "binary"}


def deciding(card: dict | None) -> dict:
    """The tasks of an added battery that can reject: pass/fail tasks and RULER v2 (WMT24++ chrF++ and the needles are reported only)."""
    return {t: r for t, r in (card or {}).get("tasks", {}).items() if r["kind"] == "binary" or t.startswith("ruler2")}


def verdict(l1: dict | None, l2: dict | None, l3: dict | None, stage: str | None, extra: dict | None = None) -> tuple[str, list[str]]:
    """The profile the candidate can enter, and why (the rules in the module docstring)."""
    why = []
    l1_within = l1 is not None and all(d["within_floor"] for k, d in l1.items() if k != "all")
    l2_within = l2 is not None and l2["within_floor"]
    if l1_within and l2_within and (l3 is None or l3["pooled"]["p"] >= 0.05):
        return "exact", ["layers 1 and 2 within their floors" + ("" if l3 is None else ", layer 3 not different")]
    speed = l2["decode_tps"] if l2 else None
    if speed and speed.get("n") and speed["ci95"][0] <= 1 + SPEED_NOISE:
        return "rejected", [f"no measurable speed gain: decode x{speed['ratio']:.3f} [{speed['ci95'][0]:.3f}, {speed['ci95'][1]:.3f}], "
                            f"lower bound not above {1 + SPEED_NOISE:.3f}"]
    if l3 is None:
        return "undecided", ["layer 3 has not run"]
    tasks = {**binary(l3), **{t: r for card in (extra or {}).values() for t, r in deciding(card).items()}}
    bad = [f"{t} {r['delta']:+.1f} {'pp' if r['kind'] == 'binary' else 'points'} (p {r['p']:.3f})" for t, r in tasks.items()
           if r["p"] < 0.05 and r["losses"] > r["wins"]]
    pooled = l3["pooled"]
    if pooled["p"] < 0.05 and pooled["losses"] > pooled["wins"]:
        bad.append(f"pooled {pooled['delta']:+.1f} pp (p {pooled['p']:.3f})")
    if pooled["delta"] < -FAST_MARGIN:
        bad.append(f"pooled loss {pooled['delta']:+.1f} pp exceeds {FAST_MARGIN} pp")
    if bad:
        return "rejected", ["significantly worse: " + "; ".join(bad)]
    if stage == "full" and pooled["ci95"][0] >= -BALANCED_MARGIN and all(r["delta"] >= -BALANCED_MARGIN for r in binary(l3).values()):
        return "balanced", [f"full battery non-inferior at {BALANCED_MARGIN} pp: pooled {pooled['delta']:+.1f} pp {fmt_ci(pooled['ci95'])}"]
    why.append(f"loss not significant: pooled {pooled['delta']:+.1f} pp {fmt_ci(pooled['ci95'])}, p {pooled['p']:.2f}")
    if stage != "full":
        why.append("balanced needs the full stage")
    return "fast", why


def screen_is_close(l3: dict) -> bool:
    """Escalate to the full sets unless the screen rejects or already excludes a loss beyond the balanced margin."""
    v, _ = verdict(None, None, l3, "screen")
    return v != "rejected" and l3["pooled"]["ci95"][0] < -BALANCED_MARGIN


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"


def one_line(name: str, target: str, v: str, l1, l2, l3, stage, extra=None) -> str:
    parts = [f"VERDICT {name} ({target}): {v}"]
    if l2:
        s = l2["decode_tps"]
        parts.append(f"speed x{s['ratio']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
    if l1:
        worst = max((k for k in l1 if k != "all"), key=lambda k: l1[k]["kl_excess"][0])
        a = l1["all"]
        parts.append(f"L1 KL {a['kl']:.4f} vs floor {a['kl_floor']:.4f} (excess {a['kl_excess'][0]:+.4f} [{a['kl_excess'][1]:+.4f}, {a['kl_excess'][2]:+.4f}]), "
                     f"argmax {100 * a['argmax']:.1f}% (floor {100 * a['argmax_floor']:.1f}%), worst {worst} {l1[worst]['kl_excess'][0]:+.4f}")
    if l2:
        t = l2["tok_per_round"]
        parts.append(f"L2 tok/round x{t['ratio']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]")
    if l3:
        p = l3["pooled"]
        parts.append(f"L3 {stage} {p['delta']:+.1f} pp {fmt_ci(p['ci95'])} ({p['wins']}/{p['losses']}, p {p['p']:.2f})")
    for name, card in (extra or {}).items():
        parts.append(f"{name} " + ", ".join(f"{t} {r['delta']:+.{2 if r['kind'] == 'score' else 1}f} (p {r['p']:.2f})"
                                            for t, r in card["tasks"].items() if r["kind"] != "needles"))
    return " | ".join(parts)


def summary_md(card: dict) -> str:
    m, l1, l2, l3 = card["manifest"], card.get("layer1"), card.get("layer2"), card.get("layer3")
    lines = [f"# Lossy-change gate: {m['name']} ({m['target']})", "",
             f"`{card['verdict_line']}`", "",
             f"Candidate: profile `{m['candidate']['profile']}` + `{m['candidate']['env']}`; baseline: profile `{m['baseline']['profile']}` + `{m['baseline']['env']}`.",
             f"Engine `{m['engine']}`, engine source sha256 {card.get('engine_source_sha256', '?')[:12]}, "
             f"gate source {m['source']}{' (dirty)' if m['dirty'] else ''}, evals {m['evals_commit'][:10]}. Batch size 1 (layers 1-2), 4 streams (layer 3), greedy, "
             f"prompts `{m['workload']}` (sha256 {m['workload_sha256'][:12]}). Baseline arm: `{m['baseline_dir']}`.", "",
             "Why: " + "; ".join(card["why"]), ""]
    if l1:
        lines += ["## Layer 1: distribution distance (decode path, 64 rows per forward; floor: baseline at 32 rows)", "",
                  "| domain | positions | KL | floor KL | excess KL [95%] | argmax | floor argmax | within floor |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for dom, d in sorted(l1.items(), key=lambda kv: kv[0] == "all"):
            e = d["kl_excess"]
            lines.append(f"| {dom} | {d['n']} | {d['kl']:.4f} | {d['kl_floor']:.4f} | {e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}] | "
                         f"{100 * d['argmax']:.2f}% | {100 * d['argmax_floor']:.2f}% | {'yes' if d['within_floor'] else 'no'} |")
        lines.append("")
    if l2:
        lines += ["## Layer 2: speculation health (in-process, batch 1, paired per prompt)", "",
                  "| measure | candidate / baseline [95%] | floor (baseline repeat) |", "|---|---:|---:|"]
        for k in ("decode_tps", "tok_per_round", "round_ms"):
            r, f = l2[k], (l2["floor"] or {}).get(k, {})
            lines.append(f"| {k} | x{r['ratio']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | "
                         + (f"x{f['ratio']:.4f} [{f['ci95'][0]:.4f}, {f['ci95'][1]:.4f}]" if f.get("n") else "-") + " |")
        lines += ["", f"Tokens per round {l2['base_tok_per_round']:.3f} -> {l2['cand_tok_per_round']:.3f}; relaxed acceptances "
                      f"{100 * l2['relaxed_share']:.1f}% of emitted tokens. By domain: "
                  + ", ".join(f"{k} x{v['ratio']:.3f}" for k, v in l2["domains"].items() if v.get("n")), ""]
    titles = {"hard": "Hard battery (served, 4 streams, paired; WMT24++ chrF++ decides nothing)",
              "agentic": "Agentic battery: Terminal-Bench 2.1 (Terminus-2, 2 tasks at a time, two attempts, paired McNemar)",
              "long": "Long battery: RULER v2 (served, 4 streams, paired bootstrap of the 0-100 item score)"}
    for title, sc in [(f"Layer 3: task battery ({card['stage']}, served, 4 streams, paired, McNemar)", l3)] + \
                     [(titles[k], v) for k, v in (card.get("extra") or {}).items()]:
        if not sc:
            continue
        lines += [f"## {title}", "", "| task | n | baseline | candidate | delta [95%] | better/worse | p | tokens/item | capped |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t, r in sc["tasks"].items():
            if r["kind"] == "needles":
                continue
            pct = r["kind"] == "binary"
            val = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.2f}")
            cap = f"{r['base_side']['capped']} / {r['cand_side']['capped']}" if r["base_side"]["capped"] is not None else ""
            lines.append(f"| {t} | {r['n']} | {val(r['base'])} | {val(r['cand'])} | {r['delta']:+.{1 if pct else 2}f} {fmt_ci(r['ci95'])} | "
                         f"{r['wins']}/{r['losses']} | {r['p']:.3f} | {r['base_side']['tokens_per_item']:.0f} / {r['cand_side']['tokens_per_item']:.0f} | {cap} |")
        if "pooled" in sc and sc is l3:
            r = sc["pooled"]
            lines.append(f"| pooled | {r['n']} | {100 * r['base']:.1f}% | {100 * r['cand']:.1f}% | {r['delta']:+.1f} {fmt_ci(r['ci95'])} | "
                         f"{r['wins']}/{r['losses']} | {r['p']:.3f} | | |")
        for t in ("mgsm", "mgsm-pl24"):
            if (sc["tasks"].get(t) or {}).get("groups"):
                lines += ["", f"{t} by language (delta pp): " + ", ".join(f"{g} {100 * (c - b):+.0f}" for g, (b, c) in sc["tasks"][t]["groups"].items())]
        nd = sc["tasks"].get("needles")
        if nd:
            for side in ("base", "cand"):
                lines.append(f"Needles ({'baseline' if side == 'base' else 'candidate'}): {nd[side]['correct']}/{nd[side]['n']} correct; decode tok/s {nd[side]['decode_tps']}")
        lines.append("")
    lines += ["Rules: `tools/lossy-gate.py` docstring and CONTRIBUTING.md (Judge a lossy option).", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- driver

def parse_env(items: list[str]) -> dict:
    out = {}
    for kv in items:
        k, sep, v = kv.partition("=")
        if not sep or not k.startswith("SHARAR_"):
            raise SystemExit(f"--env takes SHARAR_KEY=VALUE, got {kv!r}")
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("flashnext", "qwen38"), required=True)
    ap.add_argument("--name", required=True, help="candidate name (the output directory: <date>-lossy-gate-<name>)")
    ap.add_argument("--profile", choices=PROFILE_NAMES, default="exact", help="the candidate's serving profile")
    ap.add_argument("--env", action="append", default=[], help="SHARAR_KEY=VALUE override of the candidate (repeatable)")
    ap.add_argument("--baseline-profile", choices=PROFILE_NAMES, default="exact")
    ap.add_argument("--baseline-env", action="append", default=[])
    ap.add_argument("--through", choices=("inprocess", "screen", "full", "auto"), default="auto",
                    help="inprocess: layers 1-2; screen: also the layer-3 screen (forced); full: also the full sets; auto: the cascade")
    ap.add_argument("--port", type=int, default=30100, help="served arms' port (30100-30199)")
    ap.add_argument("--battery", choices=sorted(BATTERIES), default="quick",
                    help="quick: the screen, then the full sets when close; hard / long / all: also sharar-eval's hard suite "
                         "(4-6 h per arm), long suite (RULER v2, ~3 h per arm) or both; agentic: Terminal-Bench 2.1 (~18 h per arm)")
    ap.add_argument("--evals", type=Path, default=Path(os.environ.get("SHARAR_EVALS") or REPO.parent / "sharar-eval"),
                    help="sharar-eval checkout (default: SHARAR_EVALS, else beside this one)")
    ap.add_argument("--python", default=os.environ.get("SHARAR_EVALS_PYTHON"),
                    help="host Python for sharar-eval (default: its .venv)")
    ap.add_argument("--hold-minutes", type=float, default=45, help="longest GPU hold per phase")
    ap.add_argument("--no-needles", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rebuild the scorecard from what exists")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved arms and exit")
    args = ap.parse_args(argv)
    if not 30100 <= args.port <= 30199:
        ap.error("--port must be in 30100-30199")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.name):
        ap.error("--name: letters, digits, . _ -")

    conf_path = launch.config_file()
    cand_over, base_over = parse_env(args.env), parse_env(args.baseline_env)
    try:
        arms = {"baseline": resolve(args.target, args.baseline_profile, base_over), "candidate": resolve(args.target, args.profile, cand_over)}
    except ValueError as exc:
        raise SystemExit(f"lossy-gate: {exc}")
    plans = {k: arm_plan(v) for k, v in arms.items()}
    results, hf_cache = arms["baseline"].results, str(arms["baseline"].hf_cache)
    global LOCK
    LOCK = results / ".gpu.lock"  # the shared GPU lock every Sharar tool takes
    engine = engine_identity()
    if not (args.evals / "evals" / "suite.py").is_file():
        raise SystemExit(f"{args.evals}: no sharar-eval with evals.suite; set --evals or SHARAR_EVALS")
    args.python = args.python or str(args.evals / ".venv" / "bin" / "python")
    if not Path(args.python).is_file():
        raise SystemExit(f"{args.python} not found: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt in {args.evals}")
    evals_commit = git("rev-parse", "HEAD", cwd=args.evals)
    workload_sha = sha256(WORKLOAD.read_bytes())
    key_src = {"protocol": PROTOCOL, "target": args.target, "baseline": {"values": arms["baseline"].values, "plan": plans["baseline"]},
               "engine": engine, "workload": workload_sha, "worker": sha256(WORKER.read_bytes()),
               "evals": git("rev-parse", "HEAD:evals", cwd=args.evals)}
    key = sha256(json.dumps(key_src, sort_keys=True).encode())[:12]
    base_dir = results / "reviews" / "lossy-gate-baselines" / f"{args.target}-{args.baseline_profile}-{key}"
    spec = {"target": args.target, "candidate": {"profile": args.profile, "env": cand_over}, "baseline": {"profile": args.baseline_profile, "env": base_over},
            "engine": engine, "baseline_key": key}
    existing = sorted((results / "reviews").glob(f"*-lossy-gate-{args.name}"))
    run_dir = None
    for d in reversed(existing):
        m = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
        if {k: m.get(k) for k in spec} == spec:
            run_dir = d
            break
    if run_dir is None:
        if existing:
            raise SystemExit(f"{existing[-1]} holds another candidate spec; choose another --name")
        run_dir = results / "reviews" / f"{dt.date.today().isoformat()}-lossy-gate-{args.name}"
    manifest = {**spec, "name": args.name, "config_file": str(conf_path) if conf_path else None, "through": args.through, "battery": args.battery,
                "source": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "tools")),
                "evals": str(args.evals), "evals_commit": evals_commit, "workload": json.loads(WORKLOAD.read_text())["version"],
                "workload_sha256": workload_sha, "baseline_dir": str(base_dir), "port": args.port,
                "arms": {k: {"settings": v.values, "plan": plans[k]} for k, v in arms.items()}}
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), **manifest}, indent=1))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    started = json.loads((run_dir / "manifest.json").read_text()).get("started") if (run_dir / "manifest.json").exists() else None
    manifest["started"] = started or dt.datetime.now().isoformat(timespec="seconds")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if not (base_dir / "key.json").exists():
        (base_dir / "key.json").write_text(json.dumps(key_src, indent=1) + "\n")
    log(f"candidate {args.name}: {run_dir}")
    log(f"baseline ({args.baseline_profile}): {base_dir}")

    # layers 1-2, in process: baseline once (cached), then the candidate
    b_in, c_in = base_dir / "inprocess", run_dir / "inprocess"
    if not args.report:
        ensure_venv()
        run_inprocess("baseline", tool_command(plans["baseline"], results, hf_cache, b_in, None), b_in, b_in / "floor.json", args.hold_minutes)
        run_inprocess("candidate", tool_command(plans["candidate"], results, hf_cache, c_in, b_in), c_in, c_in / "dist.json", args.hold_minutes)
    card = {"manifest": manifest, "stage": None}
    l1 = l2 = l3 = None
    if (c_in / "dist.json").exists() and (b_in / "floor.json").exists():
        l1 = layer1(json.loads((c_in / "dist.json").read_text()), json.loads((b_in / "floor.json").read_text()))
        l2 = layer2(json.loads((b_in / "spec.json").read_text()), json.loads((c_in / "spec.json").read_text()))
        card["engine_source_sha256"] = json.loads((c_in / "arm.json").read_text())["engine_source_sha256"]
    card["layer1"], card["layer2"] = l1, l2

    # layer 3, served
    stage = None
    v0, _ = verdict(l1, l2, None, None)
    want = args.through in ("screen", "full") or (args.through == "auto" and v0 not in ("exact", "rejected"))
    if want and l1 is not None:
        arm_info = {k: json.loads(((b_in if k == "baseline" else c_in) / "arm.json").read_text()) for k in plans}
        servers = {}
        for side in plans:
            info = arm_info[side]
            expect = {"config": {k: info["config"][k] for k in CHECKED}, "spec": plans[side]["spec"],
                      "draft_precision": plans[side]["draft_precision"], "w3_identity": info["w3_identity"]}
            profile, over = (args.baseline_profile, base_over) if side == "baseline" else (args.profile, cand_over)
            servers[side] = Served(side, (base_dir if side == "baseline" else run_dir) / "served", args.target, profile, over,
                                   args.port, args.python, args.evals, expect)
        dirs = {"baseline": base_dir / "l3", "candidate": run_dir / "l3"}

        def battery(stage_name: str) -> dict:
            suite, tasks = STAGES[stage_name]
            for side in plans:
                if not args.report:
                    run_suite(side, servers[side], suite, dirs[side], ["needles"] if args.no_needles else [],
                              AGENTIC_HOLD_MINUTES if stage_name == "agentic" else args.hold_minutes, engine)
            card = suite_compare(args.python, args.evals, dirs["baseline"], dirs["candidate"], tasks)
            return card if card["tasks"] else None  # --report before the arms ran

        stage, l3 = "screen", battery("screen")
        if l3 and (args.through == "full" or (args.through == "auto" and screen_is_close(l3))):
            stage, l3 = "full", battery("full")
        card["extra"] = {name: c for name in BATTERIES[args.battery] if (c := battery(name))}
    card["layer3"], card["stage"] = l3, stage
    v, why = verdict(l1, l2, l3, stage, card.get("extra"))
    card["verdict"], card["why"] = v, why
    card["verdict_line"] = one_line(args.name, args.target, v, l1, l2, l3, stage, card.get("extra"))
    card["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (run_dir / "scorecard.json").write_text(json.dumps(card, indent=1) + "\n")
    (run_dir / "SUMMARY.md").write_text(summary_md(card))
    print(card["verdict_line"], flush=True)
    log(f"scorecard: {run_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
