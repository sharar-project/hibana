#!/usr/bin/env python3
"""The lossy-change gate (#31): one scorecard, paired against a baseline profile, for a lossy speed option.

    tools/lossy-gate.py --target flashnext --name fast --profile fast
    tools/lossy-gate.py --target flashnext --name w3near --env SHARAR_FLASHNEXT_TRUNK=w3
    tools/lossy-gate.py --target qwen38 --name fewer-experts --profile exact --env SHARAR_X=1 --through full

A candidate is a serving profile (`--profile`, default exact) plus `--env KEY=VALUE` overrides on
the target's launcher config (config/flashnext.env, or config/gb10.env for the 27B), resolved the
way run-sharar.sh resolves them; the baseline is `--baseline-profile` (default exact) with
`--baseline-env`. Both run the same image (`--image`, default the config's SHARAR_IMAGE). The run is
unattended and resumable: every step writes its own file and a rerun with the same name skips what
exists. Do not wrap the tool in flock: it takes /home/yke/sharar-results/.gpu.lock itself for each
GPU phase (at most --hold-minutes), and waits while another model container runs.

Layers, cheapest first:
  1  Distribution distance (in-process, bench/lossy_gate.py): the baseline's own greedy
     continuations of the 28 frozen prompts of bench/workloads/lossy_v1.json (code, math, prose,
     multilingual, structured, short chat), scored 64 rows per forward (the decode path) as
     argmax agreement and truncated KL(baseline || candidate); the floor is the baseline scored at
     32 rows per forward against itself. Per domain.
  2  Speculation health (same process): tokens per verify round, round cost and decode tok/s of
     each prompt, greedy with the profile's acceptance rule, paired per prompt; the floor is a
     second baseline pass. The speed delta of the verdict is the decode tok/s ratio here.
  3  Task battery (served over HTTP: run-sharar.sh starts container sharar-lossy-gate on --port,
     sharar-evals' accuracy_eval with the server's own acceptance default): paired per item,
     McNemar. Screen: GSM8K first 300, HumanEval 164, MGSM 24 per language (264), plus long-context
     needles at 16K and 64K (smoke). Full, when the screen is close: GSM8K 1319, HumanEval 164,
     MBPP 500, MGSM 2750.

Verdict rules (also in CONTRIBUTING.md, "Judge a lossy option"). Intervals are 95%: bootstrap over
64-position blocks (layer 1) or prompts (layer 2), normal approximation of the paired difference
(layer 3); "pooled" adds the discordant pairs of every task of the stage.
  exact     layer 1 within its floor in every domain (excess KL interval reaches 0, argmax
            agreement not below the floor's) and layer-2 tokens per round within its floor
            (interval includes 1); a lossless change. Layer 3 is skipped (--through screen forces it).
  rejected  any of: speed interval upper bound <= 1.00 (a lossy option must pay for itself; layer 3
            is skipped); a task or the pooled battery significantly worse (McNemar p < 0.05, more
            losses than wins); pooled accuracy delta below -2.0 points.
  balanced  not rejected, the full stage ran, pooled delta interval lower bound >= -1.0 point and
            every task's delta >= -1.0 point.
  fast      not rejected otherwise (the loss is not significant and at most 2.0 points).
The screen escalates to the full stage (--through auto) unless it rejects the candidate or its
pooled interval already excludes a loss of more than 1.0 point. Layer 1 decides nothing beyond
`exact`; it locates the damage (which domain) and is the cheap early warning.

Outputs: $SHARAR_RESULTS/reviews/<date>-lossy-gate-<name>/ (manifest.json, arm configs, logs,
scorecard.json, SUMMARY.md); the baseline arm is cached under
$SHARAR_RESULTS/reviews/lossy-gate-baselines/<target>-<profile>-<key>/ and reused by every
candidate with the same baseline, image, prompts and eval code.
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
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKLOAD = REPO / "bench" / "workloads" / "lossy_v1.json"
WORKER = REPO / "bench" / "lossy_gate.py"
LOCK = Path("/home/yke/sharar-results/.gpu.lock")
CONTAINER = "sharar-lossy-gate"
TOOL_CONTAINER = "sharar-lossy-gate-tool"
PROTOCOL = 1  # bump when a step's meaning changes (it keys the baseline cache)
CHUNK, FLOOR_CHUNK = 64, 32
FLOOR_GIB = 16.0  # tools/memory-guard.sh floor
BLOCK = 64  # layer-1 bootstrap unit: consecutive continuation positions

# run-sharar.sh's profile table (tests/unit/test_lossy_gate.py keeps them equal)
PROFILES = {
    "exact": {"SHARAR_ACCEPT_DEFAULT": "off", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "0", "SHARAR_W3_DECODE": "0",
              "SHARAR_W3_DRAFT": "0", "SHARAR_FLASHNEXT_W3_CHECKPOINT": "off"},
    "balanced": {"SHARAR_ACCEPT_DEFAULT": "2:1.0", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "0", "SHARAR_W3_DECODE": "1",
                 "SHARAR_W3_DRAFT": "1", "SHARAR_FLASHNEXT_W3_CHECKPOINT": "off"},
    "fast": {"SHARAR_ACCEPT_DEFAULT": "3:2.0", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "1", "SHARAR_W3_DECODE": "1",
             "SHARAR_W3_DRAFT": "1", "SHARAR_FLASHNEXT_W3_CHECKPOINT": "auto"},
}
# Settings run-sharar.sh forwards into the container unchanged (when set).
FORWARDED = ("SHARAR_DEVICE_ACCEPT", "SHARAR_DRAFT_VOCAB", "SHARAR_DRAFT_HEAD", "SHARAR_QSA_SPARSE_GRAPHS", "SHARAR_COLD_TIER_GIB")
# Settings only the launcher reads (paths, container, serve flags); every other --env override also goes
# into the in-process container, and the served arm's configuration check catches one the launcher drops.
LAUNCHER_ONLY = {"SHARAR_IMAGE", "SHARAR_CONTAINER", "SHARAR_PORT", "SHARAR_PROPOSER", "SHARAR_SPEC", "SHARAR_VERIFY_STATE",
                 "SHARAR_DRAFT_PRECISION", "SHARAR_DRAFT", "SHARAR_DRAFT_RUNS", "SHARAR_TRACES", "SHARAR_TRACES_MAX_MIB",
                 "SHARAR_MAX_CONTEXT", "SHARAR_HF_CACHE", "SHARAR_RESULTS", "SHARAR_FLASHNEXT_CACHE", "SHARAR_SESSION_DIR",
                 "SHARAR_SESSION_GIB", "SHARAR_SESSION_MIN_TOKENS", "SHARAR_MIN_AVAILABLE_KIB", "SHARAR_REFERENCE_CONTAINER",
                 "SHARAR_REFERENCE_URL", "SHARAR_CONFIG_FILE"}
# EngineConfig fields the served arm must report exactly as the in-process arm installed them.
CHECKED = ("target", "profile", "w3_decode", "w3_head", "w3_draft", "w3_checkpoint", "stable_numerics", "flashnext_trunk",
           "flashnext_mixers", "flashnext_w3_checkpoint", "flashnext_mtp_trunk", "trunk_keep_bf16", "draft_vocab", "draft_head",
           "draft_head_rescore", "device_accept", "accept_default", "accept_default_sampled", "qsa_sparse_graphs")

SCREEN = [("gsm8k", {"limit": 300}), ("humaneval", {}), ("mgsm", {"per_language": 24})]
FULL = [("gsm8k", {}), ("humaneval", {}), ("mbpp", {}), ("mgsm", {})]
TASK_SIZE = {"gsm8k": 1319, "humaneval": 164, "mbpp": 500, "mgsm": 2750}
SHARD = 100  # the screen's GSM8K-300 is the first three shards of the full set
SECONDS_PER_ITEM = {"gsm8k": 3.0, "humaneval": 2.5, "mbpp": 1.5, "mgsm": 3.5}  # first-shard guesses (Flash-Next, 4 streams)
NEEDLES = ["--lengths", "16384,65536", "--depths", "0.9,0.5,0.1", "--output-tokens", "64"]
BALANCED_MARGIN, FAST_MARGIN = 1.0, 2.0  # accuracy points


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- configuration

def load_env(path: Path) -> dict:
    """KEY=VALUE lines of a launcher config file, in order (later lines win, as `source` does)."""
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def target_config(target: str) -> Path:
    if target == "flashnext":
        return REPO / "config" / "flashnext.env"
    own = REPO / "config" / "gb10.env"
    return Path(os.environ.get("SHARAR_CONFIG_FILE") or (own if own.is_file() else "/home/yke/sharar/config/gb10.env"))


def resolve(config: dict, profile: str, overrides: dict) -> dict:
    """The launcher's view of one arm: config file, then overrides, then the profile's default for
    every knob still unset (run-sharar.sh: `${SHARAR_X:-$P_X}`)."""
    env = {**config, **overrides, "SHARAR_PROFILE": profile, "_OVERRIDES": sorted(overrides)}
    for k, v in PROFILES[profile].items():
        if not env.get(k):
            env[k] = v
    results = env.get("SHARAR_RESULTS", "/home/yke/sharar-results")
    if env["SHARAR_FLASHNEXT_W3_CHECKPOINT"] == "auto":
        current = Path(results) / "w3-checkpoints" / "flashnext-current"
        env["SHARAR_FLASHNEXT_W3_CHECKPOINT"] = str(current.resolve()) if (current / "manifest.json").is_file() else "off"
    if env["SHARAR_W3_DECODE"] != "1":
        env["SHARAR_W3_CHECKPOINT"] = ""
    draft = env.get("SHARAR_DRAFT", "")
    if draft == "pinned":
        draft = ""
    elif not draft:
        current = Path(env.get("SHARAR_DRAFT_RUNS") or Path(results) / "draft-runs") / "current"
        if current.is_symlink() and (current.resolve() / "model.safetensors").is_file():
            draft = str(current.resolve())
    env["_DRAFT_DIR"] = draft
    return env


def accept_rule(text: str) -> str:
    """run-sharar.sh's K:M (or off) as `sharar serve` installs it."""
    if text == "off":
        return "1:0.0"
    k, m = text.split(":")
    return f"{int(k)}:{float(m)}"


def arm_plan(env: dict) -> dict:
    """What the arm runs: the container environment, mounts and the executor built as `sharar serve` builds it."""
    target = env.get("SHARAR_TARGET") or "qwen38"
    proposer = env.get("SHARAR_PROPOSER", "dflash2")
    spec = int(env.get("SHARAR_SPEC", "15"))
    precision = env.get("SHARAR_DRAFT_PRECISION", "fp8")
    accept = accept_rule(env["SHARAR_ACCEPT_DEFAULT"]) if spec else "1:0.0"
    cenv = {"SHARAR_PROFILE": env["SHARAR_PROFILE"], "SHARAR_SPEC": str(spec), "SHARAR_ACCEPT_DEFAULT": accept,
            "SHARAR_ACCEPT_DEFAULT_SAMPLED": "1" if env["SHARAR_ACCEPT_DEFAULT_SAMPLED"] == "1" and accept != "1:0.0" else "0",
            "SHARAR_W3_DECODE": env["SHARAR_W3_DECODE"], "SHARAR_W3_HEAD": env.get("SHARAR_W3_HEAD") or "0",
            "SHARAR_W3_DRAFT": env["SHARAR_W3_DRAFT"]}
    mounts = {}
    if env.get("SHARAR_W3_CHECKPOINT"):
        mounts["/w3"] = str(Path(env["SHARAR_W3_CHECKPOINT"]).resolve())
        cenv["SHARAR_W3_CHECKPOINT"] = "/w3"
    if target == "flashnext":
        cenv.update(SHARAR_TARGET="flashnext", SHARAR_FLASHNEXT_CACHE="/cache/flash-next",
                    SHARAR_FLASHNEXT_TRUNK=env.get("SHARAR_FLASHNEXT_TRUNK") or "fp8")
        mounts["/cache/flash-next"] = env["SHARAR_FLASHNEXT_CACHE"]
        if env["SHARAR_FLASHNEXT_W3_CHECKPOINT"] != "off":
            mounts["/fn-w3"] = env["SHARAR_FLASHNEXT_W3_CHECKPOINT"]
            cenv["SHARAR_FLASHNEXT_W3_CHECKPOINT"] = "/fn-w3"
    elif env.get("SHARAR_TARGET"):
        cenv["SHARAR_TARGET"] = env["SHARAR_TARGET"]
    for k in FORWARDED:
        if env.get(k):
            cenv[k] = env[k]
    for k in env["_OVERRIDES"]:
        if k not in cenv and k not in LAUNCHER_ONLY and k not in mounts.values():
            cenv[k] = env[k]
    draft = None
    if env["_DRAFT_DIR"] and proposer in ("dflash2", "dspark") and spec:
        mounts["/draft"] = env["_DRAFT_DIR"]
        draft = "/draft"
    return {"target": target, "env": cenv, "mounts": mounts, "proposer": proposer, "spec": spec,
            "verify_state": env.get("SHARAR_VERIFY_STATE", "reconstruct"), "draft_precision": precision, "draft": draft,
            "accept": accept}


def served_config(base: dict, profile: str, overrides: dict, image: str, port: int) -> str:
    """The launcher config file of a served arm: the target's file, the arm's profile and overrides (the launcher
    resolves them itself; the configuration check compares the result with the in-process arm), our container and port."""
    lines = [f"{k}={v}" for k, v in base.items()]
    lines += [f"SHARAR_PROFILE={profile}"] + [f"{k}={v}" for k, v in overrides.items()]
    lines += [f"SHARAR_IMAGE={image}", f"SHARAR_CONTAINER={CONTAINER}", f"SHARAR_PORT={port}", "SHARAR_SESSION_DIR=off", "SHARAR_TRACES="]
    return "# written by tools/lossy-gate.py\n" + "\n".join(lines) + "\n"


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


def other_model_containers() -> list[str]:
    out = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"], capture_output=True, text=True).stdout
    rows = [line.split("\t") for line in out.splitlines() if "\t" in line]
    return [f"{n} ({i})" for n, i in rows if (i.startswith("sharar") or n.startswith("sharar") or "sglang" in i) and n not in (CONTAINER, TOOL_CONTAINER)]


@contextlib.contextmanager
def gpu_window(what: str, lock: Path = LOCK):
    """Hold the shared GPU lock for one phase; start only when no other model container runs."""
    while True:
        f = open(lock, "a")
        log(f"waiting for the GPU lock ({what})")
        fcntl.flock(f, fcntl.LOCK_EX)
        busy = other_model_containers()
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
        for name in (CONTAINER, TOOL_CONTAINER):
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
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


def tool_container(image: str, plan: dict, results: Path, hf_cache: str, out: Path, base: Path | None) -> list[str]:
    cmd = ["docker", "run", "--rm", "--name", TOOL_CONTAINER, "--network", "none", "--pull", "never",
           "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
           "--pids-limit", "4096", "--shm-size", "16g", "--workdir", "/gate"]
    lib = "/usr/lib/aarch64-linux-gnu"
    for dev in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/dri/renderD128"):
        cmd += ["--device", dev]
    for so, dst in (("libcuda.so.1", "libcuda.so.1"), ("libcuda.so.1", "libcuda.so"), ("libnvidia-ptxjitcompiler.so.1", "libnvidia-ptxjitcompiler.so.1"),
                    ("libnvidia-ptxjitcompiler.so.1", "libnvidia-ptxjitcompiler.so"), ("libnvidia-ml.so.1", "libnvidia-ml.so.1"), ("libnvidia-ml.so.1", "libnvidia-ml.so"),
                    ("libnvidia-nvvm.so.4", "libnvidia-nvvm.so.4"), ("libnvidia-nvvm.so.4", "libnvidia-nvvm.so"), ("libcudadebugger.so.1", "libcudadebugger.so.1")):
        cmd += ["--mount", f"type=bind,src={os.path.realpath(f'{lib}/{so}')},dst={lib}/{dst},readonly"]
    for gpucomp in Path(lib).glob("libnvidia-gpucomp.so.*"):
        cmd += ["--mount", f"type=bind,src={gpucomp},dst={gpucomp},readonly"]
    cmd += ["--mount", "type=bind,src=/usr/bin/nvidia-smi,dst=/usr/bin/nvidia-smi,readonly",
            "--mount", f"type=bind,src={hf_cache},dst=/cache/huggingface,readonly",
            "--mount", f"type=bind,src={results}/runtime-state,dst=/state",
            # only the gate's bench package over the image: the engine is the image's own code
            "--mount", f"type=bind,src={REPO / 'bench'},dst=/gate/bench,readonly",
            "--mount", f"type=bind,src={out},dst=/out"]
    if base is not None:
        cmd += ["--mount", f"type=bind,src={base},dst=/base,readonly"]
    for dst, src in plan["mounts"].items():
        cmd += ["--mount", f"type=bind,src={src},dst={dst},readonly"]
    env = {"PYTHONPATH": "/gate:/workspace/sharar", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "HOME": "/state",
           "SHARAR_HF_CACHE": "/cache/huggingface", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "TRITON_LIBCUDA_PATH": lib, "SHARAR_RESULTS": "/state", **plan["env"]}
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    cmd += ["--entrypoint", "python3", image, "-m", "bench.lossy_gate", "--prompts", "/gate/bench/workloads/lossy_v1.json",
            "--out", "/out", "--accept", plan["accept"], "--proposer", plan["proposer"], "--spec", str(plan["spec"]),
            "--verify-state", plan["verify_state"], "--draft-precision", plan["draft_precision"], "--chunk", str(CHUNK)]
    if plan["draft"]:
        cmd += ["--draft", plan["draft"]]
    if base is None:
        cmd += ["--passes", "2", "--floor-chunk", str(FLOOR_CHUNK)]
    else:
        cmd += ["--base", "/base"]
    return cmd


def run_inprocess(label: str, cmd: list[str], out: Path, done: Path, minutes: float) -> None:
    if done.exists():
        log(f"{label}: in-process layers already done ({done})")
        return
    out.mkdir(parents=True, exist_ok=True)
    with gpu_window(f"{label} in-process layers 1-2"):
        (out / "command.json").write_text(json.dumps(cmd, indent=1) + "\n")
        with (out / "worker.log").open("a") as logf, (out / "memory.jsonl").open("a") as trace:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            rc = watch(proc, time.monotonic() + 60 * minutes, trace, label,
                       lambda: subprocess.run(["docker", "stop", "--timeout", "10", TOOL_CONTAINER], capture_output=True))
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


def shards(stage: list) -> list[dict]:
    out = []
    for task, opt in stage:
        if task == "mgsm" and opt.get("per_language"):
            out.append({"task": task, "start": 0, "limit": 0, "per_language": opt["per_language"], "file": f"mgsm-pl{opt['per_language']}"})
            continue
        total = opt.get("limit") or TASK_SIZE[task]
        for s in range(0, total, SHARD):
            out.append({"task": task, "start": s, "limit": min(SHARD, total - s), "per_language": 0, "file": f"{task}-s{s:05d}"})
    return out


def shard_items(sh: dict) -> int:
    return sh["per_language"] * 11 if sh["per_language"] else sh["limit"]


def shard_path(l3: Path, sh: dict) -> Path:
    return l3 / f"{sh['file']}-n{shard_items(sh)}.json"


class Served:
    """One served arm: start with run-sharar.sh, check the installed configuration, run eval shards, stop."""

    def __init__(self, label, arm_dir: Path, conf_text: str, port: int, python: str, evals: Path, expect: dict):
        self.label, self.dir, self.port, self.python, self.evals, self.expect = label, arm_dir, port, python, evals, expect
        self.url = f"http://127.0.0.1:{port}"
        self.conf = arm_dir / "served.env"
        arm_dir.mkdir(parents=True, exist_ok=True)
        self.conf.write_text(conf_text)
        self.env = {**os.environ, "SHARAR_CONFIG_FILE": str(self.conf)}

    def start(self) -> None:
        subprocess.run([str(REPO / "run-sharar.sh"), "--stop"], env=self.env, capture_output=True)
        r = subprocess.run([str(REPO / "run-sharar.sh")], env=self.env, capture_output=True, text=True, cwd=REPO)
        (self.dir / "launch.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f"{self.label}: run-sharar.sh failed: {(r.stdout + r.stderr).strip().splitlines()[-1:]}")
        deadline = time.monotonic() + 1200
        while not health(self.url):
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER], capture_output=True, text=True).stdout.strip()
            if state != "true" or time.monotonic() > deadline or available_gib() < FLOOR_GIB:
                raise RuntimeError(f"{self.label}: server did not come up (running={state})")
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
            raise RuntimeError(f"{self.label}: served configuration differs from the in-process arm: {bad}")
        log(f"{self.label}: serving on {self.url}, configuration matches the in-process arm")

    def stop(self) -> None:
        with (self.dir / "server.log").open("a") as f:
            subprocess.run(["docker", "logs", CONTAINER], stdout=f, stderr=subprocess.STDOUT)
        subprocess.run([str(REPO / "run-sharar.sh"), "--stop"], env=self.env, capture_output=True)

    def eval_cmd(self, sh: dict, out: Path) -> list[str]:
        cmd = [self.python, "-m", "evals.accuracy_eval", "--url", self.url, "--task", sh["task"], "--label", sh["file"],
               "--accept", "server", "--output", str(out)]
        if sh["per_language"]:
            cmd += ["--per-language", str(sh["per_language"])]
        else:
            cmd += ["--start", str(sh["start"]), "--limit", str(sh["limit"])]
        return cmd


def run_served(label: str, served: Served, todo: list[dict], l3: Path, needles: bool, minutes: float, rates: dict) -> None:
    """Run the missing shards (and the needle smoke) in GPU windows of at most `minutes` each."""
    l3.mkdir(parents=True, exist_ok=True)
    needle_file = l3 / "needles.jsonl"
    while True:
        missing = [sh for sh in todo if not shard_path(l3, sh).exists()]
        want_needles = needles and not (l3 / "needles.done").exists()
        if not missing and not want_needles:
            return
        with gpu_window(f"{label} layer 3 ({len(missing)} shards{', needles' if want_needles else ''})"):
            started = time.monotonic()
            served.start()
            try:
                with (served.dir / "evals.log").open("a") as logf:
                    for sh in missing:
                        est = shard_items(sh) * rates.get(sh["task"], SECONDS_PER_ITEM[sh["task"]])
                        used = time.monotonic() - started
                        if used > 0 and used + est > 60 * minutes and sh is not missing[0]:
                            break
                        log(f"{label}: {sh['file']} ({shard_items(sh)} items, ~{est / 60:.0f} min)")
                        tmp = l3 / "partial"
                        t0 = time.monotonic()
                        proc = subprocess.Popen(served.eval_cmd(sh, tmp), cwd=served.evals, stdout=logf, stderr=subprocess.STDOUT)
                        rc = watch(proc, started + 60 * (minutes + 15), None, sh["file"], proc.kill)
                        if rc != 0:
                            raise RuntimeError(f"{label}: {sh['file']} failed (exit {rc}); see {served.dir / 'evals.log'}")
                        rates[sh["task"]] = (time.monotonic() - t0) / shard_items(sh)
                        (tmp / f"{sh['file']}-{sh['task']}.json").rename(shard_path(l3, sh))
                    else:
                        if want_needles and time.monotonic() - started < 60 * minutes:
                            log(f"{label}: long-context needles")
                            needle_file.unlink(missing_ok=True)
                            proc = subprocess.Popen([served.python, "-m", "evals.long_context", "--url", served.url, *NEEDLES, "--log", str(needle_file)],
                                                    cwd=served.evals, stdout=logf, stderr=subprocess.STDOUT)
                            if watch(proc, started + 60 * (minutes + 15), None, "needles", proc.kill) != 0:
                                raise RuntimeError(f"{label}: needles failed; see {served.dir / 'evals.log'}")
                            (l3 / "needles.done").write_text("")
            finally:
                served.stop()


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


def mcnemar_p(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(wins - n / 2)) / 2 ** n)


def paired_acc(base: list[bool], cand: list[bool]) -> dict:
    n = len(base)
    wins = sum(1 for x, y in zip(base, cand) if y and not x)
    losses = sum(1 for x, y in zip(base, cand) if x and not y)
    diff = (wins - losses) / n
    se = math.sqrt(max(0.0, (wins + losses) / n - diff ** 2) / n)
    return {"n": n, "base": sum(base) / n, "cand": sum(cand) / n, "delta_pp": 100 * diff,
            "ci95_pp": [100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se)], "wins": wins, "losses": losses, "p": mcnemar_p(wins, losses)}


def load_items(l3: Path, stage: list) -> dict[str, list[dict]]:
    out = {}
    for sh in shards(stage):
        out.setdefault(sh["task"], []).extend(json.loads(shard_path(l3, sh).read_text())["items"])
    return out


def layer3(base_l3: Path, cand_l3: Path, stage: list) -> dict:
    b, c = load_items(base_l3, stage), load_items(cand_l3, stage)
    out, pb, pc = {"tasks": {}}, [], []
    for task in b:
        xb, xc = [x["correct"] for x in b[task]], [x["correct"] for x in c[task]]
        assert len(xb) == len(xc), task
        out["tasks"][task] = paired_acc(xb, xc)
        if task == "mgsm":
            langs = {}
            for ib, ic in zip(b[task], c[task]):
                langs.setdefault(ib["lang"], ([], []))[0].append(ib["correct"])
                langs[ib["lang"]][1].append(ic["correct"])
            out["tasks"][task]["languages"] = {k: paired_acc(*v) for k, v in sorted(langs.items())}
        for side, items, acc in (("base", b[task], pb), ("cand", c[task], pc)):
            toks = sum(x.get("completion_tokens") or 0 for x in items)
            out["tasks"][task][f"{side}_tokens_per_item"] = toks / len(items)
        pb += xb
        pc += xc
    out["pooled"] = paired_acc(pb, pc)
    return out


def needles(l3: Path) -> dict | None:
    f = l3 / "needles.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    hits = [r for r in rows if r.get("kind") == "needle"]
    return {"correct": sum(r["correct"] for r in hits), "n": len(hits),
            "decode_tps": {f"{r['length']}/{r['mode']}": r["decode_tps"] for r in rows if r.get("kind") == "decode"}}


# ---------------------------------------------------------------- verdict

def verdict(l1: dict | None, l2: dict | None, l3: dict | None, stage: str | None) -> tuple[str, list[str]]:
    """The profile the candidate can enter, and why (the rules in the module docstring)."""
    why = []
    l1_within = l1 is not None and all(d["within_floor"] for k, d in l1.items() if k != "all")
    l2_within = l2 is not None and l2["within_floor"]
    if l1_within and l2_within and (l3 is None or l3["pooled"]["p"] >= 0.05):
        return "exact", ["layers 1 and 2 within their floors" + ("" if l3 is None else ", layer 3 not different")]
    speed = l2["decode_tps"] if l2 else None
    if speed and speed.get("n") and speed["ci95"][1] <= 1.0:
        return "rejected", [f"no speed gain: decode x{speed['ratio']:.3f} [{speed['ci95'][0]:.3f}, {speed['ci95'][1]:.3f}]"]
    if l3 is None:
        return "undecided", ["layer 3 has not run"]
    bad = [f"{t} {r['delta_pp']:+.1f} pp (p {r['p']:.3f})" for t, r in l3["tasks"].items() if r["p"] < 0.05 and r["losses"] > r["wins"]]
    pooled = l3["pooled"]
    if pooled["p"] < 0.05 and pooled["losses"] > pooled["wins"]:
        bad.append(f"pooled {pooled['delta_pp']:+.1f} pp (p {pooled['p']:.3f})")
    if pooled["delta_pp"] < -FAST_MARGIN:
        bad.append(f"pooled loss {pooled['delta_pp']:+.1f} pp exceeds {FAST_MARGIN} pp")
    if bad:
        return "rejected", ["significantly worse: " + "; ".join(bad)]
    if stage == "full" and pooled["ci95_pp"][0] >= -BALANCED_MARGIN and all(r["delta_pp"] >= -BALANCED_MARGIN for r in l3["tasks"].values()):
        return "balanced", [f"full battery non-inferior at {BALANCED_MARGIN} pp: pooled {pooled['delta_pp']:+.1f} pp {fmt_ci(pooled['ci95_pp'])}"]
    why.append(f"loss not significant: pooled {pooled['delta_pp']:+.1f} pp {fmt_ci(pooled['ci95_pp'])}, p {pooled['p']:.2f}")
    if stage != "full":
        why.append("balanced needs the full stage")
    return "fast", why


def screen_is_close(l3: dict) -> bool:
    """Escalate to the full sets unless the screen rejects or already excludes a loss beyond the balanced margin."""
    v, _ = verdict(None, None, l3, "screen")
    return v != "rejected" and l3["pooled"]["ci95_pp"][0] < -BALANCED_MARGIN


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"


def one_line(name: str, target: str, v: str, l1, l2, l3, stage) -> str:
    parts = [f"VERDICT {name} ({target}): {v}"]
    if l2:
        s = l2["decode_tps"]
        parts.append(f"speed x{s['ratio']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
    if l1:
        worst = max((k for k in l1 if k != "all"), key=lambda k: l1[k]["kl_excess"][0])
        a = l1["all"]
        parts.append(f"L1 KL +{a['kl_excess'][0]:.4f} [{a['kl_excess'][1]:+.4f}, {a['kl_excess'][2]:+.4f}] over floor {a['kl_floor']:.4f}, "
                     f"argmax {100 * a['argmax']:.1f}% (floor {100 * a['argmax_floor']:.1f}%), worst {worst} +{l1[worst]['kl_excess'][0]:.4f}")
    if l2:
        t = l2["tok_per_round"]
        parts.append(f"L2 tok/round x{t['ratio']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]")
    if l3:
        p = l3["pooled"]
        parts.append(f"L3 {stage} {p['delta_pp']:+.1f} pp {fmt_ci(p['ci95_pp'])} ({p['wins']}/{p['losses']}, p {p['p']:.2f})")
    return " | ".join(parts)


def summary_md(card: dict) -> str:
    m, l1, l2, l3 = card["manifest"], card.get("layer1"), card.get("layer2"), card.get("layer3")
    lines = [f"# Lossy-change gate: {m['name']} ({m['target']})", "",
             f"`{card['verdict_line']}`", "",
             f"Candidate: profile `{m['candidate']['profile']}` + `{m['candidate']['env']}`; baseline: profile `{m['baseline']['profile']}` + `{m['baseline']['env']}`.",
             f"Image `{m['image']}` ({m['image_id'][:19]}), engine source sha256 {card.get('engine_source_sha256', '?')[:12]} (the image's), "
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
    if l3:
        lines += [f"## Layer 3: task battery ({card['stage']}, served, 4 streams, paired, McNemar)", "",
                  "| task | n | baseline | candidate | delta pp [95%] | better/worse | p | tokens/item |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for t, r in list(l3["tasks"].items()) + [("pooled", l3["pooled"])]:
            tok = f"{r['base_tokens_per_item']:.0f} / {r['cand_tokens_per_item']:.0f}" if "base_tokens_per_item" in r else ""
            lines.append(f"| {t} | {r['n']} | {100 * r['base']:.1f}% | {100 * r['cand']:.1f}% | {r['delta_pp']:+.1f} {fmt_ci(r['ci95_pp'])} | "
                         f"{r['wins']}/{r['losses']} | {r['p']:.3f} | {tok} |")
        if "mgsm" in l3["tasks"]:
            langs = l3["tasks"]["mgsm"]["languages"]
            lines += ["", "MGSM by language (delta pp, better/worse): " + ", ".join(f"{k} {v['delta_pp']:+.0f} ({v['wins']}/{v['losses']})" for k, v in langs.items())]
        for side in ("baseline", "candidate"):
            nd = card.get(f"needles_{side}")
            if nd:
                lines.append(f"Needles ({side}): {nd['correct']}/{nd['n']} correct; decode tok/s {nd['decode_tps']}")
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
    ap.add_argument("--profile", choices=sorted(PROFILES), default="exact", help="the candidate's serving profile")
    ap.add_argument("--env", action="append", default=[], help="SHARAR_KEY=VALUE override of the candidate (repeatable)")
    ap.add_argument("--baseline-profile", choices=sorted(PROFILES), default="exact")
    ap.add_argument("--baseline-env", action="append", default=[])
    ap.add_argument("--through", choices=("inprocess", "screen", "full", "auto"), default="auto",
                    help="inprocess: layers 1-2; screen: also the layer-3 screen (forced); full: also the full sets; auto: the cascade")
    ap.add_argument("--image", default=None, help="runtime image (default: the config's SHARAR_IMAGE)")
    ap.add_argument("--port", type=int, default=30100, help="served arms' port (30100-30199)")
    ap.add_argument("--evals", type=Path, default=Path(os.environ.get("SHARAR_EVALS", "/home/yke/sharar-evals")))
    ap.add_argument("--python", default=os.environ.get("SHARAR_EVALS_PYTHON", "/home/yke/sharar/.venv-cpu/bin/python"),
                    help="host Python for sharar-evals (the needles need `tokenizers`)")
    ap.add_argument("--hold-minutes", type=float, default=45, help="longest GPU hold per phase")
    ap.add_argument("--no-needles", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rebuild the scorecard from what exists")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved arms and exit")
    args = ap.parse_args(argv)
    if not 30100 <= args.port <= 30199:
        ap.error("--port must be in 30100-30199")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.name):
        ap.error("--name: letters, digits, . _ -")

    conf_path = target_config(args.target)
    config = load_env(conf_path)
    results = Path(os.environ.get("SHARAR_RESULTS") or config.get("SHARAR_RESULTS") or "/home/yke/sharar-results")
    image = args.image or config.get("SHARAR_IMAGE") or "sharar:qwen38-27b"
    config.pop("SHARAR_IMAGE", None)
    hf_cache = config.get("SHARAR_HF_CACHE") or os.environ.get("SHARAR_HF_CACHE")
    cand_over, base_over = parse_env(args.env), parse_env(args.baseline_env)
    arms = {"baseline": resolve(config, args.baseline_profile, base_over), "candidate": resolve(config, args.profile, cand_over)}
    plans = {k: arm_plan(v) for k, v in arms.items()}
    image_id = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True).stdout.strip()
    if not image_id:
        raise SystemExit(f"image {image} not found")
    acc = args.evals / "evals" / "accuracy_eval.py"
    if not acc.is_file() or "mgsm" not in acc.read_text() or "--start" not in acc.read_text():
        raise SystemExit(f"{args.evals} has no MGSM/MBPP/--start accuracy_eval (sharar-evals branch swarm/S_lossy_gate or later)")
    evals_commit = git("rev-parse", "HEAD", cwd=args.evals)
    workload_sha = sha256(WORKLOAD.read_bytes())
    key_src = {"protocol": PROTOCOL, "target": args.target, "baseline": arms["baseline"],
               "image_id": image_id, "workload": workload_sha, "worker": sha256(WORKER.read_bytes()),
               "evals": git("rev-parse", "HEAD:evals", cwd=args.evals)}
    key = sha256(json.dumps(key_src, sort_keys=True).encode())[:12]
    base_dir = results / "reviews" / "lossy-gate-baselines" / f"{args.target}-{args.baseline_profile}-{key}"
    spec = {"target": args.target, "candidate": {"profile": args.profile, "env": cand_over}, "baseline": {"profile": args.baseline_profile, "env": base_over},
            "image": image, "baseline_key": key}
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
    manifest = {**spec, "name": args.name, "image_id": image_id, "config_file": str(conf_path), "through": args.through,
                "source": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "bench", "tools")),
                "evals": str(args.evals), "evals_commit": evals_commit, "workload": json.loads(WORKLOAD.read_text())["version"],
                "workload_sha256": workload_sha, "baseline_dir": str(base_dir), "port": args.port,
                "arms": {k: {"launcher_env": {kk: vv for kk, vv in v.items() if kk.startswith("SHARAR_") or kk == "_DRAFT_DIR"}, "plan": plans[k]} for k, v in arms.items()}}
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
        run_inprocess("baseline", tool_container(image, plans["baseline"], results, hf_cache, b_in, None), b_in, b_in / "floor.json", args.hold_minutes)
        run_inprocess("candidate", tool_container(image, plans["candidate"], results, hf_cache, c_in, b_in), c_in, c_in / "dist.json", args.hold_minutes)
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
        rates: dict = {}
        stages = [("screen", SCREEN)]
        for stage_name, tasks in stages:
            for side, l3dir in (("baseline", base_dir / "l3"), ("candidate", run_dir / "l3")):
                info = arm_info[side]
                expect = {"config": {k: info["config"][k] for k in CHECKED}, "spec": plans[side]["spec"],
                          "draft_precision": plans[side]["draft_precision"], "w3_identity": info["w3_identity"]}
                profile, over = (args.baseline_profile, base_over) if side == "baseline" else (args.profile, cand_over)
                served = Served(side, (base_dir if side == "baseline" else run_dir) / "served",
                                served_config(config, profile, over, image, args.port), args.port, args.python, args.evals, expect)
                if not args.report:
                    run_served(side, served, shards(tasks), l3dir, stage_name == "screen" and not args.no_needles, args.hold_minutes, rates)
            stage = stage_name
            l3 = layer3(base_dir / "l3", run_dir / "l3", tasks)
            if stage_name == "screen" and (args.through == "full" or (args.through == "auto" and screen_is_close(l3))):
                stages.append(("full", FULL))
        card["needles_baseline"], card["needles_candidate"] = needles(base_dir / "l3"), needles(run_dir / "l3")
    card["layer3"], card["stage"] = l3, stage
    v, why = verdict(l1, l2, l3, stage)
    card["verdict"], card["why"] = v, why
    card["verdict_line"] = one_line(args.name, args.target, v, l1, l2, l3, stage)
    card["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (run_dir / "scorecard.json").write_text(json.dumps(card, indent=1) + "\n")
    (run_dir / "SUMMARY.md").write_text(summary_md(card))
    print(card["verdict_line"], flush=True)
    log(f"scorecard: {run_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
