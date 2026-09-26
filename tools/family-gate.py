#!/usr/bin/env python3
"""The family gate: paired 105-task evaluation of engine candidates in exclusive GPU windows.

Runs `bench.gb10_family_eval run` once per pass, each in a fresh container from
the runtime image with the working tree mounted read-only, then pairs every
candidate pass against the first native pass (and reports the noise floor from
the last native pass). Production and the memory guard are stopped before the
first pass and restored after the last, whatever happens.

    tools/family-gate.py --label stable-vs-native --order N,C1,N \
        --candidates env:SHARAR_STABLE_NUMERICS=1
    tools/family-gate.py --label w3 --order N,C1,C2,N \
        --candidates bench.gate_hooks:w3_decode,bench.gate_hooks:w3_decode_head_draft

Order tokens: N = native (FP8 draft unless --draft-precision), B = native with
the BF16 draft, Ck = the k-th entry of --candidates. A candidate is a hook
`module:function` (see bench/gate_hooks.py) or `env:KEY=VALUE`; append `|bf16`
to run that pass with the BF16 draft. Results go to
$SHARAR_RESULTS/reviews/<date>-family-gate/<label>/ with the frozen inputs, one
log per pass, a MemAvailable trace and comparison-*.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN_SHA = {  # dataset name -> sha256 of task-inputs.json; regenerate with bench/family_inputs.py freeze
    "family-eval-v1": "0c0da566c177b9c55dee6a776ad5ba62ff466bd678f20c2285aa2647aebdd2ac",
    "family-eval-v2": "fc7332830b97f84218dc40c8a6763688350f4db466e2dad984642c9df7ff0451",
    "family-eval-v3": "b72bdecd50c2fb0a7a00eda1433bb697a305160b8f6e72c1c74cb0b3530124a3",
}
DEFAULT_SET = "family-eval-v3"  # 101 families plus the four benchmark prompts (v2 = the 18-family set plus benchmark_code_v2)
REFERENCE_V1_SHA = "6dd2b3cc868bcf99c9465d7355806732226de0f2b6b89ba5d5045d933a79cfd5"
SERVICE = "sharar-qwen38"
GUARD_UNIT = "sharar-memory-guard.service"


def load_env(path: Path) -> dict:
    env = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"')
    return env


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable not found")


def health(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def container_base(image: str, uid: str, name: str, results: Path, hf_cache: str) -> list[str]:
    """Device and library mounts: cloned from the production container when it exists, else the launcher's set."""
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--pull", "never",
           "--user", uid, "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
           "--pids-limit", "4096", "--shm-size", "16g", "--workdir", "/workspace/sharar"]
    inspect = subprocess.run(["docker", "inspect", SERVICE], capture_output=True, text=True)
    if inspect.returncode == 0:
        source = json.loads(inspect.stdout)[0]
        for device in source["HostConfig"]["Devices"]:
            cmd += ["--device", device["PathOnHost"]]
        for mount in source["Mounts"]:
            if mount["Destination"] in ("/traces", "/draft", "/runs", "/proof", "/w3"):
                continue
            cmd += ["--mount", f"type=bind,src={mount['Source']},dst={mount['Destination']}" + (",readonly" if not mount["RW"] else "")]
    else:
        lib = "/usr/lib/aarch64-linux-gnu"
        for device in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/dri/renderD128"):
            cmd += ["--device", device]
        cmd += ["--mount", f"type=bind,src={hf_cache},dst=/cache/huggingface,readonly",
                "--mount", f"type=bind,src={results}/runtime-state,dst=/state",
                "--mount", "type=bind,src=/usr/bin/nvidia-smi,dst=/usr/bin/nvidia-smi,readonly"]
        for so, dst in (("libcuda.so.1", "libcuda.so.1"), ("libcuda.so.1", "libcuda.so"), ("libnvidia-ptxjitcompiler.so.1", "libnvidia-ptxjitcompiler.so.1"),
                        ("libnvidia-ptxjitcompiler.so.1", "libnvidia-ptxjitcompiler.so"), ("libnvidia-ml.so.1", "libnvidia-ml.so.1"), ("libnvidia-ml.so.1", "libnvidia-ml.so"),
                        ("libnvidia-nvvm.so.4", "libnvidia-nvvm.so.4"), ("libnvidia-nvvm.so.4", "libnvidia-nvvm.so"), ("libcudadebugger.so.1", "libcudadebugger.so.1")):
            cmd += ["--mount", f"type=bind,src={os.path.realpath(f'{lib}/{so}')},dst={lib}/{dst},readonly"]
        for gpucomp in Path(lib).glob("libnvidia-gpucomp.so.*"):
            cmd += ["--mount", f"type=bind,src={gpucomp},dst={gpucomp},readonly"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--order", default="N,N", help="comma list of N, B, Ck")
    ap.add_argument("--candidates", default="", help="comma list: module:function or env:KEY=VALUE, optional |bf16")
    ap.add_argument("--inputs", type=Path, default=None, help="frozen task-inputs.json (default: $SHARAR_RESULTS/datasets/<set>/)")
    ap.add_argument("--set", default=DEFAULT_SET, choices=sorted(FROZEN_SHA), help="frozen dataset under $SHARAR_RESULTS/datasets/")
    ap.add_argument("--reference", type=Path, default=None, help="reference-ids.json for the benchmark prompts")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--output-tokens", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--draft-precision", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--image", default=os.environ.get("SHARAR_IMAGE", "sharar:qwen38-27b"))
    ap.add_argument("--results", type=Path, default=None, help="run directory (default: $SHARAR_RESULTS/reviews/<date>-family-gate/<label>)")
    ap.add_argument("--time-guard", type=float, default=4 * 3600)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every pass")
    ap.add_argument("--w3-checkpoint", type=Path, help="mount a completed calibrated artifact at /w3; enable it in a candidate with env:SHARAR_W3_CHECKPOINT=/w3")
    ap.add_argument("--dry-run", action="store_true", help="print the pass commands and exit")
    args = ap.parse_args()

    cfg = load_env(REPO / "config" / "gb10.env")
    results_root = Path(os.environ.get("SHARAR_RESULTS") or cfg.get("SHARAR_RESULTS") or "")
    hf_cache = os.environ.get("SHARAR_HF_CACHE") or cfg.get("SHARAR_HF_CACHE") or ""
    if not results_root or not hf_cache:
        sys.exit("set SHARAR_RESULTS and SHARAR_HF_CACHE (config/gb10.env)")
    inputs = args.inputs or results_root / "datasets" / args.set / "task-inputs.json"
    reference = args.reference or (results_root / "datasets" / args.set / "reference-ids.json" if args.inputs is None else None)
    if not inputs.is_file():
        sys.exit(f"missing frozen inputs {inputs}; regenerate with bench/family_inputs.py freeze")
    inputs_sha = sha256(inputs)
    if args.inputs is None and inputs_sha != FROZEN_SHA[args.set]:
        sys.exit(f"{inputs} does not match the frozen {args.set} hash")
    run_dir = args.results or results_root / "reviews" / f"{dt.date.today().isoformat()}-family-gate" / args.label
    if run_dir.exists():
        sys.exit(f"{run_dir} exists")

    hooks = [h for h in args.candidates.split(",") if h]
    passes = []
    for i, token in enumerate(t.strip() for t in args.order.split(",")):
        if token == "N":
            passes.append((token, f"native-{i}", args.draft_precision, None, {}))
        elif token == "B":
            passes.append((token, f"bf16draft-{i}", "bf16", None, {}))
        elif token.startswith("C"):
            spec = hooks[int(token[1:] or "1") - 1]
            spec, _, precision = spec.partition("|")
            env = {}
            hook = None
            if spec.startswith("env:"):
                k, v = spec[4:].split("=", 1)
                env[k] = v
                name = k.lower().replace("sharar_", "")
            else:
                hook = spec
                name = spec.split(":")[-1]
            passes.append((token, f"{name}-{i}", precision or args.draft_precision, hook, env))
        else:
            sys.exit(f"unknown order token {token}")

    uid = f"{os.getuid()}:{os.getgid()}"
    name = f"sharar-family-gate"
    base = container_base(args.image, uid, name, results_root, hf_cache)
    base += ["--mount", f"type=bind,src={REPO},dst=/workspace/sharar,readonly",
             "--mount", f"type=bind,src={run_dir},dst=/proof",
             "--env", "PYTHONPATH=/workspace/sharar", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "SHARAR_RUN_GPU=1",
             "--env", "SHARAR_HF_CACHE=/cache/huggingface", "--env", "HF_HUB_OFFLINE=1", "--env", "HOME=/state",
             "--env", "TOKENIZERS_PARALLELISM=false", "--env", "TRITON_LIBCUDA_PATH=/usr/lib/aarch64-linux-gnu"]
    for kv in args.env:
        base += ["--env", kv]
    if args.w3_checkpoint:
        if not (args.w3_checkpoint / "manifest.json").is_file():
            sys.exit("--w3-checkpoint needs a completed manifest.json")
        base += ["--mount", f"type=bind,src={args.w3_checkpoint.resolve()},dst=/w3,readonly"]
    commands = []
    for token, label, precision, hook, env in passes:
        cmd = list(base)
        for k, v in env.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += ["--entrypoint", "python3", args.image, "-u", "-m", "bench.gb10_family_eval", "run",
                "--inputs", "/proof/inputs/task-inputs.json", "--output", f"/proof/passes/{label}", "--label", label,
                "--tasks", args.tasks, "--output-tokens", str(args.output_tokens), "--repeats", str(args.repeats),
                "--draft-precision", precision]
        if reference is not None:
            cmd += ["--reference", "/proof/inputs/reference-ids.json"]
        if hook:
            cmd += ["--candidate", hook]
        commands.append((token, label, cmd))
    if args.dry_run:
        for _, label, cmd in commands:
            print(label, " ".join(cmd))
        return 0

    run_dir.mkdir(parents=True)
    (run_dir / "inputs").mkdir()
    (run_dir / "passes").mkdir()
    (run_dir / "inputs" / "task-inputs.json").write_bytes(inputs.read_bytes())
    if reference is not None:
        (run_dir / "inputs" / "reference-ids.json").write_bytes(reference.read_bytes())
    manifest = {"label": args.label, "order": args.order, "candidates": hooks, "tasks": args.tasks, "output_tokens": args.output_tokens,
                "repeats": args.repeats, "draft_precision": args.draft_precision, "image": args.image, "inputs_sha256": inputs_sha,
                "reference_sha256": sha256(reference) if reference is not None else None,
                "source": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "dirty": bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
                "commands": {label: cmd for _, label, cmd in commands}, "started": dt.datetime.now().isoformat(timespec="seconds")}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    inspect = subprocess.run(["docker", "inspect", SERVICE], capture_output=True, text=True)
    running = inspect.returncode == 0 and json.loads(inspect.stdout)[0]["State"]["Running"]
    guard = subprocess.run(["systemctl", "--user", "is-active", "--quiet", GUARD_UNIT]).returncode == 0
    if running:
        try:
            with urllib.request.urlopen("http://127.0.0.1:30001/get_server_info", timeout=5) as r:
                info = json.load(r)
            if any(info["scheduler"][k] for k in ("queued", "prefilling", "running")):
                sys.exit("production has active requests; try again later")
        except urllib.error.URLError:
            pass
    completed, failed = [], []
    try:
        if guard:
            subprocess.run(["systemctl", "--user", "stop", GUARD_UNIT], check=True)
        if running:
            subprocess.run(["docker", "stop", "--timeout", "30", SERVICE], check=True)
        started = time.monotonic()
        with (run_dir / "resources.jsonl").open("w") as resources:
            for token, label, cmd in commands:
                print(f"==> {label}", flush=True)
                with (run_dir / f"{label}.log").open("w") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        while proc.poll() is None:
                            avail = available_gib()
                            resources.write(json.dumps({"pass": label, "elapsed_s": round(time.monotonic() - started, 1), "available_gib": round(avail, 2)}) + "\n")
                            resources.flush()
                            if avail < 16 or time.monotonic() - started > args.time_guard:
                                raise RuntimeError("memory/time guard reached")
                            time.sleep(2)
                    finally:
                        if proc.poll() is None:
                            subprocess.run(["docker", "stop", "--timeout", "10", name], check=False)
                            proc.wait(timeout=60)
                if proc.returncode:
                    print(f"    {label} FAILED with exit {proc.returncode}; see {run_dir / (label + '.log')}", flush=True)
                    failed.append(label)
                    continue
                completed.append((token, label))
                report = json.loads((run_dir / "passes" / label / "report.json").read_text())
                summary = [(t["id"], round(t["metrics"]["decode_tps"], 2), t["spec_rounds"], t["metrics"]["first_eos"]) for t in report["tasks"]]
                print(f"    {label}: all_ok={report.get('all_ok')} tasks={len(summary)}", flush=True)
    finally:
        if running:
            subprocess.run(["docker", "start", SERVICE], check=False)
        if guard:
            subprocess.run(["systemd-run", "--user", "--quiet", "--collect", "--unit", GUARD_UNIT,
                            "--property", "Restart=on-failure", "--property", "RestartSec=2s",
                            "--setenv", f"SHARAR_RESULTS={results_root}", str(REPO / "tools" / "memory-guard.sh"), SERVICE], check=False)
        if running:
            for _ in range(180):
                if health("http://127.0.0.1:30001"):
                    print("production restored", flush=True)
                    break
                time.sleep(1)
            else:
                print("WARNING: production did not report healthy within 3 minutes", flush=True)

    natives = [label for token, label in completed if token == "N"]
    bf16_natives = [label for token, label in completed if token == "B"]
    precision_of = {label: precision for _, label, precision, _, _ in passes}
    groups: dict[str, list[str]] = {}
    for token, label in completed:
        if token not in ("N", "B"):
            groups.setdefault(token, []).append(label)
    python = REPO / ".venv-cpu" / "bin" / "python"
    python = str(python) if python.exists() else sys.executable
    if natives and groups:
        for token, labels in groups.items():
            # A BF16-draft candidate is paired with the BF16 native pass when one ran.
            controls = bf16_natives if all(precision_of[l] == "bf16" for l in labels) and bf16_natives else natives
            cmd = [python, "-m", "bench.gb10_family_eval", "compare", "--baseline", str(run_dir / "passes" / controls[0]),
                   "--candidate", *[str(run_dir / "passes" / l) for l in labels], "--output", str(run_dir / f"comparison-{token}.json")]
            if len(controls) > 1:
                cmd += ["--baseline-repeat", str(run_dir / "passes" / controls[-1])]
            print(f"==> compare {token} ({', '.join(labels)}) vs {controls[0]}", flush=True)
            subprocess.run(cmd, cwd=REPO, check=False)
        if len(natives) > 1:
            subprocess.run([python, "-m", "bench.gb10_family_eval", "compare", "--baseline", str(run_dir / "passes" / natives[0]),
                            "--candidate", str(run_dir / "passes" / natives[-1]), "--output", str(run_dir / "comparison-noise.json")], cwd=REPO, check=False)
    elif len(natives) > 1:
        subprocess.run([python, "-m", "bench.gb10_family_eval", "compare", "--baseline", str(run_dir / "passes" / natives[0]),
                        "--candidate", str(run_dir / "passes" / natives[-1]), "--output", str(run_dir / "comparison-noise.json")], cwd=REPO, check=False)
    manifest["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    manifest["completed_passes"] = [label for _, label in completed]
    manifest["failed_passes"] = failed
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"results: {run_dir}")
    return 0 if len(completed) == len(commands) else 1


if __name__ == "__main__":
    sys.exit(main())
