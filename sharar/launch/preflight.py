"""Before serving: the host can serve this target, its pinned snapshots are present, its 3-bit checkpoint is current.

Torch-free (the launcher must not import torch before it hands over to the engine); the 3-bit checkpoint fetch runs
in a child process for that reason.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from sharar.launch import guard
from sharar.launch.settings import Settings

HEADROOM = 10 * 2**30  # left free after a download: logs, JIT caches, stored prefixes
LOAD_PEAK_KIB = 24 * 2**20  # the streaming load's measured MemAvailable drop, with margin (qwen38_config.WEIGHT_LOAD_PEAK_BYTES)


class PreflightError(RuntimeError):
    pass


def meminfo() -> dict[str, int]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        out[key] = int(rest.split()[0])
    return out


def check_host(floor_kib: int | None = None) -> dict[str, str]:
    """One NVIDIA GB10 (SM 12.1), driver 580+, about 128 GB unified memory, and the load's 24 GiB available above
    the host floor (`floor_kib`, default `guard.floor_kib()`: 30 GiB in all at the default 6 GiB floor)."""
    need = LOAD_PEAK_KIB + (guard.floor_kib() if floor_kib is None else floor_kib)
    if platform.machine() != "aarch64":
        raise PreflightError("host architecture must be aarch64 (a DGX Spark)")
    if shutil.which("nvidia-smi") is None:
        raise PreflightError("nvidia-smi is not installed")
    rows = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    if len(rows) != 1:
        raise PreflightError("exactly one NVIDIA GPU is required")
    name, cc, driver = (x.strip() for x in rows[0].split(","))
    if name != "NVIDIA GB10" or cc != "12.1":
        raise PreflightError(f"GPU must be NVIDIA GB10 SM 12.1 (found {name} {cc})")
    if int(driver.split(".")[0]) < 580:
        raise PreflightError("NVIDIA driver must be 580 or newer")
    mem = meminfo()
    if mem["MemTotal"] < 120_000_000:
        raise PreflightError("at least 120 GB host memory is required")
    if mem["MemAvailable"] < need:
        raise PreflightError(f"at least {need // 2**20} GiB must be available before the model loads (have {mem['MemAvailable'] // 2**20} GiB); "
                             "something else is using the memory")
    if not Path("/usr/local/cuda/bin/nvcc").is_file():
        raise PreflightError("the CUDA 13 toolkit is missing from /usr/local/cuda (DGX OS ships it): the GB10 kernels compile on first use")
    return {"gpu": f"{name} (SM {cc}), driver {driver}",
            "memory": f"{mem['MemTotal'] // 2**20} GiB total, {mem['MemAvailable'] // 2**20} GiB available"}


def snapshot_dir(cache: Path, repo: str, revision: str) -> Path | None:
    """The complete pinned snapshot (config.json and every weight file its index names) under cache or cache/hub."""
    for root in (cache, cache / "hub"):
        d = root / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        index = d / "model.safetensors.index.json"
        files = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
        if (d / "config.json").is_file() and all((d / f).is_file() for f in files):
            return d
    return None


def download(cache: Path, repo: str, revision: str, log=print) -> None:
    """One pinned snapshot into cache/hub, after checking it fits on the disk (an interrupted download resumes)."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi, snapshot_download
    hub = cache / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    snap = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    missing = sum(s.size or 0 for s in info.siblings if not (snap / s.rfilename).is_file())
    free = shutil.disk_usage(hub).free
    if missing + HEADROOM > free:
        raise PreflightError(f"{repo}@{revision[:7]} needs {missing / 1e9:.1f} GB more, and {cache} has {free / 1e9:.1f} GB free "
                             f"(keeping {HEADROOM / 1e9:.0f} GB spare); free space, or point SHARAR_HF_CACHE/SHARAR_FLASHNEXT_CACHE at a larger disk")
    log(f"downloading {repo}@{revision[:7]} into {cache}: {missing / 1e9:.1f} GB")
    snapshot_download(repo, revision=revision, cache_dir=hub)


def pinned_snapshots(s: Settings) -> list[tuple[Path, str, str]]:
    """(cache, repo, revision) of the target's checkpoint and, for a parallel draft served from the pinned snapshot, its draft."""
    if s.target == "flashnext":
        from sharar.model.flashnext import config as c
    else:
        from sharar.model import qwen38_config as c
    out = [(s.model_cache, c.MODEL_ID, c.MODEL_REVISION)]
    if s.parallel_draft and s.draft_dir is None:
        out.append((s.hf_cache, *c.DRAFT_PINS[s.proposer]))  # the target's own drafter
    return out


def ensure_snapshots(s: Settings, log=print, fetch: bool = True) -> list[str]:
    """Download whatever the cache lacks; returns what is still missing (with fetch False: what would be downloaded)."""
    missing = []
    for cache, repo, revision in pinned_snapshots(s):
        if snapshot_dir(cache, repo, revision) is not None:
            continue
        if not fetch:
            missing.append(f"{repo}@{revision[:7]}")
            continue
        download(cache, repo, revision, log)
        if snapshot_dir(cache, repo, revision) is None:
            raise PreflightError(f"{repo}@{revision[:7]} is still incomplete in {cache} after the download")
    return missing


def ensure_w3(s: Settings, log=print, fetch: bool = True) -> Path | None:
    """The calibrated 3-bit checkpoint to serve, or None (with the reason in s.warnings).

    SHARAR_W3_CHECKPOINT=auto (the default) serves w3-checkpoints/<target>-current and keeps that link on the
    target's pinned published revision (weights/w3_checkpoint.fetch: a new pin downloads, a local calibration stays
    with a note). Without one Flash-Next serves its FP8 trunk and the 27B nearest-rounded copies."""
    if s.get("SHARAR_W3_DECODE", "0") != "1":
        return None
    chosen = s.get("SHARAR_W3_CHECKPOINT", "auto")
    if chosen != "auto":
        path = Path(chosen).expanduser().resolve()
        if not (path / "manifest.json").is_file():
            raise PreflightError(f"SHARAR_W3_CHECKPOINT={chosen} needs a completed manifest.json")
        return path
    current = s.w3_root / f"{s.target}-current"
    if fetch:
        s.w3_root.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HF_HUB_OFFLINE": "0"}
        r = subprocess.run([sys.executable, "-m", "sharar.weights.w3_checkpoint", "fetch", "--target", s.target, "--root", str(s.w3_root)],
                           env=env, stdout=subprocess.DEVNULL)
        if r.returncode:
            log(f"could not update the 3-bit checkpoint for {s.target}; " +
                ("serving the one in " + current.name if (current / "manifest.json").is_file() else "none to serve"))
    if (current / "manifest.json").is_file():
        return current.resolve()
    if s.target == "flashnext":
        s.warnings.append("no calibrated 3-bit checkpoint: serving the FP8 trunk, about 24% slower single-stream than "
                          f"{s.profile} with it")
    else:
        s.warnings.append("no calibrated 3-bit checkpoint: nearest-rounded copies (about 2 GSM8K points and 8% tokens per "
                          "round below calibrated)")
    s.warnings[-1] += "; calibrate with docs/w3-calibration.md or set SHARAR_W3_CHECKPOINT"
    return None


def summary(s: Settings, host: dict[str, str], w3: Path | None, tp: int = 1) -> str:
    on = lambda v: "on" if v == "1" else "off"  # noqa: E731
    if s.proposer == "mtp":
        draft = "the model's own MTP head"
    elif s.draft_dir:
        draft = f"checkpoint {s.draft_dir}"
    elif s.parallel_draft:
        _, repo, revision = pinned_snapshots(s)[-1]
        draft = f"pinned {repo}@{revision[:7]}"
    else:
        draft = "none"
    sampled = s.get("SHARAR_ACCEPT_DEFAULT_SAMPLED", "0")
    ssd = f"{s.ssd}, {s.get('SHARAR_SSD_GIB', '200')} GiB" if s.ssd else "off"
    lines = [f"Sharar {s.target} ({'two Sparks' if tp > 1 else 'one Spark'}) on {s.url}, profile {s.profile}",
             f"  GPU: {host.get('gpu', '?')}; memory: {host.get('memory', '?')}",
             f"  speculation: {s.proposer}, {s.spec} drafts, draft {draft}",
             f"  acceptance: greedy {s.get('SHARAR_ACCEPT_DEFAULT')}, sampled "
             + (f"{s.get('SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD') or s.get('SHARAR_ACCEPT_DEFAULT')}" if sampled == "1" else "exact"),
             f"  3-bit copies: {on(s.get('SHARAR_W3_DECODE'))}{f' ({w3})' if w3 else ''}; device-resident acceptance: {on(s.get('SHARAR_DEVICE_ACCEPT'))}",
             f"  context {s.max_context} tokens, up to {s.get('SHARAR_MAX_BATCH')} requests"
             + (f", verify depth by batch {s.get('SHARAR_SPEC_DEPTHS')}" if s.get("SHARAR_SPEC_DEPTHS") else ""),
             f"  traces: {s.traces or 'off'}; SSD tier: {ssd}"]
    lines += [f"  WARNING: {w}" for w in s.warnings]
    return "\n".join(lines)
