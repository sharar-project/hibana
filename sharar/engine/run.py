"""CLI drivers for generation and the decode-path canary check."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.engine.prompts import NEEDLE, read_prompts
from sharar.tokenizer import SnapshotTokenizer

def _canary_checks():
    return {
        "canary_exact": lambda text: text.strip() == "SHARAR_OK",
        "canary_arith": lambda text: text.strip() == "42",
        "canary_json": lambda text: json.loads(text.strip()) == {"device": "GB10", "model": "Qwen3.8-27B"},
        "canary_needle": lambda text: text.strip() == NEEDLE,
    }


def _load(graphs: bool = True, spec: int = 0) -> tuple[Qwen38, Executor, SnapshotTokenizer]:
    import os
    from sharar import target as targets
    from sharar.engine.config import current
    cfg = current()
    target = targets.get(cfg.target)
    snap = target.snapshot_dir()
    print(f"loading {target.model_id}@{target.model_revision[:8]} ({target.name})")
    model = targets.load_model(target, snap)
    spec_config = None
    if spec:
        from sharar.cache.spec_budget import SpecConfig
        spec_config = SpecConfig("mtp", spec, cfg.bench_verify_state, "bf16", None)  # reserve the verify/draft budget in the pool plan
    max_context = cfg.bench_max_context or None  # smaller pools for large models
    ex = Executor(model, log=print, spec_config=spec_config, max_context=max_context)  # a spec_config enables speculation itself
    if cfg.commit_profile:
        ex.state.commit_profile = {}
    if graphs:
        started = time.perf_counter()
        ex.capture_graphs()
        print(f"  graphs captured in {time.perf_counter() - started:.1f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")
    started = time.perf_counter()
    ex.warmup()
    print(f"  warmup {time.perf_counter() - started:.1f}s")
    ex.enable_cold_tier(log=print)
    ex.timing = True  # CLI benchmarks report synchronized prefill/decode wall time
    return model, ex, SnapshotTokenizer(snap)


def generate_cli(prompts_path: str, name: str, max_new_tokens: int, temperature: float, graphs: bool = True) -> int:
    prompts = read_prompts(Path(prompts_path))
    model, ex, tok = _load(graphs)
    req = ex.generate(prompts[name], SamplingParams(max_new_tokens=max_new_tokens, temperature=temperature))
    text = tok.decode(req.output_ids)
    print(f"[{name}] prefill {req.prefill_seconds:.2f}s for {len(req.prompt_ids)} tokens; "
          f"{len(req.output_ids)} tokens in {req.decode_seconds:.2f}s ({(len(req.output_ids) - 1) / max(1e-9, req.decode_seconds):.2f} tok/s); finish={req.finish_reason}")
    print(f"  ids: {req.output_ids}")
    print(f"  text: {text!r}")
    return 0


def profile_cli(prompts_path: str, name: str, mode: str, steps: int = 8, top: int = 30) -> int:
    """Kernel-level time breakdown of prefill and/or one decode step (eager)."""
    prompts = read_prompts(Path(prompts_path))
    model, ex, tok = _load(graphs=False)
    for m in (("prefill", "decode") if mode == "both" else (mode,)):
        _profile_one(ex, prompts[name if m == "prefill" else "rand128"], m, steps, top)
    return 0


def _profile_one(ex, ids, mode, steps, top):
    from torch.profiler import ProfilerActivity, profile

    params = SamplingParams(max_new_tokens=steps + 1, ignore_eos=True)
    if mode == "prefill":
        req = ex.admit(ids, params)
        ex.prefill(req)  # warm
        ex.release(req)
        ex.flush_cache()  # measure a cold prefill, not a prefix-cache hit on the warm run
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            req = ex.admit(ids, params)
            ex.prefill(req)
            torch.cuda.synchronize()
        ex.release(req)
        print(f"prefill T={len(ids)}: {req.prefill_seconds * 1000:.1f} ms")
    else:
        req = ex.admit(ids, params)
        ex.prefill(req)
        for _ in range(2):
            ex.decode_step([req])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            t0 = time.perf_counter()
            for _ in range(steps):
                ex.decode_step([req])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        ex.release(req)
        print(f"decode batch 1: {1000 * dt / steps:.1f} ms/step eager over {steps} steps")
    rows = []
    for e in prof.key_averages():
        if e.device_time_total > 0 and e.self_device_time_total > 0:
            rows.append((e.self_device_time_total, e.count, e.key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    div = steps if mode == "decode" else 1
    print(f"  total GPU kernel time {total / 1000 / div:.1f} ms per {'step' if mode == 'decode' else 'prefill'}; top {top} kernels:")
    print(f"  {'ms':>8} {'count':>7}  kernel")
    for t, c, k in rows[:top]:
        print(f"  {t / 1000 / div:8.2f} {c // div:7d}  {k[:110]}")
