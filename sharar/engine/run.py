"""CLI drivers for generation and the decode-path canary check."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import CANARIES, NEEDLE, read_prompts
from sharar.tokenizer import SnapshotTokenizer

SURE_GAP = 5.0


def _canary_checks():
    return {
        "canary_exact": lambda text: text.strip() == "SPARKLANG_OK",
        "canary_arith": lambda text: text.strip() == "42",
        "canary_json": lambda text: json.loads(text.strip()) == {"device": "GB10", "model": "Qwen3.8-27B"},
        "canary_needle": lambda text: text.strip() == NEEDLE,
    }


def _load(graphs: bool = True) -> tuple[Qwen38, Executor, SnapshotTokenizer]:
    snap = config.snapshot_dir()
    print(f"loading {config.MODEL_ID}@{config.MODEL_REVISION[:8]}")
    model = Qwen38(snap)
    ex = Executor(model)
    print(f"  KV pool {ex.kv_pool.bytes / 2**30:.2f} GiB, state pool {ex.state_pool.num_slots} slots x {ex.state_pool.bytes_per_slot / 2**20:.0f} MiB")
    if graphs:
        started = time.perf_counter()
        ex.capture_graphs()
        print(f"  graphs captured in {time.perf_counter() - started:.1f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")
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


def _compare_continuation(name: str, mine: list[int], ref: dict) -> tuple[int, bool]:
    """Compare token by token until the first divergence. Returns (matched
    tokens, ok) where ok is False only if a divergence happened where the
    reference's margin was at least SURE_GAP nats."""
    ref_ids = ref["output_ids"]
    ref_top = ref["output_top_logprobs"]
    matched = 0
    for i, (a, b) in enumerate(zip(mine, ref_ids)):
        if a != b:
            top = ref_top[i]
            gap = (top[0][0] - top[1][0]) if len(top) > 1 else 9.0
            print(f"      diverges at token {i}: mine {a} ref {b} (ref margin {gap:.2f} nats{'' if gap >= SURE_GAP else ', within noise'})")
            return matched, gap < SURE_GAP
        matched += 1
    return matched, True


def canaries_cli(reference: str, max_new_tokens: int, batch: int, graphs: bool = True) -> int:
    ref_dir = Path(reference)
    prompts = read_prompts(ref_dir.parent / "prompts.json")
    model, ex, tok = _load(graphs)
    checks = _canary_checks()
    all_ok = True
    rows = []
    for name, ids in prompts.items():
        http = ref_dir / "http" / f"{name}.json"
        if not http.is_file():
            continue
        ref = json.loads(http.read_text())
        req = ex.generate(ids, SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0))
        text = tok.decode(req.output_ids, skip_special_tokens=True)
        tps = (len(req.output_ids) - 1) / max(1e-9, req.decode_seconds)
        matched, ok = _compare_continuation(name, req.output_ids, ref)
        sem = None
        if name in checks:
            try:
                sem = bool(checks[name](text))
            except Exception:  # noqa: BLE001 - a failed parse is a failed canary
                sem = False
            ok = ok and sem
        all_ok &= ok
        rows.append((name, len(ids), req.prefill_seconds, tps, matched, len(req.output_ids), sem))
        print(f"  [{name}] T={len(ids)} prefill {req.prefill_seconds:.2f}s  decode {tps:.2f} tok/s  "
              f"matched {matched}/{min(len(req.output_ids), len(ref['output_ids']))} reference tokens"
              f"{'' if sem is None else '  semantic ' + ('PASS' if sem else 'FAIL')}")
        print(f"      text: {text[:100]!r}")
    if batch > 1:
        names = [n for n in CANARIES if n != "canary_needle"][: batch]
        items = [(prompts[n], SamplingParams(max_new_tokens=max_new_tokens)) for n in names]
        reqs = ex.generate_batch(items)
        for n, r in zip(names, reqs):
            text = tok.decode(r.output_ids, skip_special_tokens=True)
            try:
                sem = bool(checks[n](text)) if n in checks else None
            except Exception:  # noqa: BLE001
                sem = False
            print(f"  [batch {len(reqs)}] {n}: {text[:60]!r} semantic {'PASS' if sem else 'FAIL'} ({r.decode_steps} steps in {r.decode_seconds:.2f}s, {r.decode_steps * len(reqs) / r.decode_seconds:.1f} aggregate tok/s)")
            all_ok &= bool(sem)
    print("CANARIES", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def bench_decode_cli(prompts_path: str, name: str, max_new_tokens: int, batches: str) -> int:
    """Decode throughput with and without graphs, plus determinism checks."""
    prompts = read_prompts(Path(prompts_path))
    ids = prompts[name]
    model, ex, tok = _load(graphs=True)
    weight_bytes = 19.2 * 2**30
    params = SamplingParams(max_new_tokens=max_new_tokens, ignore_eos=True)
    ok = True
    for b in (int(x) for x in batches.split(",")):
        items = [(ids, params) for _ in range(b)]
        ex.use_graphs = True
        reqs = ex.generate_batch(items)
        steps, secs = reqs[0].decode_steps, reqs[0].decode_seconds
        step_ms = 1000 * secs / steps
        print(f"  batch {b} graphs: {steps * b / secs:.2f} aggregate tok/s, {step_ms:.1f} ms/step, "
              f"{weight_bytes / (secs / steps) / 1e9:.0f} GB/s effective weight bandwidth")
        # determinism: rows identical across the batch and across a second run
        same_rows = all(r.output_ids == reqs[0].output_ids for r in reqs)
        reqs2 = ex.generate_batch(items)
        same_run = reqs2[0].output_ids == reqs[0].output_ids
        # Eager at the same batch size: GEMM tiling differs by M, so batch 1
        # and batch 4 are legitimately different roundings (the reference
        # documents the same "concurrent hash drift").
        ex.use_graphs = False
        reqs3 = ex.generate_batch(items)
        eager_secs, eager_steps = reqs3[0].decode_seconds, reqs3[0].decode_steps
        same_eager = reqs3[0].output_ids == reqs[0].output_ids
        print(f"  batch {b} eager:  {eager_steps * b / eager_secs:.2f} aggregate tok/s, {1000 * eager_secs / eager_steps:.1f} ms/step")
        print(f"  determinism: rows identical {same_rows}, rerun identical {same_run}, eager vs graph identical {same_eager}")
        ok &= same_rows and same_run and same_eager
    print("  text:", repr(tok.decode(reqs[0].output_ids)[:80]))
    print("BENCH", "OK" if ok else "DETERMINISM FAIL")
    return 0 if ok else 1


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
