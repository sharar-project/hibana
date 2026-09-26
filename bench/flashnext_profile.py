# SPDX-License-Identifier: Apache-2.0
"""Kernel-level breakdown of eager Flash-Next decode steps (torch profiler) plus graph-replay step time."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from sharar.engine.run import _load
from sharar.engine.sampler import SamplingParams


def main():
    prompts = json.loads(Path(sys.argv[1]).read_text())
    ids = prompts[sys.argv[2] if len(sys.argv) > 2 else "code512"]
    steps = int(os.environ.get("STEPS", "12"))
    top = int(os.environ.get("TOP", "40"))
    model, ex, tok = _load(graphs=True)
    params = SamplingParams(max_new_tokens=steps, ignore_eos=True)
    ex.use_graphs = True
    r = ex.generate(ids, params)
    print(f"  graphs: {1000 * r.decode_seconds / r.decode_steps:.1f} ms/step over {r.decode_steps} steps", flush=True)
    # GPU-only step time: replay the batch-1 graph with its static inputs unchanged
    g = ex.graphs._graphs[1, False]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):
        g.replay()
    b.record(); torch.cuda.synchronize()
    print(f"  graph replay only: {a.elapsed_time(b) / 20:.1f} ms/step (GPU time of one decode step)", flush=True)
    # host cost of the launch itself (no sync): how long replay() blocks the host
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    t_launch = time.perf_counter() - t0
    torch.cuda.synchronize()
    print(f"  graph launch host time: {t_launch * 1000:.2f} ms (replay() call)", flush=True)
    # one full decode_step through the executor, phase-timed on the host
    req0 = ex.admit(ids, SamplingParams(max_new_tokens=4, ignore_eos=True))
    ex.prefill(req0)
    torch.cuda.synchronize()
    for i in range(3):
        t0 = time.perf_counter()
        ex.decode_step([req0])
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        print(f"  decode_step {i}: host wall {1000 * (t1 - t0):.2f} ms, to GPU idle {1000 * (time.perf_counter() - t0):.2f} ms", flush=True)
    ex.release(req0)
    # host work per step outside the graph
    req = ex.admit(ids, SamplingParams(max_new_tokens=2, ignore_eos=True))
    ex.prefill(req)
    if ex.extras is not None and ex.extras.ple is not None:
        t0 = time.perf_counter()
        for _ in range(50):
            ex.extras._decode_rows([req])
        torch.cuda.synchronize()
        print(f"  host PLE rows per step: {(time.perf_counter() - t0) / 50 * 1000:.2f} ms", flush=True)
    t0 = time.perf_counter()
    for _ in range(50):
        ex._plan_decode([req], [ex.state.step_pages(req) if req.reservation is not None else req.pages])
    print(f"  host decode plan per step: {(time.perf_counter() - t0) / 50 * 1000:.2f} ms", flush=True)
    ex.release(req)
    ex.use_graphs = False
    ex.generate(ids, params)  # warm
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        r = ex.generate(ids, params)
    print(f"  eager:  {1000 * r.decode_seconds / r.decode_steps:.1f} ms/step over {r.decode_steps} steps (prefill excluded from the ratio, included in the profile)", flush=True)
    by_kernel = defaultdict(lambda: [0.0, 0])
    total = 0.0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0 and not e.cpu_children:
            by_kernel[e.name][0] += e.device_time_total / 1000.0
            by_kernel[e.name][1] += 1
            total += e.device_time_total / 1000.0
    print(f"  CUDA kernel time {total:.1f} ms in the profiled window ({steps} decode steps + prefill); top {top} by time:")
    for name, (ms, n) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"    {ms:8.2f} ms  {n:6d} calls  {ms / n * 1000:7.1f} us/call  {name[:110]}")
    print("  top 25 CPU-side ops by self CPU time (host overhead and syncs):")
    rows = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:25]
    for e in rows:
        print(f"    {e.self_cpu_time_total / 1000:8.2f} ms  {e.count:6d} calls  {e.key[:100]}")
    print("done")


if __name__ == "__main__":
    main()
