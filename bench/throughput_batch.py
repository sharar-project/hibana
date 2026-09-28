# SPDX-License-Identifier: Apache-2.0
"""Aggregate throughput by batch size and verify depth through the scheduler (#59).

One process loads the served target with `max_batch` 16, captures verify graphs
for every depth in `--depths` at every batch bucket, then runs each cell
`BATCH:DEPTH` as a continuous-batching workload: `--prompts` GSM8K problems (the
accuracy eval's prompt, thinking off, greedy, exact acceptance, 640 tokens)
submitted at once to a `Scheduler` with `max_running` = BATCH, run until idle.
DEPTH 0 decodes plainly (the proposer still follows every row); `plain` detaches
the proposer for the whole cell (the upper bound on plain decode). Wall time is
synchronized at both ends, so it includes prefill, host work and the tail where
fewer than BATCH requests remain, as an eval run does.

    SHARAR_TARGET=flashnext python3 -m bench.throughput_batch --proposer mtp --spec 4 --depths 1,2,4 \\
        --cells 4:4,8:0,8:1,8:2,16:0,16:1,16:2 --prompts 64 --out /results/...json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch

GSM_SUFFIX = "\n\nSolve the problem step by step. On the last line write only: Answer: <number>"
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm_correct(text: str, answer: str) -> bool:
    tail = text.rsplit("Answer:", 1)
    nums = NUM.findall(tail[1] if len(tail) == 2 else text) or NUM.findall(text)
    if not nums:
        return False
    pred, gold = nums[-1].replace(",", "").rstrip("."), answer.split("####")[-1].strip().replace(",", "")
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def load(args):
    from sharar import target as targets
    from sharar.cache.spec_budget import SpecConfig
    from sharar.engine.config import current, install
    from sharar.engine.executor import Executor
    install(current().replace(max_batch=args.max_batch))
    served = targets.get(current().target)
    snap = served.snapshot_dir()
    model = targets.load_model(served, snap, log=print)
    spec = SpecConfig(args.proposer, args.spec, "reconstruct", args.draft_precision, args.draft) if args.spec else None
    ex = Executor(model, log=print, spec_config=spec, max_context=args.context, max_batch=args.max_batch)
    if ex.spec is not None:
        ex.spec.verify_depths = tuple(sorted({int(d) for d in args.depths.split(",")} | {args.spec}))
    started = time.perf_counter()
    ex.capture_graphs(log=print)
    print(f"graphs captured in {time.perf_counter() - started:.1f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)
    ex.warmup(log=print)
    return served, snap, ex


def run_cell(ex, prompts, batch: int, depth, max_tokens: int):
    from sharar.engine.depth import BatchDepth
    from sharar.engine.graphs import capture_sizes
    from sharar.engine.sampler import SamplingParams
    from sharar.engine.scheduler import Scheduler, SchedulerConfig
    ex.flush_cache()  # every cell prefills its prompts cold
    spec = ex.spec
    if depth == "plain":
        ex.spec = None
    else:
        spec.depth_policy = BatchDepth(dict.fromkeys(capture_sizes(ex.max_batch), depth), spec.k)
        spec.reset_stats()
    try:
        sched = Scheduler(ex, SchedulerConfig(max_running=batch, max_queued=len(prompts)))
        params = SamplingParams(max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reqs = [sched.submit(ids, params) for ids in prompts]
        occupancy = []
        while sched.active:
            sched.step()
            occupancy.append(len(sched.running))
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    finally:
        ex.spec = spec
    tokens = sum(len(r.output_ids) for r in reqs)
    stats = spec.stats() if depth != "plain" and spec is not None else {}
    return reqs, {"batch": batch, "depth": depth, "prompts": len(prompts), "wall_s": round(wall, 3), "output_tokens": tokens,
                  "tok_s": round(tokens / wall, 2), "mean_running": round(sum(occupancy) / max(1, len(occupancy)), 2),
                  "steps": sched.metrics.steps, "retractions": sched.metrics.retractions,
                  "tokens_per_round": round(stats.get("tokens_per_round", 0.0), 3), "rounds": stats.get("rounds", 0)}


def profile_cell(ex, prompts, batch: int, depth, steps: int = 24) -> dict:
    """Steady-state steps with `batch` requests decoding (every prefill done): wall time against GPU kernel
    time by category (the remainder is host work, launches and idle gaps), and the host functions that
    took the longest."""
    from torch.profiler import ProfilerActivity, profile
    from bench.flashnext_round import kernel_time
    from sharar.engine.depth import BatchDepth
    from sharar.engine.graphs import capture_sizes
    from sharar.engine.sampler import SamplingParams
    from sharar.engine.scheduler import Scheduler, SchedulerConfig
    ex.flush_cache()
    spec = ex.spec
    if depth == "plain":
        ex.spec = None
    else:
        spec.depth_policy = BatchDepth(dict.fromkeys(capture_sizes(ex.max_batch), depth), spec.k)
    try:
        sched = Scheduler(ex, SchedulerConfig(max_running=batch, max_queued=batch))
        for ids in prompts[:batch]:
            sched.submit(ids, SamplingParams(max_new_tokens=640, ignore_eos=True))
        while sched.queue or sched.prefilling:
            sched.step()
        for _ in range(4):
            sched.step()
        torch.cuda.synchronize()
        before = sum(len(r.output_ids) for r in sched.running)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(steps):
                sched.step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        tokens = sum(len(r.output_ids) for r in sched.running) - before
        for r in list(sched.running):
            sched.cancel(r)
        sched.step()
    finally:
        ex.spec = spec
    cats, total, top = kernel_time(prof)
    host = sorted(((e.key, e.self_cpu_time_total / 1000) for e in prof.key_averages()), key=lambda kv: -kv[1])[:12]
    return {"batch": batch, "depth": depth, "steps": steps, "running": len(sched.running) or batch, "wall_ms_per_step": round(1000 * wall / steps, 2),
            "kernel_ms_per_step": round(total / steps, 2), "tokens_per_step": round(tokens / steps, 2), "tok_s": round(tokens / wall, 1),
            "kernel_by_category_ms_per_step": {k: round(v / steps, 2) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
            "top_kernels_ms_per_step": {k: round(v / steps, 3) for k, v in top.items()},
            "host_self_ms_per_step": {k: round(v / steps, 3) for k, v in host}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposer", default="mtp")
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--draft-precision", default="bf16")
    ap.add_argument("--draft", default=None)
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--cells", default="4:4,16:0,16:1")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--offset", type=int, default=0, help="first GSM8K item")
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--data", default=os.path.join(os.environ.get("SHARAR_RESULTS", "/results"), "datasets/accuracy/gsm8k_test.jsonl"))
    ap.add_argument("--profile", default="", help="BATCH:DEPTH cells to profile in steady state after the timed cells")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from sharar.api.chat_template import ChatTemplate
    from sharar.tokenizer import SnapshotTokenizer
    with torch.inference_mode():
        served, snap, ex = load(args)
        tok, tmpl = SnapshotTokenizer(snap), ChatTemplate(snap)
        items = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()][args.offset:args.offset + args.prompts]
        prompts = [tok.encode(tmpl.render([{"role": "user", "content": it["question"] + GSM_SUFFIX}], enable_thinking=False)) for it in items]
        report = {"target": served.name, "proposer": args.proposer, "spec": args.spec, "draft_precision": args.draft_precision,
                  "draft": args.draft, "max_tokens": args.max_tokens, "offset": args.offset, "prompts": len(prompts),
                  "kv_pages": ex.kv_pool.num_pages, "state_slots": ex.state_pool.num_slots,
                  "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                  "config": {k: v for k, v in __import__("sharar.engine.config", fromlist=["current"]).current().describe().items()},
                  "cells": []}
        run_cell(ex, prompts[:args.max_batch], args.max_batch, "plain" if ex.spec is None else 0, 32)  # warm the scheduler path at every size
        for cell in filter(None, args.cells.split(",")):
            b, d = cell.split(":")
            depth = d if d == "plain" else int(d)
            reqs, row = run_cell(ex, prompts, int(b), depth, args.max_tokens)
            texts = [tok.decode(r.output_ids, skip_special_tokens=True) for r in reqs]
            row["correct"] = sum(gsm_correct(t, it["answer"]) for t, it in zip(texts, items))
            row["outputs_sha"] = hashlib.sha256(json.dumps([r.output_ids for r in reqs]).encode()).hexdigest()[:16]
            row["per_item"] = [[len(r.output_ids), int(gsm_correct(t, it["answer"]))] for r, t, it in zip(reqs, texts, items)]
            report["cells"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "per_item"}), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1))
        report["profiles"] = []
        for cell in filter(None, args.profile.split(",")):
            b, d = cell.split(":")
            row = profile_cell(ex, prompts, int(b), d if d == "plain" else int(d))
            report["profiles"].append(row)
            print(json.dumps(row), flush=True)
            args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
