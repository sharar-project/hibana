"""Decode-round profiling target: prefill 1-4 frozen prompts, then profile verification.

Run under a profiler inside the runtime image, as root with CAP_SYS_ADMIN when
hardware counters are wanted (the host driver restricts them to administrators):

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o /proof/nsys-bs1 \
        python3 -m bench.gb10_round_profile --batch 1 --rounds 20
    ncu --profile-from-start off --graph-profiling node --target-processes all \
        --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --csv --log-file /proof/ncu-bs1.csv python3 -m bench.gb10_round_profile --batch 1 --rounds 1

The profiled window is bracketed by torch.cuda.profiler.start()/stop(): it
begins after the prompts are prefilled and the first `--skip` rounds have run,
and covers exactly `--rounds` decode rounds of the whole batch. Everything else
(load, capture, warmup, prefill) is outside the window. An unprofiled baseline
precedes capture; CUDA-event phase diagnostics follow it. Phase spans overlap
and host intervals include waits for earlier GPU work, so they must not be
summed or interpreted as removable CPU overhead. Greedy, exact acceptance,
FP8 DFlash2 at k=15, cold prefix cache, the frozen family prompts. Use --draft
and SHARAR_W3_CHECKPOINT to profile the artifacts actually served.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time
from pathlib import Path

import torch

from bench.support.evidence import clear
from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, default=Path("/proof/inputs/task-inputs.json"))
    ap.add_argument("--tasks", default="ring_buffer,deadline_queue,build_graph,stream_parser", help="prompts for the batch rows, in order")
    ap.add_argument("--batch", type=int, default=1, choices=(1, 2, 3, 4))
    ap.add_argument("--skip", type=int, default=8, help="rounds run before the window opens (steady state)")
    ap.add_argument("--rounds", type=int, default=20, help="rounds inside the profiled window")
    ap.add_argument("--draft-precision", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--draft", help="trained draft directory; omitted uses the pinned publisher checkpoint")
    ap.add_argument("--context", type=int, default=10000, help="prefix length from each frozen prompt")
    ap.add_argument("--unprofiled-rounds", type=int, default=32, help="baseline window before profiler capture")
    ap.add_argument("--phase-rounds", type=int, default=8, help="event-instrumented rounds after profiler capture")
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--num-pages", type=int, default=60000)
    ap.add_argument("--num-slots", type=int, default=24)
    ap.add_argument("--summary", type=Path, default=None, help="write round timings here")
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs: every kernel is a plain launch (Nsight Compute per-kernel replay needs this)")
    args = ap.parse_args()

    frozen = json.loads(args.inputs.read_text())
    wanted = [t.strip() for t in args.tasks.split(",")][:args.batch]
    prompts = [next(t["input_ids"] for t in frozen["tasks"] if t["id"] == w)[:args.context] for w in wanted]
    if len(prompts) != args.batch or args.context < 1 or any(len(ids) != args.context for ids in prompts):
        ap.error("each batch row needs a frozen prompt covering --context")
    model = Qwen38(config.snapshot_dir(), log=print)
    ex = Executor(model, num_pages=args.num_pages, num_slots=args.num_slots, max_batch=4, log=print)
    ex.enable_spec(args.depth, proposer="dflash2", state_strategy="reconstruct", draft_precision=args.draft_precision,
                   draft_dir=args.draft, log=print)
    if args.eager:
        ex.use_graphs = False
        print("  eager mode: graphs not captured", flush=True)
    else:
        ex.capture_graphs(log=print)
    ex.warmup(log=print)
    clear(ex)
    reqs = [ex.admit(ids, SamplingParams(max_new_tokens=4000, ignore_eos=True)) for ids in prompts]
    for req in reqs:
        while not req.prefill_done:
            ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
    torch.cuda.synchronize()
    for _ in range(args.skip):
        ex.decode_step(reqs)
    torch.cuda.synchronize()
    unprofiled = []
    t0 = time.perf_counter()
    for _ in range(args.unprofiled_rounds):
        before = [len(r.output_ids) for r in reqs]
        ex.decode_step(reqs)
        unprofiled.append([len(r.output_ids) - b for r, b in zip(reqs, before)])
    torch.cuda.synchronize()
    unprofiled_wall = time.perf_counter() - t0
    timings = []
    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    for _ in range(args.rounds):
        before = [len(r.output_ids) for r in reqs]
        ex.decode_step(reqs)
        timings.append([len(r.output_ids) - b for r, b in zip(reqs, before)])
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    torch.cuda.profiler.stop()
    from bench.verify_cost import phases, summarize_phases
    records = defaultdict(list)
    with phases(ex, records):
        for _ in range(args.phase_rounds):
            ex.decode_step(reqs)
    torch.cuda.synchronize()
    output_ids = [r.output_ids[:] for r in reqs]
    for req in reqs:
        req.cacheable = False
        ex.release(req)
    summary = {"batch": args.batch, "tasks": wanted, "skip": args.skip, "rounds": args.rounds, "window_seconds": wall,
               "round_ms": 1000 * wall / args.rounds, "emitted_per_round": timings,
               "tokens_per_round": sum(map(sum, timings)) / args.rounds, "draft_precision": args.draft_precision,
               "draft": args.draft, "w3_target_identity": getattr(model, "w3_identity", None),
               "context": args.context, "unprofiled_rounds": args.unprofiled_rounds,
               "unprofiled_seconds": unprofiled_wall, "unprofiled_emitted": unprofiled,
               "phases": summarize_phases(records), "output_ids": output_ids}
    print(json.dumps(summary), flush=True)
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
