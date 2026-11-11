# SPDX-License-Identifier: Apache-2.0
"""Decode speed against concurrency at a fixed long context, prefill excluded: the steady-state speculative round
with every request active on a populated 10,000-token context (KV cache and recurrent state), on the frozen
benchmark prompts of `hibana.benchmark_workloads` (code, design, reasoning).

For each batch size B, the script prefills B requests untimed and decodes them round by round: it skips `--warmup`
rounds, then times the next `--rounds` while every request is still active. The B prompts are the three workloads in
turn with request indices 0, 1, ... (the index in each prompt's opening line, "Review task i"), so concurrent
requests share no prefix; each point covers at least `--min-prompts` of them in groups of B. Per point: ms per round,
accepted tokens per request per round, and decode tokens per second (tokens emitted in the timed rounds / their wall
time).

    python3 -m profiling.decode_curve --out fp32.json
    SHARAR_SSM_STATE=int8 python3 -m profiling.decode_curve --out int8.json

The defaults are the `throughput` serving profile's: DFlash2 with 15 drafts in FP8, accepted-state reconstruction,
relaxed greedy acceptance 3:2.0, up to 16 requests. The 3-bit decode copies follow SHARAR_W3_DECODE and
SHARAR_W3_CHECKPOINT; the output records what was in effect. To compare a change, run control and candidate in
separate processes, alternating (A B B A), on an otherwise idle GPU, and compare per point.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from hibana import benchmark_workloads
from hibana import target as targets
from hibana.cache.spec_budget import SpecConfig
from hibana.engine.config import current
from hibana.engine.executor import Executor
from hibana.engine.sampler import SamplingParams
from hibana.tokenizer import SnapshotTokenizer


def benchmark_prompts(encode, count: int, length: int = 10000) -> list[list[int]]:
    """`count` distinct AIPerf prompts: code, design, reasoning with request index 0, then index 1, ... (index 0 is the
    benchmark's own request)."""
    out = []
    for i in range(-(-count // len(benchmark_workloads.WORKLOADS))):
        def indexed(text, i=i):
            return encode(text.replace("Review task 0.", f"Review task {i}.").replace("Independent review 0.", f"Independent review {i}."))
        out += [benchmark_workloads.materialize(indexed, case, length) for case in benchmark_workloads.WORKLOADS]
    return out[:count]


def _groups(min_prompts: int, batch: int) -> int:
    return max(1, -(-min_prompts // batch))


def point(ex: Executor, prompts: list[list[int]], batch: int, args, params) -> dict:
    """Steady-state rounds at one batch size over at least args.min_prompts prompts."""
    groups = _groups(args.min_prompts, batch)
    times, emitted, order = [], [], 0
    for _ in range(groups):
        ex.flush_cache()
        reqs = [ex.admit(prompts[order + i], params) for i in range(batch)]
        order += batch
        for r in reqs:
            ex.prefill(r)
        for step in range(args.warmup + args.rounds):
            before = sum(len(r.output_ids) for r in reqs)
            torch.cuda.synchronize()
            started = time.perf_counter()
            ex.decode_step(reqs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if any(r.finished for r in reqs):
                raise RuntimeError(f"a request finished during round {step}: raise --output-tokens")
            if step >= args.warmup:
                times.append(elapsed)
                emitted.append(sum(len(r.output_ids) for r in reqs) - before)
        for r in reqs:
            ex.release(r)
    return {"batch": batch, "prompts": groups * batch, "rounds": len(times), "ms_per_round": 1e3 * sum(times) / len(times),
            "tokens_per_request_round": sum(emitted) / len(emitted) / batch, "tok_s": sum(emitted) / sum(times)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="1,2,4,8,16", help="comma list of concurrent requests")
    ap.add_argument("--rounds", type=int, default=30, help="timed rounds per prompt group")
    ap.add_argument("--warmup", type=int, default=4, help="untimed rounds after prefill")
    ap.add_argument("--min-prompts", type=int, default=16, help="distinct prompts each point covers (rotating)")
    ap.add_argument("--accept", default="3:2.0", help="relaxed greedy acceptance K:M ('1:0' is exact)")
    ap.add_argument("--proposer", default="dflash2")
    ap.add_argument("--spec", type=int, default=15)
    ap.add_argument("--draft-precision", default="fp8")
    ap.add_argument("--output-tokens", type=int, default=2048, help="max new tokens per request (only bounds the run)")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    top_k, margin = args.accept.split(":")
    params = SamplingParams(max_new_tokens=args.output_tokens, ignore_eos=True, accept_top_k=int(top_k), accept_margin=float(margin))

    cfg = current()
    target = targets.get(cfg.target)
    snap = target.snapshot_dir()
    needed = max(_groups(args.min_prompts, b) * b for b in batches)
    prompts = benchmark_prompts(SnapshotTokenizer(snap).encode, needed)
    model = targets.load_model(target, snap)
    context = len(prompts[0]) + args.output_tokens
    ex = Executor(model, max_batch=max(batches), max_context=context, log=print,
                  spec_config=SpecConfig(args.proposer, args.spec, "reconstruct", args.draft_precision, None))
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    out = {"target": target.name, "ssm_state": ex.ssm_format.describe(), "w3_decode": cfg.w3_decode,
           "w3_checkpoint": cfg.w3_checkpoint, "proposer": args.proposer, "spec": args.spec,
           "draft_precision": args.draft_precision, "accept": args.accept, "prompt_tokens": len(prompts[0]),
           "prompt_set": f"{benchmark_workloads.VERSION} code/design/reasoning, request indices 0-{(len(prompts) - 1) // 3}", "points": []}
    for b in batches:
        ex.flush_cache()
        # Compiles anything new at this batch size; short prompts, since only the shapes matter.
        ex.generate_batch([(prompts[i][:512], SamplingParams(max_new_tokens=32, ignore_eos=True)) for i in range(b)])
        p = point(ex, prompts, b, args, params)
        budget = ex.spec.stats().get("budget") if ex.spec is not None else None
        if budget is not None:
            p["budget"] = budget  # what adaptive verification learned so far
        out["points"].append(p)
        print(f"[{out['ssm_state']}] batch {b:2d}: {p['tok_s']:.1f} tok/s, {p['ms_per_round']:.1f} ms/round, "
              f"{p['tokens_per_request_round']:.2f} tokens/request/round ({p['rounds']} rounds, {p['prompts']} prompts)", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
