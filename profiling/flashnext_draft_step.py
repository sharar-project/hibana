# SPDX-License-Identifier: Apache-2.0
"""One Flash-Next MTP draft step on the head alone (#9): graph-replay time, kernels per step, and a bitwise check
of the step's outputs across code versions.

Loads only what the draft reads (the MTP layer, the embedding, the FP8 lm_head and its 3-bit draft copy, about
5 GB and under a minute), fills the draft KV for a context of `--context` tokens and chains `--steps` draft steps
per batch size, eagerly and through the draft graphs (`proposers/mtp_graphs.py`) the server replays.

    SHARAR_TARGET=flashnext python3 -m profiling.flashnext_draft_step --save before.pt
    SHARAR_TARGET=flashnext python3 -m profiling.flashnext_draft_step --check before.pt   # after a change

`--check` compares every step's hyper stream and draft token, eager and graphed, with the saved run (bitwise),
and reports the first mismatch. Inputs are seeded random token ids and hidden rows of the trunk's scale; the
weights are the checkpoint's.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

from sharar.model.flashnext.hyper import HyperConnection
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.flashnext.model import PREFIX
from sharar.model.rope import PartialRotary
from sharar.target import FLASHNEXT
from sharar.weights.safetensors import SnapshotWeights


def load(device, max_batch: int, k: int, max_context: int):
    from sharar.engine.config import current
    from sharar.model.flashnext.mtp import FlashNextMtpHead
    from sharar.proposers.mtp import MtpProposer
    t = FLASHNEXT
    snap = t.snapshot_dir()
    w = SnapshotWeights(snap)
    try:
        model = SimpleNamespace(target=t, precision=current().flashnext_trunk or "fp8",
                                embed=w.get(f"{PREFIX}.embed_tokens.weight").to(device),
                                rope=PartialRotary(t.head_dim, t.rotary_dim, t.rope_theta, t.max_position_embeddings, device),
                                lm_head=TrunkLinear.load(w, ["lm_head.weight"], device, current().flashnext_trunk or "fp8"),
                                mixer=HyperConnection.load(w, f"{PREFIX}.hyper_connection_mixer", t.hc_count, t.hidden_size, t.rms_norm_eps,
                                                           device, combine=False, precision=current().flashnext_mixers))
        head = FlashNextMtpHead(w, device, model, max_batch * (max_context + 16) + 64, padding_rows=max_batch)
    finally:
        w.close()
    cfg = current()
    if cfg.draft_head != "fp8":
        head.set_draft_head(None, cfg.draft_head, cfg.draft_head_rescore)
    context = SimpleNamespace(device=device, max_batch=max_batch, max_context=max_context)
    return head, MtpProposer(head, context, k)


def fill(head, b: int, context: int, gen: torch.Generator, scale: float):
    """Draft KV for `b` sequences of `context` tokens on disjoint pages; returns each sequence's pages."""
    device = head.device
    pages = []
    for i in range(b):
        p = torch.arange(1 + i * (context + 16), 1 + i * (context + 16) + context + 16, dtype=torch.int32, device=device)
        toks = torch.randint(0, 150000, (context - 1,), generator=gen, device="cpu").to(device)
        hid = (torch.randn(context - 1, head.hidden_width, generator=gen, device="cpu") * scale).to(torch.bfloat16).to(device)
        pos = torch.arange(1, context, dtype=torch.int64, device=device)
        head.write_kv(toks, hid, pos, p[1:context])
        pages.append(p)
    return pages


def chain(proposer, b: int, context: int, pages, tokens, hidden, steps: int, graphs: bool):
    """`steps` chained draft steps for b sequences at position `context`; returns [(hidden, token)] per step."""
    out = []
    positions = torch.full((b,), context, dtype=torch.int64, device=tokens.device)
    for j in range(steps):
        kv = [torch.cat([p[1:context], p[context:context + j + 1]]) for p in pages]
        write = torch.stack([p[context + j] for p in pages]).contiguous()
        hidden, tokens, _ = proposer._step(tokens, hidden, positions + j, write, kv, graphs)
        out.append((hidden.clone(), tokens.clone()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=600)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--scale", type=float, default=1.0, help="hidden row scale")
    ap.add_argument("--save")
    ap.add_argument("--check")
    ap.add_argument("--out")
    args = ap.parse_args()
    device = torch.device("cuda")
    batches = [int(b) for b in args.batches.split(",")]
    started = time.perf_counter()
    head, proposer = load(device, max(batches), args.steps, args.context + 64)
    print(f"loaded in {time.perf_counter() - started:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB; head bytes {head.bytes / 1e6:.0f} MB", flush=True)
    proposer.capture_graphs(log=print)
    report, saved = {}, {}
    for b in batches:
        gen = torch.Generator().manual_seed(1234 + b)
        pages = fill(head, b, args.context, gen, args.scale)
        tokens = torch.randint(0, 150000, (b,), generator=gen).to(device)
        hidden = (torch.randn(b, head.hidden_width, generator=gen) * args.scale).to(torch.bfloat16).to(device)
        eager = chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=False)
        graphed = chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=True)
        saved[b] = {"eager": [(h.cpu(), t.cpu()) for h, t in eager], "graphed": [(h.cpu(), t.cpu()) for h, t in graphed]}
        agree = sum(torch.equal(a[1], g[1]) for a, g in zip(eager, graphed))
        # timing: whole chained draft (k steps, host plan + replays), and the replays alone
        for _ in range(5):
            chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=True)
        torch.cuda.synchronize()
        chain_ms = 1000 * (time.perf_counter() - t0) / args.iters
        g = proposer.graphs._graphs[proposer.graphs.pick(b)]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(args.iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        replay_ms = e0.elapsed_time(e1) / args.iters
        torch.cuda.synchronize()
        e0.record()
        for _ in range(args.iters):
            chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=False)
        e1.record()
        torch.cuda.synchronize()
        eager_ms = e0.elapsed_time(e1) / args.iters
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
        names, count, busy = {}, 0, 0.0
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0 and not e.cpu_children:
                n = e.name[:90]
                c, ms = names.get(n, (0, 0.0))
                names[n] = (c + 1, ms + e.device_time_total / 1000)
                count += 1
                busy += e.device_time_total / 1000
        row = {"batch": b, "draft_chain_ms": round(chain_ms, 3), "graph_replay_ms": round(replay_ms, 3), "eager_chain_ms": round(eager_ms, 3),
               "kernels_per_step": round(count / 10, 1), "kernel_ms_per_step": round(busy / 10, 3), "eager_graphed_token_agree": f"{agree}/{args.steps}",
               "kernels": {n: [round(c / 10, 1), round(ms / 10, 4)] for n, (c, ms) in sorted(names.items(), key=lambda kv: -kv[1][1])}}
        report[b] = row
        print(json.dumps(row, indent=1), flush=True)
    if args.save:
        torch.save(saved, args.save)
    if args.check:
        ref = torch.load(args.check)
        result = {}
        for b in batches:
            for mode in ("eager", "graphed"):
                bad = [j for j, ((h0, t0), (h1, t1)) in enumerate(zip(ref[b][mode], saved[b][mode]))
                       if not (torch.equal(h0, h1) and torch.equal(t0, t1))]
                tok = [j for j, ((_, t0), (_, t1)) in enumerate(zip(ref[b][mode], saved[b][mode])) if not torch.equal(t0, t1)]
                diff = [float((h0.float() - h1.float()).abs().max()) for (h0, _), (h1, _) in zip(ref[b][mode], saved[b][mode])]
                result[f"batch{b}-{mode}"] = {"bitwise_equal_steps": f"{args.steps - len(bad)}/{args.steps}", "token_mismatch_steps": tok,
                                              "max_abs_diff_per_step": diff}
        report["check"] = result
        print(json.dumps(result, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
