# SPDX-License-Identifier: Apache-2.0
"""One Nemotron MTP draft step on the head alone: graph-replay time, kernels per step by part of the step, a bitwise
check of the step's outputs across code versions, and draft acceptance on recorded target rows.

Loads only what the draft reads (the MTP head, the embedding and the FP8 lm_head, about 3.5 GB and under a minute),
fills the draft KV for a context of `--context` tokens and chains `--steps` draft steps per batch size, eagerly and
through the draft graphs (`proposers/mtp_graphs.py`) the server replays.

    python3 -m profiling.nemotron_draft_step [--precision bf16|fp8] [--moe bf16|fp8] --save before.pt
    python3 -m profiling.nemotron_draft_step --check before.pt        # after a change: bitwise per step
    python3 -m profiling.nemotron_draft_step --accept rows.pt         # acceptance on recorded target rows

SHARAR_DRAFT_HEAD / SHARAR_DRAFT_HEAD_RESCORE / SHARAR_DRAFT_VOCAB choose the draft head as the server does.

`--breakdown` attributes every kernel of a replayed step to the part of the step that launches it (fuse: embedding,
enorm, hnorm and eh_proj; attention: norm_0, q|k|v, KV write, attention, o_proj; moe: norm_1 and the latent MoE; final
norm; head: lm_head and argmax), matching the replay's kernels in launch order to an eager step's, whose parts are
marked with profiler ranges.

`--accept` reads what `--record` saved from the full model (`record()`: each sequence's token ids and the target's
norm_f rows from one prefill over prompt + greedy continuation) and reports, per sequence and overall: teacher-forced
top-1 agreement per chain depth (the reference's measure, `NemotronReference.mtp_chain`: step 1 from the true token and
the target's row, later steps from the drafted token and the head's own d), the draft head's containment of the FP8
argmax in its shortlist, and the tokens per round an exact-acceptance decode at depth `--steps` would emit (each round
drafts from the committed position with the draft KV the engine keeps: target-conditioned rows for the committed
prefix, the chain's own rows for the round's earlier drafts). Recording needs the full model (67 GiB: hold the GPU lock).
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from sharar.model.nemotron.config import PREFIX
from sharar.target import NEMOTRON
from sharar.weights.safetensors import SnapshotWeights

BACKBONE = f"{PREFIX}.backbone"
PARTS = ("fuse", "attention", "moe", "final_norm", "head")


def load(device, max_batch: int, k: int, max_context: int, precision: str | None = None, moe: str | None = None,
         moe_topk: int | None = None):
    from sharar.engine.config import current
    from sharar.model.nemotron.mtp import NemotronMtpHead
    from sharar.proposers.mtp import MtpProposer
    from sharar.weights.modelopt import load_fp8_linear
    t = NEMOTRON
    w = SnapshotWeights(t.snapshot_dir())
    try:
        model = SimpleNamespace(target=t, embed=w.get(f"{BACKBONE}.embeddings.weight").to(device),
                                lm_head=load_fp8_linear(w, [f"{PREFIX}.lm_head"], device))
        head = NemotronMtpHead(w, device, model, max_batch * (max_context + 16) + 64, precision=precision,
                               moe_precision=moe, padding_rows=max_batch, moe_topk=moe_topk)
    finally:
        w.close()
    cfg = current()
    if cfg.draft_head != "fp8" or cfg.draft_vocab:
        from sharar.proposers.draft_vocab import draft_vocab
        ids = draft_vocab(t.snapshot_dir(), cfg.draft_vocab, head.lm_head.out_features) if cfg.draft_vocab else None
        head.set_draft_head(ids, cfg.draft_head, cfg.draft_head_rescore)
    context = SimpleNamespace(device=device, max_batch=max_batch, max_context=max_context)
    return head, MtpProposer(head, context, k)


def fill(head, b: int, context: int, gen: torch.Generator, scale: float):
    """Draft KV for `b` sequences of `context` tokens on disjoint pages; returns each sequence's pages."""
    device = head.device
    pages = []
    for i in range(b):
        p = torch.arange(1 + i * (context + 16), 1 + i * (context + 16) + context + 16, dtype=torch.int32, device=device)
        toks = torch.randint(0, head.lm_head.out_features, (context - 1,), generator=gen).to(device)
        hid = (torch.randn(context - 1, head.hidden_width, generator=gen) * scale).to(torch.bfloat16).to(device)
        head.write_kv(toks, hid, torch.arange(1, context, dtype=torch.int64, device=device), p[1:context])
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


def _kernels(prof) -> list[tuple[str, float, float]]:
    """(name, start us, duration us) of every CUDA kernel in launch order (memcpys and memsets included)."""
    rows = [(e.name, e.time_range.start, e.time_range.end - e.time_range.start) for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0]
    return sorted(rows, key=lambda r: r[1])


def breakdown(head, proposer, b: int, context: int, pages, tokens, hidden, iters: int = 10) -> dict:
    """Per part of one step: kernels and kernel time (from graph replays), matched to an eager step's parts."""
    from sharar.proposers.mtp import draft_argmax
    positions = torch.full((b,), context, dtype=torch.int64, device=tokens.device)
    write = torch.stack([p[context] for p in pages]).contiguous()
    kv = [torch.cat([p[1:context], p[context:context + 1]]) for p in pages]
    head.plan([1] * b, kv)

    def eager():
        with record_function("part:fuse"):
            u = head._fuse(tokens, hidden)
        with record_function("part:attention"):
            a = head._attention(head._attention_input(u), positions, write, head._attn)
        with record_function("part:moe"):
            x, resid = head.moe_norm.fused_add(a, u)
            m = head.moe(x, None)
        with record_function("part:final_norm"):
            d = head._final(m, resid)
        with record_function("part:head"):
            draft_argmax(head, d)
    eager()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        eager()
        torch.cuda.synchronize()
    # each kernel's part: the innermost "part:" range whose CPU interval holds the kernel's launch
    ranges = [(e.name[5:], e.time_range.start, e.time_range.end) for e in prof.events() if e.name.startswith("part:")]
    launches = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0:
            parent = e.cpu_parent
            t0 = parent.time_range.start if parent is not None else e.time_range.start
            part = next((n for n, s, f in ranges if s <= t0 <= f), "?")
            launches.append((e.time_range.start, e.name, part))
    eager_parts = [(n, p) for _, n, p in sorted(launches)]
    g = proposer.graphs._graphs[proposer.graphs.pick(b)]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
    ks = _kernels(prof)
    per = len(ks) // iters
    # align the replay's kernels with the eager step's by name (eager launches a few more: host-to-device copies, the
    # plan's buffers); a replayed kernel takes its aligned eager kernel's part
    import difflib
    replay_names, eager_names = [n for n, _, _ in ks[:per]], [n for n, _ in eager_parts]
    part_of = ["?"] * per
    for blk in difflib.SequenceMatcher(None, replay_names, eager_names, autojunk=False).get_matching_blocks():
        for o in range(blk.size):
            part_of[blk.a + o] = eager_parts[blk.b + o][1]
    names_match = "?" not in part_of
    parts: dict[str, list] = {p: [0, 0.0] for p in PARTS}
    kernels: dict[str, list] = {}
    for i, (name, _, us) in enumerate(ks):
        part = part_of[i % per]
        parts.setdefault(part, [0, 0.0])
        parts[part][0] += 1
        parts[part][1] += us
        key = f"{part}: {name[:80]}"
        c, t = kernels.get(key, (0, 0.0))
        kernels[key] = (c + 1, t + us)
    return {"kernels_per_step": per, "eager_kernels": len(eager_parts), "matched": names_match,
            "kernel_ms_per_step": round(sum(us for _, _, us in ks) / iters / 1000, 4),
            "parts": {p: {"kernels": round(c / iters, 1), "ms": round(us / iters / 1000, 4)} for p, (c, us) in parts.items()},
            "kernels": {k: [round(c / iters, 1), round(us / iters / 1000, 4)] for k, (c, us) in sorted(kernels.items(), key=lambda kv: -kv[1][1])}}


# ---- acceptance on recorded target rows -----------------------------------------------------------------------------
@torch.inference_mode()
def accept(head, data: dict, depth: int, device) -> dict:
    """Teacher-forced top-1 per depth, shortlist containment and simulated exact-acceptance tokens per round."""
    from sharar.proposers.mtp import draft_argmax
    full = head.lm_head  # the FP8 head the target verifies with
    shortlist = getattr(head.draft_head, "rescore", 0)
    out, agg = {}, {"tf": [[0, 0] for _ in range(depth)], "contain": [0, 0], "sims": {}, "argmax_vs_fp8": [0, 0]}
    for name, rec in data["sequences"].items():
        ids = rec["ids"].to(device).long()
        norm = rec["norm"].to(device)  # [T, H]: the target's norm_f row at every position
        start = int(rec["prompt_len"])  # score only the generated part (prompt rows still feed the draft KV)
        t_len = ids.numel()
        pages = torch.arange(1, t_len + depth + 2, dtype=torch.int32, device=device)  # page of position p: p + 1
        # draft KV of every position >= 1, target-conditioned: (token at p, target row at p - 1)
        head.write_kv(ids[1:], norm[:-1], torch.arange(1, t_len, device=device), pages[1:t_len])
        tf = [[0, 0] for _ in range(depth)]
        contain = [0, 0]
        argmax_agree = [0, 0]
        # teacher-forced chains from every generated base position i: step 1 reads (ids[i+1], norm[i]) at
        # position i+1 and predicts ids[i+2]; step j reads its own previous draft and d at position i+j
        bases = list(range(start - 1, t_len - 2))
        for chunk in range(0, len(bases), 64):
            rows = bases[chunk:chunk + 64]
            b = len(rows)
            base = torch.tensor(rows, device=device)
            tok, hid = ids[base + 1], norm[base]
            scratch = torch.arange(t_len + 1, t_len + 1 + b * depth, dtype=torch.int32, device=device).view(b, depth) + 16
            for j in range(depth):
                pos = base + 1 + j
                kv = [torch.cat([pages[1:int(r) + 1], scratch[i, :j + 1]]) for i, r in enumerate(rows)]  # positions 1..r, then the chain's own
                d = head.forward(tok, hid, pos, scratch[:, j].contiguous(), [1] * b, kv)
                pick = draft_argmax(head, d)
                target = ids[(base + 2 + j).clamp(max=t_len - 1)]
                valid = base + 2 + j < t_len
                hit = (pick == target) & valid
                tf[j][0] += int(hit.sum())
                tf[j][1] += int(valid.sum())
                if shortlist:  # does the low-bit top `rescore` hold what the FP8 head would pick?
                    hin = head.head_input(d)
                    fp8 = full(hin).argmax(-1)
                    logits = head.draft_head._logits(hin.reshape(-1, hin.shape[-1]))
                    cand = head.draft_head._global(logits.topk(shortlist, dim=-1).indices)
                    contain[0] += int((cand == fp8[:, None]).any(-1).sum())
                    contain[1] += b
                    argmax_agree[0] += int((pick == fp8).sum())
                    argmax_agree[1] += b
                tok, hid = pick, d
        # simulated exact-acceptance decodes at depth 3 and `depth` over the generated part
        sim_pages = torch.arange(t_len + 2 + 64 * depth + 64, t_len + 2 + 64 * depth + 64 + depth + 1, dtype=torch.int32, device=device)
        sims = {}
        for sim_depth in sorted({min(3, depth), depth}):
            sims[sim_depth] = _simulate(head, ids, norm, pages, sim_pages, start, sim_depth, device)
        out[name] = {"tf_top1": [round(a / max(c, 1), 4) for a, c in tf],
                     "tokens_per_round": {d: round(e / max(r, 1), 3) for d, (e, r) in sims.items()}, "positions": len(bases)}
        if "golden_mtp_logits" in rec:  # the reference's step-1 logits at the prompt's last positions
            gpos = rec["golden_positions"].to(device).long()  # reference row i = (ids[i+1], h_i) = this head's position i+1
            kv = [pages[1:int(i) + 2] for i in gpos]
            d = head.forward(ids[gpos + 1], norm[gpos], gpos + 1, pages[gpos + 1].contiguous(), [1] * len(kv), kv)
            mine, ref = full(head.head_input(d)).float(), rec["golden_mtp_logits"].to(device).float()
            out[name]["golden_step1_top1_agree"] = round(float((mine.argmax(-1) == ref.argmax(-1)).float().mean()), 4)
            out[name]["golden_step1_logit_cos_min"] = round(float(torch.nn.functional.cosine_similarity(mine, ref, dim=-1).min()), 5)
        if shortlist:
            out[name]["shortlist_holds_fp8_argmax"] = round(contain[0] / max(contain[1], 1), 4)
        for j in range(depth):
            agg["tf"][j][0] += tf[j][0]
            agg["tf"][j][1] += tf[j][1]
        agg["contain"][0] += contain[0]
        agg["contain"][1] += contain[1]
        agg["argmax_vs_fp8"][0] += argmax_agree[0]
        agg["argmax_vs_fp8"][1] += argmax_agree[1]
        for d, (e, r) in sims.items():
            agg["sims"].setdefault(d, [0, 0])
            agg["sims"][d][0] += e
            agg["sims"][d][1] += r
    out["all"] = {"tf_top1": [round(a / max(c, 1), 4) for a, c in agg["tf"]], "tf_counts": [c for _, c in agg["tf"]],
                  "tokens_per_round": {d: round(e / max(r, 1), 3) for d, (e, r) in agg["sims"].items()}}
    if shortlist:
        out["all"]["shortlist_holds_fp8_argmax"] = round(agg["contain"][0] / max(agg["contain"][1], 1), 4)
        out["all"]["pick_equals_fp8_argmax"] = round(agg["argmax_vs_fp8"][0] / max(agg["argmax_vs_fp8"][1], 1), 4)
    return out


def _simulate(head, ids, norm, pages, sim_pages, start: int, depth: int, device) -> tuple[int, int]:
    """(emitted tokens, rounds) of an exact-acceptance decode drafting `depth` a round over the generated part."""
    from sharar.proposers.mtp import draft_argmax
    t_len = ids.numel()
    p, emitted, rounds = start - 1, 0, 0  # committed through position p; the pending input is ids[p + 1]
    while p + 2 < t_len:
        tok, hid = ids[p + 1:p + 2], norm[p:p + 1]
        acc = 0
        for j in range(depth):
            kv = [torch.cat([pages[1:p + 1], sim_pages[:j + 1]])]  # committed positions 1..p, then this round's rows
            d = head.forward(tok, hid, torch.tensor([p + 1 + j], device=device), sim_pages[j:j + 1], [1], kv)
            pick = draft_argmax(head, d)
            if acc < j or p + 2 + j >= t_len or int(pick) != int(ids[p + 2 + j]):
                break  # the round's later drafts are never verified past the first miss
            acc += 1
            tok, hid = pick, d
        n = min(acc + 1, t_len - (p + 2))  # accepted drafts plus the target's own token
        emitted += n
        rounds += 1
        p += n
    return emitted, rounds


TEMPLATE = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think></think>"
# The first four are `profiling/nemotron_bench.py`'s prompts (its tokens/round); the rest widen the sample.
PROMPTS = ("Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short usage example.",
           "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
           "Explain step by step how to compute 37 * 48 mentally, then verify with long multiplication.",
           "Describe the causes of the French Revolution in five bullet points, with one sentence each.",
           "Write a SQL schema for a library (books, members, loans) and three queries over it, with comments.",
           "Translate into French, German and Spanish: 'The meeting moved to Thursday because the projector broke.' "
           "Then explain one grammatical difference between the three translations.",
           "Solve: a tank fills in 6 hours through pipe A and empties in 9 hours through pipe B. Both open, how long "
           "to fill it? Show the algebra.",
           "Write a JSON object describing three fictional planets (name, radius_km, moons, atmosphere), then a "
           "TypeScript interface for it.")
GOLDEN = "/home/brian/Desktop/research/nemotron-logs/golden"


def record(path: str, new: int) -> None:
    """Full model (hold the GPU lock): greedy continuations of PROMPTS (`new` tokens) and of the golden prompts (128),
    then each whole sequence's norm_f rows from one stand-alone prefill; saved for `--accept`."""
    from pathlib import Path
    from sharar import target as targets
    from sharar.engine import config
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    from sharar.tokenizer import SnapshotTokenizer
    t = NEMOTRON
    cfg = config.current().replace(target="nemotron", hf_cache=str(Path.home() / ".cache/huggingface/hub"), w3_decode=False,
                                   w3_checkpoint=None)
    with config.using(cfg):
        model = targets.load_model(t, t.snapshot_dir(), log=print)
        tok = SnapshotTokenizer(t.snapshot_dir())
        items = {f"p{i}": (tok.encode(TEMPLATE.format(p)), new, None) for i, p in enumerate(PROMPTS)}
        for k in range(4):
            g = torch.load(Path(GOLDEN) / f"prompt{k}.pt")
            items[f"golden{k}"] = (g["ids"].tolist(), 128, g)
        ex = Executor(model, log=None, max_batch=4, max_context=8192)
        ex.capture_graphs(log=None)
        names = list(items)
        sequences = {}
        for i in range(0, len(names), 4):
            group = names[i:i + 4]
            reqs = ex.generate_batch([(items[n][0], SamplingParams(max_new_tokens=items[n][1], ignore_eos=True)) for n in group])
            for n, r in zip(group, reqs):
                ids = list(items[n][0]) + list(r.output_ids)
                norm = model.prefill(torch.tensor(ids, dtype=torch.int64, device="cuda"))["norm"].to(torch.bfloat16).cpu()
                rec = {"ids": torch.tensor(ids, dtype=torch.int32), "norm": norm, "prompt_len": len(items[n][0])}
                g = items[n][2]
                if g is not None:
                    rec["golden_positions"] = g["positions"]
                    rec["golden_mtp_logits"] = g["mtp_logits"]
                    ref = g["greedy"].tolist()
                    rec["golden_greedy_prefix_match"] = next((j for j, (a, b) in enumerate(zip(r.output_ids, ref)) if a != b), len(ref))
                sequences[n] = rec
                print(n, len(ids), "tokens", flush=True)
            ex.flush_cache()
        torch.save({"engine": "Sharar", "sequences": sequences}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=600)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scale", type=float, default=1.0, help="hidden row scale (norm_f rows are about unit RMS)")
    ap.add_argument("--precision", choices=("bf16", "fp8"), default=None, help="eh_proj and attention projections")
    ap.add_argument("--moe", choices=("bf16", "fp8"), default=None, help="the MoE's router|fc1 and fc2 latent projections")
    ap.add_argument("--moe-topk", type=int, default=None, help="experts per row in the draft's MoE (default: 22)")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--check")
    ap.add_argument("--accept", help="recorded target rows (`--record` output)")
    ap.add_argument("--record", help="full model: save recorded target rows here (hold the GPU lock)")
    ap.add_argument("--new", type=int, default=384)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.record:
        return record(args.record, args.new)
    device = torch.device("cuda")
    batches = [int(b) for b in args.batches.split(",")]
    started = time.perf_counter()
    data = torch.load(args.accept) if args.accept else None
    if data is not None:  # one pool page per position, then the chains' scratch pages
        longest = max(int(r["ids"].numel()) for r in data["sequences"].values())
        batches, args.context = [1], longest + 64 * args.steps + 64
    head, proposer = load(device, max(batches), args.steps, args.context + 64, args.precision, args.moe, args.moe_topk)
    print(f"loaded in {time.perf_counter() - started:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
          f"head bytes {head.bytes / 1e6:.0f} MB", flush=True)
    from sharar.engine.config import current
    report: dict = {"engine": "Sharar", "precision": head.eh.weight.dtype.__repr__(), "moe": args.moe, "moe_topk": head.moe.draft_topk,
                    "draft_head": current().draft_head, "rescore": current().draft_head_rescore, "draft_vocab": current().draft_vocab}
    if args.accept:
        report["accept"] = accept(head, data, args.steps, device)
        print(json.dumps(report["accept"], indent=1), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
        return
    proposer.capture_graphs(log=None)
    saved = {}
    for b in batches:
        gen = torch.Generator().manual_seed(1234 + b)
        pages = fill(head, b, args.context, gen, args.scale)
        tokens = torch.randint(0, head.lm_head.out_features, (b,), generator=gen).to(device)
        hidden = (torch.randn(b, head.hidden_width, generator=gen) * args.scale).to(torch.bfloat16).to(device)
        eager = chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=False)
        graphed = chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=True)
        saved[b] = {"eager": [(h.cpu(), t.cpu()) for h, t in eager], "graphed": [(h.cpu(), t.cpu()) for h, t in graphed]}
        agree = sum(torch.equal(a[1], g[1]) for a, g in zip(eager, graphed))
        for _ in range(5):
            chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters // 4):
            chain(proposer, b, args.context, pages, tokens, hidden, args.steps, graphs=True)
        torch.cuda.synchronize()
        chain_ms = 1000 * (time.perf_counter() - t0) / (args.iters // 4)
        g = proposer.graphs._graphs[proposer.graphs.pick(b)]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(args.iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        replay_ms = e0.elapsed_time(e1) / args.iters
        row = {"batch": b, "graph_replay_ms": round(replay_ms, 4), "draft_chain_ms": round(chain_ms, 3), "steps": args.steps,
               "eager_graphed_token_agree": f"{agree}/{args.steps}"}
        if args.breakdown:
            row.update(breakdown(head, proposer, b, args.context, pages, tokens, hidden))
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
                diff = [float((h0.float() - h1.float()).abs().max()) for (h0, _), (h1, _) in zip(ref[b][mode], saved[b][mode])]
                result[f"batch{b}-{mode}"] = {"bitwise_equal_steps": f"{args.steps - len(bad)}/{args.steps}", "max_abs_diff_per_step": diff}
        report["check"] = result
        print(json.dumps(result, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
