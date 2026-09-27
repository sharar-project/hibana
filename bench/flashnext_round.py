# SPDX-License-Identifier: Apache-2.0
"""Where a Flash-Next speculative round spends its time (#25, #7, #9).

  1. Graphed MTP generation per prompt: tokens/round and the synchronized phase split
     (draft, verify, accept) from the spec decoder's timers.
  2. The same generation under the torch profiler: GPU kernel time by category against
     wall time; the remainder is launch, host work and idle gaps.
  3. The M-row curve: one continuation forward of M new tokens (M = 1..64) at a fixed
     context, GPU kernel time by category and distinct routed experts per MoE layer. A
     verification of s rows reads what an M = s forward reads; this is the cost a
     longer draft block pays.

    SHARAR_TARGET=flashnext python3 -m bench.flashnext_round --out round.json

`--gap-profile` (with `--batch`, `--accept`) and `--kernel-gaps` split a served round into GPU
work, GPU idle time between replays (host work the GPU waits for) and bubbles inside the replays.
"""
import argparse
import json
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from sharar.engine.run import _load
from sharar.engine.sampler import SamplingParams

PROMPTS = {
    "code": "Write a Python module implementing an LRU cache class with get, put, delete and resize, plus a thread-safe wrapper. Include docstrings and pytest unit tests.",
    "prose": "Write an essay of about 700 words on how the printing press changed European society between 1450 and 1600, covering religion, science, politics and literacy.",
    "math": "A loan of $250,000 at 6.5% annual interest is repaid monthly over 30 years. Derive the payment formula, compute the monthly payment, the total interest, and the balance after 10 years, showing all steps.",
}
EXTRA_PROMPTS = {  # a fourth request for batch 4 (kept apart: `m_curve` and draft_head.py read PROMPTS)
    "json": "Extract every person, company, date and amount from this text as JSON with a schema: On 3 March 2021 Maria Lopez of Acme Corp paid $12,400 to Jonas Berg (Nordic Tools AB) for 40 drills; the balance of $3,100 was settled on 9 April by Acme's CFO, Tom Reed.",
}
CATEGORIES = [  # first match wins
    ("moe", r"moe|expert|groupproblemshape|grouped|fused_moe|route|swizzle"),
    ("attention", r"flashinfer|BatchDecode|BatchPrefill|sparse_gqa|attention|qsa|score_kernel"),
    ("gdn", r"chunk_|recurrent|delta|gdn|fla_|gated_delta|conv1d|causal_conv"),
    ("gemv/gemm", r"gemv|gemm|scaled_mm|cutlass|cublas|sm\d+_xmma|Kernel2|nvjet|w3_"),
    ("norm/elementwise", r"norm|elementwise|vectorized|reduce|add|mul|copy|cat|index|gather|scatter|fill|sigmoid|silu|rope|quant|mix|combine|hyper"),
    ("sampling", r"argmax|topk|softmax|sort|multinomial"),
]


def category(name: str) -> str:
    low = name.lower()
    for cat, pattern in CATEGORIES:
        if re.search(pattern, low):
            return cat
    return "other"


def kernel_time(prof) -> tuple[dict, float, dict]:
    cats, total, names = defaultdict(float), 0.0, defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0 and not e.cpu_children:
            ms = e.device_time_total / 1000.0
            cats[category(e.name)] += ms
            names[e.name[:90]] += ms
            total += ms
    return dict(cats), total, dict(sorted(names.items(), key=lambda kv: -kv[1])[:15])


def chat_ids(tok, prompt: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def rounds(ex, tok, out, tokens: int, profiled: bool = True):
    ex.spec.timing = True
    for name, prompt in PROMPTS.items():
        ids = chat_ids(tok, prompt)
        params = SamplingParams(max_new_tokens=tokens, ignore_eos=True)
        ex.generate(ids, params)  # warm (prefix cache, graphs)
        ex.flush_cache()
        ex.spec.reset_stats()
        r = ex.generate(ids, params)
        s = ex.spec.stats()
        row = {"prompt": name, "tok_s": round((len(r.output_ids) - 1) / r.decode_seconds, 2), "decode_ms_per_token": round(1000 * r.decode_seconds / len(r.output_ids), 2),
               **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ("rounds", "tokens_per_round", "accept_rate", "round_ms",
                                                                                                 "draft_ms_per_round", "verify_ms_per_round", "accept_ms_per_round", "accept_split_ms")}}
        out["rounds"].append(row)
        print(json.dumps(row), flush=True)
    ex.spec.timing = False
    if not profiled:
        return
    ids = chat_ids(tok, PROMPTS["code"])
    params = SamplingParams(max_new_tokens=96, ignore_eos=True)
    ex.flush_cache()
    ex.generate(ids, params)
    ex.flush_cache()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        r = ex.generate(ids, params)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    cats, total, top = kernel_time(prof)
    row = {"what": "graphed MTP generation, 96 tokens incl. prefill", "wall_ms": round(1000 * wall, 1), "prefill_ms": round(1000 * r.prefill_seconds, 1),
           "kernel_ms": round(total, 1), "by_category_ms": {k: round(v, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}, "top_kernels_ms": {k: round(v, 2) for k, v in top.items()}}
    out["graphed_profile"] = row
    print(json.dumps(row)[:1500], flush=True)


def m_curve(ex, model, tok, out, context: int):
    """Continuation forwards of M rows at `context` cached tokens; GPU kernel time and distinct experts."""
    seen = []
    moe_layers = [layer.moe for layer in model.layers if getattr(layer, "moe", None) is not None]
    originals = [m.route for m in moe_layers]

    def hooked(m, orig):
        def route(x):
            ids, w = orig(x)
            seen.append(int(torch.unique(ids).numel()))
            return ids, w
        return route
    base = chat_ids(tok, " ".join(PROMPTS.values()) * 200)[:context]
    assert len(base) == context, len(base)
    for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        ids = base + chat_ids(tok, PROMPTS["code"])[:m] if m <= 40 else base + (chat_ids(tok, PROMPTS["code"]) * 3)[:m]
        req = ex.admit(ids, SamplingParams(max_new_tokens=1))
        ex.prefill(req, max_tokens=context)  # the cached context
        torch.cuda.synchronize()
        batch = ex._prepare_single_prefill(req, m)
        model.forward(batch.ids, batch.context)  # warm this shape
        ex.release(req)
        req = ex.admit(ids, SamplingParams(max_new_tokens=1))
        ex.prefill(req, max_tokens=context)
        torch.cuda.synchronize()
        batch = ex._prepare_single_prefill(req, m)
        for mo, orig in zip(moe_layers, originals):
            mo.route = hooked(mo, orig)
        seen.clear()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            model.forward(batch.ids, batch.context)
            t1.record()
            torch.cuda.synchronize()
        for mo, orig in zip(moe_layers, originals):
            mo.route = orig
        ex.release(req)
        ex.flush_cache()
        cats, total, _ = kernel_time(prof)
        row = {"m": m, "kernel_ms": round(total, 2), "eager_wall_ms": round(t0.elapsed_time(t1), 2),
               "experts_per_layer": round(sum(seen) / max(1, len(seen)), 1),
               "by_category_ms": {k: round(v, 2) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}}
        out["m_curve"].append(row)
        print(json.dumps(row), flush=True)


def verify_breakdown(ex, tok, out, tokens: int):
    """Synchronized host timers around each piece of a speculative round, CUDA events around graph replays."""
    acc = defaultdict(float)
    spec = ex.spec

    def timed(name, fn):
        def wrapper(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            torch.cuda.synchronize()
            acc[name] += time.perf_counter() - t0
            return r
        return wrapper

    def timed_replay(name, graph):
        orig = graph.replay

        def replay():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            a.record(); orig(); b.record()
            torch.cuda.synchronize()
            acc[name] += a.elapsed_time(b) / 1000
        graph.replay = replay
    patches = []
    for obj, name in ((ex.extras, "_verify_rows"), (ex.extras, "_stream_verify_rows"), (ex.extras, "fill"), (spec, "prepare"),
                      (spec, "verify"), (spec, "_accept")):
        orig = getattr(obj, name, None)
        if orig is None:
            continue
        patches.append((obj, name, orig))
        setattr(obj, name, timed(name, orig))
    for depth, graphs in spec.width_graphs.items():
        patches.append((graphs, "_plan", graphs._plan))
        graphs._plan = timed("verify plan()", graphs._plan)
        for g in graphs._graphs.values():
            timed_replay("verify graph replay (GPU)", g)
    if getattr(spec.proposer, "graphs", None) is not None:
        for g in spec.proposer.graphs._graphs.values():
            timed_replay("draft graph replays (GPU)", g)
    ids = chat_ids(tok, PROMPTS["code"])
    params = SamplingParams(max_new_tokens=tokens, ignore_eos=True)
    ex.generate(ids, params)
    ex.flush_cache()
    acc.clear(); spec.reset_stats()
    r = ex.generate(ids, params)
    rounds_n = max(1, spec.rounds)
    row = {"rounds": spec.rounds, "decode_ms_per_round": round(1000 * r.decode_seconds / rounds_n, 2),
           **{k: round(1000 * v / rounds_n, 2) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}}
    out["verify_breakdown"] = row
    print(json.dumps(row), flush=True)
    for obj, name, orig in patches:
        setattr(obj, name, orig)


def _params(tokens: int, accept: str | None) -> SamplingParams:
    """Greedy, fixed length; `accept` "K:M" is the served relaxed rule (the fast profile's 3:2.0)."""
    top_k, margin = (1, 0.0) if not accept else (int(accept.split(":")[0]), float(accept.split(":")[1]))
    return SamplingParams(max_new_tokens=tokens, ignore_eos=True, accept_top_k=top_k, accept_margin=margin)


def _batches(tok, batch: int) -> list[tuple[str, list[list[int]]]]:
    """Batch 1: each prompt alone. Batch B: one generation of B different prompts."""
    names = list(PROMPTS) + list(EXTRA_PROMPTS)
    text = {**PROMPTS, **EXTRA_PROMPTS}
    if batch == 1:
        return [(n, [chat_ids(tok, text[n])]) for n in PROMPTS]
    return [("+".join(names[:batch]), [chat_ids(tok, text[n]) for n in names[:batch]])]


def _replay_graphs(ex) -> list[tuple[str, torch.cuda.CUDAGraph]]:
    """Every graph a round replays, by kind: draft steps, verification, device acceptance, plain decode."""
    spec = ex.spec
    out = [("draft", g) for g in spec.proposer.graphs._graphs.values()]
    for graphs in spec.width_graphs.values():
        out += [("verify", g) for g in graphs._graphs.values()]
        accept = getattr(graphs, "accept", None)
        if accept is not None:
            out += [("accept", g) for g in accept._graphs.values()]
    if ex.graphs is not None:
        out += [("decode", g) for g in ex.graphs._graphs.values()]
    return out


def _evict_ple(model) -> None:
    """Drop the PLE table from the page cache (a new request's rows are mostly not cached)."""
    import os
    for fd in model.ple.table.fds:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _ple_stats(model) -> tuple[int, int]:
    """Rows read from the page cache and through the threads so far (PleTable counters)."""
    table = model.ple.table
    return getattr(table, "hits", 0), getattr(table, "misses", 0)


def gap_profile(ex, tok, out, tokens: int, batch: int = 1, accept: str | None = None, cold: bool = False):
    """GPU idle gaps inside a round, unsynchronized: CUDA events recorded before and after every graph
    replay (a before-event completes when the GPU reaches it, so after-draft -> before-verify is the time
    the GPU waited for the host between draft and verify, after-verify -> before-next-draft likewise).
    Batch 1 runs each prompt alone; batch B runs B prompts together. Sums over the runs, per round.
    `cold`: the PLE table leaves the page cache before each measured generation (the warm-up run
    otherwise leaves every row it reads cached)."""
    spec = ex.spec
    marks = []  # (kind, before, after)

    def instrument(graph, kind):
        orig = graph.replay

        def replay():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); orig(); b.record()
            marks.append((kind, a, b))
        graph.replay = replay
        return orig
    restore = [(g, instrument(g, kind)) for kind, g in _replay_graphs(ex)]
    params = _params(tokens, accept)
    gaps, busy = defaultdict(list), defaultdict(float)
    rounds_n = emitted = 0
    wall = span = 0.0
    per_prompt = []
    for name, prompts in _batches(tok, batch):
        items = [(ids, params) for ids in prompts]
        ex.generate_batch(items)  # warm (prefix cache, graphs)
        ex.flush_cache()
        if cold:
            _evict_ple(ex.model)
        marks.clear(); spec.reset_stats()
        ple0 = _ple_stats(ex.model)
        reqs = ex.generate_batch(items)
        torch.cuda.synchronize()
        ple = [a - b for a, b in zip(_ple_stats(ex.model), ple0)]
        n = max(1, spec.rounds)
        run_gaps = defaultdict(float)
        for (k1, a1, b1), (k2, a2, b2) in zip(marks, marks[1:]):
            g = b1.elapsed_time(a2)
            gaps[f"{k1} -> {k2}"].append(g)
            run_gaps[f"{k1} -> {k2}"] += g
        run_busy = defaultdict(float)
        for k, a, b in marks:
            run_busy[k] += a.elapsed_time(b)
        for k, v in run_busy.items():
            busy[k] += v
        run_span = marks[0][1].elapsed_time(marks[-1][2])
        rounds_n += spec.rounds
        emitted += spec.emitted
        wall += 1000 * reqs[0].decode_seconds
        span += run_span
        row = {"prompt": name, "rounds": spec.rounds, "tokens_per_round": round(spec.emitted / n / len(prompts), 3),
               "decode_wall_ms_per_round": round(1000 * reqs[0].decode_seconds / n, 2),
               "replay_ms_per_round": {k: round(v / n, 2) for k, v in run_busy.items()},
               "idle_ms_per_round": round(sum(run_gaps.values()) / n, 2),
               "gap_ms_per_round": {k: round(v / n, 2) for k, v in run_gaps.items()},
               "ple_rows_cached_missed": ple}
        per_prompt.append(row)
        print(json.dumps(row), flush=True)
    n = max(1, rounds_n)
    row = {"batch": batch, "ple": "cold" if cold else "warm", "accept": accept or "exact", "spec": spec.k, "rounds": rounds_n,
           "tokens_per_round": round(emitted / n / batch, 3), "decode_wall_ms_per_round": round(wall / n, 2),
           "event_span_ms_per_round": round(span / n, 2),
           "replay_ms_per_round": {k: round(v / n, 2) for k, v in busy.items()},
           "idle_ms_per_round": round(sum(sum(v) for v in gaps.values()) / n, 2),
           "gap_ms": {k: {"count": len(v), "mean": round(sum(v) / len(v), 3), "per_round": round(sum(v) / n, 2)} for k, v in gaps.items()},
           "per_prompt": per_prompt}
    out.setdefault("gap_profile", {})[f"batch{batch}" + ("-cold" if cold else "")] = row
    print(json.dumps({k: v for k, v in row.items() if k != "per_prompt"}), flush=True)
    for g, orig in restore:
        g.replay = orig


def kernel_gaps(ex, tok, out, tokens: int, batch: int = 1, accept: str | None = None):
    """Inside each graph replay: the span from its first to its last GPU activity against the time the
    GPU spends in kernels, memcpys and memsets (their union). The difference is the bubbles between
    the graph's nodes (launch latency on the device), which no host change removes. Torch profiler
    (CUPTI) trace; replays are tied to their activities by the graph launch's correlation id."""
    from torch.profiler import record_function

    def instrument(graph, kind):
        orig = graph.replay

        def replay():
            with record_function(f"replay:{kind}"):
                orig()
        graph.replay = replay
        return orig
    restore = [(g, instrument(g, kind)) for kind, g in _replay_graphs(ex)]
    params = _params(tokens, accept)
    name, prompts = _batches(tok, batch)[0]
    items = [(ids, params) for ids in prompts]
    ex.generate_batch(items)
    ex.flush_cache()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        ex.generate_batch(items)
        torch.cuda.synchronize()
    for g, orig in restore:
        g.replay = orig
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        events = json.load(open(f.name))["traceEvents"]
    ranges = sorted((e["ts"], e["ts"] + e["dur"], e["name"].split(":", 1)[1]) for e in events
                    if e.get("ph") == "X" and str(e.get("name", "")).startswith("replay:"))
    launches = sorted((e["ts"], e["args"]["correlation"]) for e in events
                      if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", ""))
    device = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            device[e["args"].get("correlation")].append((e["ts"], e["ts"] + e["dur"]))
    acc = defaultdict(lambda: defaultdict(float))
    li = 0
    for start, end, kind in ranges:
        while li < len(launches) and launches[li][0] < start:
            li += 1
        corr = [c for ts, c in launches[li:] if ts <= end][:1]
        acts = sorted(device.get(corr[0], [])) if corr else []
        if not acts:
            continue
        busy, cur_s, cur_e = 0.0, *acts[0]
        for s0, e0 in acts[1:]:
            if s0 > cur_e:
                busy += cur_e - cur_s
                cur_s, cur_e = s0, e0
            else:
                cur_e = max(cur_e, e0)
        busy += cur_e - cur_s
        span = max(e0 for _, e0 in acts) - acts[0][0]
        a = acc[kind]
        a["replays"] += 1
        a["activities"] += len(acts)
        a["span_ms"] += span / 1000
        a["busy_ms"] += busy / 1000
    row = {"batch": batch, "prompt": name, "per_replay": {
        kind: {"replays": int(a["replays"]), "activities": round(a["activities"] / a["replays"]),
               "span_ms": round(a["span_ms"] / a["replays"], 3), "busy_ms": round(a["busy_ms"] / a["replays"], 3),
               "bubbles_ms": round((a["span_ms"] - a["busy_ms"]) / a["replays"], 3),
               "bubble_us_per_boundary": round(1000 * (a["span_ms"] - a["busy_ms"]) / max(1, a["activities"] - a["replays"]), 2)}
        for kind, a in acc.items()}}
    out.setdefault("kernel_gaps", {})[f"batch{batch}"] = row
    print(json.dumps(row), flush=True)


def host_profile(ex, tok, out, tokens: int):
    """cProfile of graphed generation: where host time goes per round (cumulative, by function)."""
    import cProfile
    import io
    import pstats
    ids = chat_ids(tok, PROMPTS["code"])
    params = SamplingParams(max_new_tokens=tokens, ignore_eos=True)
    ex.generate(ids, params)
    ex.flush_cache()
    ex.spec.reset_stats()
    prof = cProfile.Profile()
    prof.enable()
    ex.generate(ids, params)
    prof.disable()
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats("tottime")
    stats.print_stats(40)
    text = buf.getvalue()
    out["host_profile"] = {"rounds": ex.spec.rounds, "text": text}
    print(f"rounds {ex.spec.rounds}", flush=True)
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--context", type=int, default=2000)
    ap.add_argument("--skip-rounds", action="store_true")
    ap.add_argument("--rounds-only", action="store_true", help="the per-prompt round split, without the profiled generation and the M curve")
    ap.add_argument("--only-breakdown", action="store_true")
    ap.add_argument("--host-profile", action="store_true")
    ap.add_argument("--gap-profile", action="store_true")
    ap.add_argument("--kernel-gaps", action="store_true", help="bubbles inside the graph replays (torch profiler)")
    ap.add_argument("--spec", type=int, default=4, help="MTP draft depth (config/flashnext.env serves 4)")
    ap.add_argument("--batch", default="1", help="--gap-profile / --kernel-gaps: requests per generation, a comma list (1,4)")
    ap.add_argument("--accept", default=None, help="relaxed acceptance K:M for every request (the fast profile: 3:2.0)")
    ap.add_argument("--ple-cold", action="store_true", help="--gap-profile also with the PLE table out of the page cache")
    args = ap.parse_args()
    model, ex, tok = _load(graphs=True, spec=args.spec)
    out = {"rounds": [], "m_curve": []}
    if args.gap_profile or args.kernel_gaps or args.only_breakdown:  # these combine in one load
        for batch in (int(b) for b in args.batch.split(",")):
            if args.gap_profile:
                gap_profile(ex, tok, out, args.tokens, batch, args.accept)
                if args.ple_cold:
                    gap_profile(ex, tok, out, args.tokens, batch, args.accept, cold=True)
            if args.kernel_gaps:
                kernel_gaps(ex, tok, out, min(args.tokens, 96), batch, args.accept)
        if args.only_breakdown:
            verify_breakdown(ex, tok, out, args.tokens)
    elif args.host_profile:
        host_profile(ex, tok, out, args.tokens)
    else:
        if not args.skip_rounds:
            rounds(ex, tok, out, args.tokens, profiled=not args.rounds_only)
        if not args.rounds_only:
            m_curve(ex, model, tok, out, args.context)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
