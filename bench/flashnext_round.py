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
"""
import argparse
import json
import re
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
CATEGORIES = [  # first match wins
    ("moe", r"moe|expert|cutlass.*grouped|fused_moe|route|swizzle"),
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


def rounds(ex, tok, out, tokens: int):
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
    base = chat_ids(tok, PROMPTS["prose"] * 40)[:context]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--context", type=int, default=2000)
    args = ap.parse_args()
    model, ex, tok = _load(graphs=True, spec=3)
    out = {"rounds": [], "m_curve": []}
    rounds(ex, tok, out, args.tokens)
    m_curve(ex, model, tok, out, args.context)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
