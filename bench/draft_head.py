# SPDX-License-Identifier: Apache-2.0
"""Draft-only lm_head variants for the Flash-Next MTP proposer (#35): pick agreement and cost.

Generates eagerly with the served draft head (the target's FP8 lm_head) and records every hidden
row a draft step reads. Each candidate head then picks over the same rows, so agreement with the
served pick is measured without the continuation drift a served A/B has. Cost is one head call
per row count, replayed in a CUDA graph.

    SHARAR_TARGET=flashnext python3 -m bench.draft_head --out draft_head.json
"""
import argparse
import json
from pathlib import Path

import torch

from bench.flashnext_round import PROMPTS, chat_ids
from sharar.engine.run import _load
from sharar.engine.sampler import SamplingParams
from sharar.proposers.draft_vocab import ReducedHead, draft_vocab

EXTRA = {
    "json": "Return a JSON array of 12 objects describing fictional employees, each with id, name, email, department, start_date, salary and a nested address object. Output only the JSON.",
    "multilingual": "Écris un court récit en français sur un phare en Bretagne, puis traduis-le en espagnol et en allemand.",
}


def head_ms(head, hidden, rows: int, iters: int = 50) -> float:
    x = hidden[:rows].contiguous()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            head(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(10):
            head(x)
    graph.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / (iters * 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--vocab", type=int, nargs="*", default=[65536])
    ap.add_argument("--rescore", type=int, nargs="*", default=[2, 4, 8, 16])
    args = ap.parse_args()
    model, ex, tok = _load(graphs=False, spec=3)
    head = ex.spec.proposer.head
    if getattr(head, "draft_head", None) is not None:
        raise SystemExit("run with the served head (SHARAR_DRAFT_HEAD and SHARAR_DRAFT_VOCAB unset)")
    served = head.draft_argmax
    rows = {}
    current = []

    def record(d):
        current.append(head.mixer.mix(d)[0].detach().clone())
        return served(d)
    head.draft_argmax = record
    stats = {}
    for name, prompt in {**PROMPTS, **EXTRA}.items():
        current.clear()
        ex.spec.reset_stats()
        ex.generate(chat_ids(tok, prompt), SamplingParams(max_new_tokens=args.tokens, ignore_eos=True))
        ex.flush_cache()
        rows[name] = torch.cat(current)
        s = ex.spec.stats()
        stats[name] = {"draft_rows": rows[name].shape[0], "tokens_per_round": round(s.get("tokens_per_round", 0), 3)}
        print(name, json.dumps(stats[name]), flush=True)
    head.draft_argmax = served

    reference = lambda x: head.lm_head(x).argmax(-1)  # noqa: E731  the served pick
    snap = model.target.snapshot_dir()
    n = head.lm_head.out_features
    candidates = {"fp8": ReducedHead(head.lm_head), "w3": ReducedHead(head.lm_head, fmt="w3")}
    for r in args.rescore:
        candidates[f"w3-r{r}"] = ReducedHead(head.lm_head, fmt="w3", rescore=r)
    for v in args.vocab:
        ids = draft_vocab(snap, v, n)
        candidates[f"fp8-{v}"] = ReducedHead(head.lm_head, ids)
        candidates[f"w3-{v}-r8"] = ReducedHead(head.lm_head, ids, "w3", 8)
    out = {"prompts": stats, "heads": {}}
    every = torch.cat(list(rows.values()))
    for label, cand in candidates.items():
        agree = {}
        for name, x in rows.items():
            agree[name] = round(float((cand.argmax(x) == reference(x)).float().mean()), 4)
        agree["all"] = round(float((cand.argmax(every) == reference(every)).float().mean()), 4)
        cost = {m: round(head_ms(cand.argmax, every, m), 3) for m in (1, 4)}
        out["heads"][label] = {"extra_gib": round(cand.bytes / 2**30, 3), "ms_per_call": cost, "agreement": agree}
        print(label, json.dumps(out["heads"][label]), flush=True)
        del cand
        candidates[label] = None
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
