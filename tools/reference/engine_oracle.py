#!/usr/bin/env python3
"""Reference self-consistency: offline engine (no CUDA graph, no chunking)
versus the serving container (CUDA graphs, chunked prefill), both SGLang.

Writes <out>/engine/<name>.json in the same format as http_oracle.py so the
two can be compared position by position. Run under --with-reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--subdir", default="engine")
    ap.add_argument("--chunked-prefill-size", type=int, default=-1)
    args = ap.parse_args()
    prompts = json.loads(Path(args.prompts).read_text())
    out_dir = Path(args.out) / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    import sglang

    engine = sglang.Engine(
        model_path="RadixArk/Qwen3.8-27B-NVFP4",
        revision="319f741cce68d7914884900c138a1fbb70a42f30",
        trust_remote_code=True,
        language_model_only=True,
        context_length=32768,
        max_total_tokens=16384,
        kv_cache_dtype="auto",
        mem_fraction_static=0.45,
        max_running_requests=1,
        chunked_prefill_size=args.chunked_prefill_size,
        max_prefill_tokens=16384,
        attention_backend="flashinfer",
        mamba_ssm_dtype="float32",
        mamba_radix_cache_strategy="extra_buffer",
        max_mamba_cache_size=5,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        log_level="warning",
    )
    try:
        for name, ids in prompts.items():
            out = engine.generate(
                input_ids=ids,
                sampling_params={"max_new_tokens": 8, "temperature": 0},
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=args.top_k,
            )
            meta = out["meta_info"]
            top = [None if e is None else [[float(lp), int(t)] for lp, t, _ in e] for e in meta["input_top_logprobs"]]
            rec = {"name": name, "prompt_len": len(ids), "input_top_logprobs": top, "output_ids": [int(t) for t in out["output_ids"]]}
            (out_dir / f"{name}.json").write_text(json.dumps(rec))
            print(f"{name}: next {rec['output_ids'][:4]}", flush=True)
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
