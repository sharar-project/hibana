#!/usr/bin/env python3
"""Run the reference SGLang engine offline and dump per-layer hidden states.

Run inside `tools/dev-shell.sh --with-reference` so sglang and the patched
tensor dumper (sitecustomize.py) are on PYTHONPATH:

  SHARAR_REFERENCE_DUMP=1 python3 tools/reference/dump_reference.py \
      --prompts /results/parity/prompts.json --out /results/parity/reference

Writes <out>/dump/<name>.pt with keys: input_ids, positions, embed,
layers (list of [hidden, residual]), norm, logits_last.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sharar.parity.prompts import DUMP_MAX_TOKENS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=DUMP_MAX_TOKENS)
    args = ap.parse_args()

    prompts = {k: v for k, v in json.loads(Path(args.prompts).read_text()).items() if len(v) <= args.max_tokens}
    out_dir = Path(args.out) / "dump"
    raw_dir = Path(args.out) / "raw_passes"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    import sglang

    engine = sglang.Engine(
        model_path="RadixArk/Qwen3.8-27B-NVFP4",
        revision="319f741cce68d7914884900c138a1fbb70a42f30",
        trust_remote_code=True,
        language_model_only=True,
        context_length=32768,
        max_total_tokens=8192,
        kv_cache_dtype="auto",
        mem_fraction_static=0.45,
        max_running_requests=1,
        chunked_prefill_size=-1,
        max_prefill_tokens=8192,
        attention_backend="flashinfer",
        mamba_ssm_dtype="float32",
        mamba_radix_cache_strategy="extra_buffer",
        max_mamba_cache_size=5,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        debug_tensor_dump_output_folder=str(raw_dir),
        log_level="warning",
    )
    order = []
    try:
        for name, ids in prompts.items():
            out = engine.generate(
                input_ids=ids, sampling_params={"max_new_tokens": 1, "temperature": 0}
            )
            order.append((name, ids, out["output_ids"]))
            print(f"{name}: {len(ids)} tokens -> next {out['output_ids']}", flush=True)
        # One extra pass flushes the last real pass out of the dumper.
        engine.generate(input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 1, "temperature": 0})
    finally:
        engine.shutdown()

    pass_files = sorted(raw_dir.rglob("Pass*.pt"))
    print(f"{len(pass_files)} pass files", flush=True)
    by_ids: dict[tuple, Path] = {}
    for pf in pass_files:
        d = torch.load(pf, map_location="cpu")
        if "input_ids" not in d:
            continue
        key = tuple(d["input_ids"].tolist())
        by_ids[key] = pf
    for name, ids, next_ids in order:
        pf = by_ids.get(tuple(ids))
        if pf is None:
            print(f"MISSING prefill pass for {name}", file=sys.stderr)
            continue
        d = torch.load(pf, map_location="cpu")
        layer_keys = sorted(
            (k for k in d if k.startswith("layers.")), key=lambda k: int(k.split(".")[-1])
        )
        embed = d["embed"]
        norm = d["norm"]
        record = {
            "input_ids": d["input_ids"],
            "embed": embed,
            "layers": [d[k] for k in layer_keys],
            "norm": norm,
            "logits_last": d["logits"],
            "next_ids": next_ids,
            "keys": list(d.keys()),
        }
        torch.save(record, out_dir / f"{name}.pt")
        print(f"saved {name}: {len(layer_keys)} layers, embed {tuple(embed.shape)}, logits {tuple(d['logits'].shape)}", flush=True)
    shutil.rmtree(raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
