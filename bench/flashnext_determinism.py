# SPDX-License-Identifier: Apache-2.0
"""Where do repeated Flash-Next decodes diverge? Graph rerun, eager vs graph, cache hit vs cold prefill."""

import json
import os
import sys
from pathlib import Path

import torch

from sharar.engine.run import _load
from sharar.engine.sampler import SamplingParams


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    prompts = json.loads(Path(sys.argv[1]).read_text())
    ids = prompts[sys.argv[2] if len(sys.argv) > 2 else "code512"]
    n = int(os.environ.get("TOKENS", "64"))
    model, ex, tok = _load(graphs=True)
    params = SamplingParams(max_new_tokens=n, ignore_eos=True)

    def gen(graphs: bool):
        ex.use_graphs = graphs
        return ex.generate(ids, params).output_ids

    runs = {}
    runs["cold1"] = gen(True)          # cold prefill (graphs)
    runs["hit1"] = gen(True)           # prefix-cache hit (graphs)
    runs["hit_eager"] = gen(False)     # prefix-cache hit (eager)
    ex.flush_cache()
    runs["cold2"] = gen(True)          # cold prefill again
    ex.flush_cache()
    runs["cold_eager"] = gen(False)    # cold prefill, eager decode
    runs["hit2"] = gen(True)
    base = runs["cold1"]
    for name, out in runs.items():
        print(f"  {name:12s} first divergence from cold1: {first_diff(base, out)}   from hit1: {first_diff(runs['hit1'], out)}", flush=True)
    print("  text:", repr(tok.decode(base)[:120]))
    print("done")


if __name__ == "__main__":
    main()
