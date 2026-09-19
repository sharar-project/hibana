# SPDX-License-Identifier: Apache-2.0
"""Engine MoeLayer with real weights: per-layer time when cycling through N layers (memory-system effects)."""

import os
import sys
from pathlib import Path

import torch

from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.flashnext.moe import MoeLayer
from sharar.target import FLASHNEXT
from sharar.weights.safetensors import SnapshotWeights

snap = FLASHNEXT.snapshot_dir()
dev = torch.device("cuda")
N = int(os.environ.get("LAYERS", "12"))
w = SnapshotWeights(snap)
layers = []
for i in range(N):
    layers.append(MoeLayer(w, f"model.language_model.layers.{i}.mlp", dev, FLASHNEXT, TrunkLinear, "fp8"))
    w.close()
print(f"{N} layers loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
torch.manual_seed(0)
for M in (1, 4):
    xs = [torch.randn((M, 2560), device=dev, dtype=torch.bfloat16) for _ in range(8)]
    for n_cycle in sorted({1, min(4, N), N}):
        def step(i):
            layers[i % n_cycle](xs[i % 8])
        for i in range(2 * n_cycle):
            step(i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(n_cycle):
                step(i)
        g.replay(); torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        iters = max(4, 48 // n_cycle)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record(); torch.cuda.synchronize()
        per_layer = a.elapsed_time(b) / iters / n_cycle
        print(f"M={M} cycling {n_cycle:2d} layers: {per_layer * 1000:6.0f} us per layer (routing + fused MoE + shared expert) -> x48 = {per_layer * 48:5.1f} ms", flush=True)
print("done")
