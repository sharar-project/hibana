"""Shared CUDA-graph bookkeeping: the batch bucket a batch pads to, and warm-up-then-capture.

Decode, verification, device acceptance and both draft bodies keep their own
static buffers and bodies; this module holds the parts they all repeated.
"""

from __future__ import annotations

from typing import Callable

import torch


def pick(sizes: tuple[int, ...], b: int) -> int:
    """The smallest captured batch size that holds b rows."""
    for s in sizes:
        if s >= b:
            return s
    raise ValueError(f"batch {b} exceeds captured sizes {sizes}")


def capture(pool, body: Callable[[], None], warmup: Callable[[], None] | None = None, rounds: int = 2) -> torch.cuda.CUDAGraph:
    """Run `warmup` (default: `body`) `rounds` times so Triton/FlashInfer JIT and the allocator settle, then
    capture `body` into a graph that allocates from `pool`."""
    warm = body if warmup is None else warmup
    for _ in range(rounds):
        warm()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        body()
    torch.cuda.synchronize()
    return graph
