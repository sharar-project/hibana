"""Include a learned virtual sink in paged draft attention's denominator.

FlashInfer FA2 returns base-2 log-sum-exp but ignores its optional sink input.
Rescale its zero-sink numerator without introducing a physical cache token.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sink(X, Lse, Sinks, Y, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // D
    lse = tl.load(Lse + row, i < N, 0)
    sink = tl.load(Sinks + row % H, i < N, 0)
    factor = 1.0 / (1.0 + tl.exp2(sink * 1.4426950408889634 - lse))
    value = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, value * factor, i < N)


def with_sink(output, lse, sinks):
    out = torch.empty_like(output)
    _sink[(triton.cdiv(out.numel(), 1024),)](output, lse, sinks, out, out.numel(), out.shape[1], out.shape[2], 1024)
    return out
