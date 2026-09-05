# SPDX-License-Identifier: Apache-2.0
"""Device-resident greedy traversal of DFlash2's conditional candidate lattice.

The publisher's top-k and bilinear scores are unchanged. One warp per request
walks candidate ranks without transferring each predecessor rank to the host.
The sampled path continues to supply its exact conditional law separately.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _greedy_lattice(candidates, scores, output,
                    CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr,
                    SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr,
                    STEPS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK)
    previous = tl.full((), 0, tl.int32)  # every first-step row represents the anchor
    for step in range(STEPS):
        values = tl.load(scores + row * SS0 + step * SS1 + previous * SS2 + ranks * SS3,
                         ranks < WIDTH, float('-inf')).to(tl.float32)
        # torch.argmax chooses the first maximum, and the first NaN if present.
        first_nan = tl.min(tl.where((ranks < WIDTH) & (values != values), ranks, BLOCK), 0)
        maximum = tl.max(tl.where(values == values, values, float('-inf')), 0)
        best = tl.min(tl.where((ranks < WIDTH) & (values == maximum), ranks, BLOCK), 0)
        previous = tl.where(first_nan < BLOCK, first_nan, best)
        token = tl.load(candidates + row * CS0 + step * CS1 + previous * CS2)
        tl.store(output + row * STEPS + step, token)


def greedy_lattice(candidates, scores):
    """Return owned int64 [B, k] IDs; inputs may be borrowed graph views."""
    batch, steps, width = candidates.shape
    if (scores.shape != (batch, steps, width, width) or not 1 <= width <= 32
            or candidates.dtype != torch.int64 or not scores.is_floating_point()
            or not candidates.is_cuda or scores.device != candidates.device):
        raise ValueError('invalid candidate lattice')
    output = torch.empty((batch, steps), device=candidates.device, dtype=torch.int64)
    _greedy_lattice[(batch,)](candidates, scores, output, *candidates.stride(), *scores.stride(),
                              steps, width, triton.next_power_of_2(width), num_warps=1)
    return output
