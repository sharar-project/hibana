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


@triton.jit
def _sampled_lattice(candidates, scores, tokens_out, laws_out, row_params, rng,
                     CS0, CS1, CS2, SS0, SS1, SS2, SS3, STEPS, WIDTH, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK)
    valid = ranks < WIDTH
    temperature = tl.load(row_params + row * 4)
    top_k = tl.load(row_params + row * 4 + 1).to(tl.int32)
    top_p = tl.load(row_params + row * 4 + 2)
    greedy = tl.load(row_params + row * 4 + 3) > 0
    seed = tl.load(rng + row * 2)
    offset = tl.load(rng + row * 2 + 1).to(tl.int32)
    truncate = (top_k > 0) & (top_k < WIDTH)
    previous = tl.full((), 0, tl.int32)
    for step in range(STEPS):
        raw = tl.load(scores + row * SS0 + step * SS1 + previous * SS2 + ranks * SS3, valid, float('-inf')).to(tl.float32)
        x = raw / temperature
        # top-k with ties: the k-th largest value bounds what is kept (distribution's mask)
        ordered = tl.sort(x, 0, descending=True)
        kth = tl.sum(tl.where(ranks == top_k - 1, ordered, 0.0), 0)
        keep = valid & ((not truncate) | (x >= kth))
        m = tl.max(tl.where(keep, x, float('-inf')), 0)
        e = tl.where(keep, tl.exp(x - m), 0.0)
        p = e / tl.sum(e, 0)
        # nucleus: keep a token when the mass strictly above it is under top_p (exact ties at the boundary all kept)
        above = tl.sum(tl.where(p[None, :] > p[:, None], p[None, :], 0.0), 1)
        law = tl.where((top_p >= 1) | (above < top_p), p, 0.0)
        law = law / tl.sum(law, 0)
        # greedy rows: the first maximum of the raw scores, as greedy_lattice
        best = tl.min(tl.where(valid & (raw == tl.max(tl.where(valid, raw, float('-inf')), 0)), ranks, BLOCK), 0)
        law = tl.where(greedy, tl.where(ranks == best, 1.0, 0.0), law)
        # inverse CDF with this step's Philox uniform
        u = tl.rand(seed, offset + step + ranks * 0)
        cdf = tl.cumsum(law, 0)
        pick = tl.minimum(tl.sum(tl.where(valid & (cdf <= u * tl.sum(law, 0)), 1, 0), 0), WIDTH - 1)
        previous = tl.where(greedy, best, pick).to(tl.int32)
        tl.store(laws_out + (row * STEPS + step) * WIDTH + ranks, law, valid)
        tl.store(tokens_out + row * STEPS + step, tl.load(candidates + row * CS0 + step * CS1 + previous * CS2))


def sampled_lattice(candidates, scores, params, rng):
    """Sampled drafts from the candidate lattice, one program per row: `distribution`'s law over the W scores that
    follow the previous choice at each step (temperature, top-k with ties, the nucleus), and a draw by inverse CDF
    with a Philox uniform at (seed, offset + step). params float32 [B, 4] (temperature, top_k, top_p, greedy), rng
    int64 [B, 2] (seed, offset). Returns (tokens int64 [B, k], laws float32 [B, k, W]); greedy rows walk the argmax."""
    batch, steps, width = candidates.shape
    if (scores.shape != (batch, steps, width, width) or not 1 <= width <= 32 or candidates.dtype != torch.int64
            or not candidates.is_cuda or params.shape != (batch, 4) or rng.shape != (batch, 2)):
        raise ValueError('invalid candidate lattice')
    tokens = torch.empty((batch, steps), device=candidates.device, dtype=torch.int64)
    laws = torch.empty((batch, steps, width), device=candidates.device, dtype=torch.float32)
    _sampled_lattice[(batch,)](candidates, scores, tokens, laws, params, rng, *candidates.stride(), *scores.stride(),
                               steps, width, triton.next_power_of_2(width), num_warps=1)
    return tokens, laws
