# SPDX-License-Identifier: Apache-2.0
"""Plain-torch references for Qwen3.8-Flash-Next, following SGLang's `qwen4_exp.py` and `qsa` package (Apache-2.0):
the Gemma RMS norm and NeoX RoPE the indexer uses, and the Qwen Sparse Attention token mask the QSA tests check the
kernels against.
"""

from __future__ import annotations

import torch


def gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xn * (1.0 + weight.float())).to(x.dtype)


def rope_neox(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    """x [T, H, D]: rotate the first rotary_dim dims, NeoX halves, cos/sin in float32."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim))
    ang = positions.float()[:, None] * inv[None]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    r, rest = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    a, b = r[..., : rotary_dim // 2], r[..., rotary_dim // 2 :]
    rotated = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).to(x.dtype)
    return torch.cat([rotated, rest], dim=-1)


def qsa_token_mask(q: torch.Tensor, keys: torch.Tensor, budget: int, ratio: int) -> torch.Tensor:
    """[T, T] bool, True where query row i attends to token j, for one sequence
    starting at position 0 (Qwen Sparse Attention, as SGLang's qsa package).

    q [T, H, D] are the normed, rotated index queries; keys [G, D] the
    compressed keys of the G = T // ratio complete groups (mean of the group's
    raw index keys, Gemma norm, rotated at the group's first position). Row i
    sees the (i + 1) // ratio groups that end at or before it, scores them
    relu(q . k) summed over heads / sqrt(D), keeps the top budget // ratio,
    and adds the tail of its own incomplete group (at most ratio - 1 tokens,
    itself included). With at most budget // ratio groups everything is kept:
    dense causal attention."""
    t, d = q.shape[0], q.shape[-1]
    g = keys.shape[0]
    rows = torch.arange(t, device=q.device)
    visible = (rows + 1) // ratio
    mask = torch.zeros((t, t), dtype=torch.bool, device=q.device)
    tail_start = visible * ratio
    cols = torch.arange(t, device=q.device)
    mask |= (cols[None, :] >= tail_start[:, None]) & (cols[None, :] <= rows[:, None])
    if g == 0:
        return mask
    top = budget // ratio
    for a in range(0, t, 1024):  # [rows, groups] logits in row chunks
        b = min(t, a + 1024)
        logits = torch.relu(torch.einsum("thd,gd->tgh", q[a:b].float(), keys.float())).sum(-1) / d ** 0.5
        seen = torch.arange(g, device=q.device)[None, :] < visible[a:b, None]
        logits = logits.masked_fill(~seen, -float("inf"))
        width = min(top, g)
        values, chosen = logits.topk(width, dim=-1)
        chosen = torch.where(torch.isfinite(values), chosen, torch.full_like(chosen, -1))
        groups = torch.zeros((b - a, g + 1), dtype=torch.bool, device=q.device)
        groups.scatter_(1, torch.where(chosen >= 0, chosen, torch.full_like(chosen, g)), True)
        member = groups[:, :g].repeat_interleave(ratio, dim=1)  # [rows, g * ratio]
        mask[a:b, : g * ratio] |= member
    return mask
