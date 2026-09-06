"""Bounded numerical oracle for right-aligned causal grouped-query attention."""
import math

import torch


def causal_attention(q, k, v, dtype=torch.float64):
    """Q [T,Hq,D], K/V [L,Hkv,D]; query zero follows L-T prefix tokens.

    Inputs are already dequantized stored values; no RoPE, KV re-quantization
    or learned scaling is introduced here. This is not a serving kernel.
    """
    if (q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or k.shape != v.shape
            or q.shape[-1] != k.shape[-1] or not k.shape[1]
            or q.shape[1] % k.shape[1] or k.shape[0] < q.shape[0]):
        raise ValueError('invalid causal grouped-query geometry')
    t, hq, d = q.shape
    length, hkv, _ = k.shape
    query = q.to(dtype).reshape(t, hkv, hq // hkv, d).permute(1, 2, 0, 3)
    keys = k.to(dtype).permute(1, 2, 0)[:, None]
    values = v.to(dtype).permute(1, 0, 2)[:, None]
    scores = (query @ keys) / math.sqrt(d)
    blocked = torch.arange(length, device=q.device)[None, :] > (length - t + torch.arange(t, device=q.device))[:, None]
    scores.masked_fill_(blocked, -torch.inf)
    return (scores.softmax(-1) @ values).permute(2, 0, 1, 3).reshape(t, hq, d)
