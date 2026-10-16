"""Counter-based random draws for speculative sampling (Philox in the kernels).

A sampled request owns a 63-bit seed (its `seed`, else random at admission). Round r of its speculation draws
the draft's uniforms at offsets ROUND_STRIDE * r + DRAFT + j and the verification's at ROUND_STRIDE * r + VERIFY + j,
r being `Request.spec_rounds`, so no draw is reused and a request's draws depend only on its seed and its rounds,
not on the batch it runs in. The kernels take the per-row values from one small upload per call site and round
(`rows`), copied from pinned memory without blocking: a pageable upload would wait for the graphs already queued.
"""
from __future__ import annotations

import secrets

import torch

ROUND_STRIDE, DRAFT, VERIFY = 64, 0, 32
MAX_DEPTH = VERIFY - 1  # drafted tokens per round the offsets leave room for (the verification draws one more)


def request_seed(seed: int | None) -> int:
    """The request's Philox seed: its own seed folded into 63 bits, else a random one."""
    return secrets.randbits(63) if seed is None else seed % (1 << 63)


def offsets(reqs, part: int) -> list[int]:
    return [ROUND_STRIDE * r.spec_rounds + part for r in reqs]


_PINNED: dict = {}  # (site, dtype, device) -> (pinned host buffer, device buffer)


def rows(site: str, values: list[list], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """values [B][n] as a device tensor, through a pinned buffer owned by `site` (one per call site: the copy of one
    site's previous round has completed, because every round ends with a synchronizing download)."""
    b, n = len(values), len(values[0])
    key = (site, dtype, str(device))
    host, dev = _PINNED.get(key, (None, None))
    if host is None or host.numel() < b * n:
        size = max(b * n, 2 * (host.numel() if host is not None else 0), 256)
        host = torch.empty(size, dtype=dtype, pin_memory=True)
        dev = torch.empty(size, dtype=dtype, device=device)
        _PINNED[key] = host, dev
    host[:b * n].copy_(torch.tensor(values, dtype=dtype).view(-1))
    out = dev[:b * n]
    out.copy_(host[:b * n], non_blocking=True)
    return out.view(b, n)
