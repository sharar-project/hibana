"""Paged FP8 KV cache for the 16 full-attention layers.

Layout is FlashInfer "NHD" paged: k[layer] and v[layer] are
[num_pages, page_size, num_kv_heads, head_dim] in float8_e4m3fn. K/V are
written with scale 1.0, as the reference does for this checkpoint (it ships no
k_scale/v_scale). Page size defaults to 1 like the reference; the roadmap's
Phase 2 measurement of 1 versus 16 decides whether that changes.
"""

from __future__ import annotations

import torch

from sharar import config

KV_DTYPE = torch.float8_e4m3fn


class KvPool:
    def __init__(self, num_pages: int, device: torch.device, page_size: int = 1):
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = device
        self.layer_ids = [i for i in range(config.NUM_LAYERS) if (i + 1) % config.FULL_ATTENTION_INTERVAL == 0]
        shape = (num_pages, page_size, config.NUM_KV_HEADS, config.HEAD_DIM)
        self.k = {i: torch.zeros(shape, dtype=KV_DTYPE, device=device) for i in self.layer_ids}
        self.v = {i: torch.zeros(shape, dtype=KV_DTYPE, device=device) for i in self.layer_ids}
        # Simple free list; page 0 is reserved as the dummy page for padded batch rows.
        self._free = list(range(num_pages - 1, 0, -1))

    @property
    def bytes(self) -> int:
        n = self.num_pages * self.page_size * config.NUM_KV_HEADS * config.HEAD_DIM
        return n * 2 * len(self.layer_ids)

    def alloc(self, n: int) -> torch.Tensor:
        if n > len(self._free):
            raise MemoryError(f"KV pool exhausted: need {n} pages, {len(self._free)} free")
        pages = [self._free.pop() for _ in range(n)]
        return torch.tensor(pages, dtype=torch.int32, device=self.device)

    def free(self, pages: torch.Tensor) -> None:
        self._free.extend(int(p) for p in pages.tolist())

    def write(self, layer_id: int, pages: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write [T, H, D] bf16 K/V for tokens whose (page, offset) are given
        with page_size 1: pages[t] is the token's page."""
        assert self.page_size == 1
        self.k[layer_id][pages, 0] = k.to(KV_DTYPE)
        self.v[layer_id][pages, 0] = v.to(KV_DTYPE)

    def cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer_id], self.v[layer_id]
