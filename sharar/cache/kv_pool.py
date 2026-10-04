"""Paged FP8 KV cache for the 16 full-attention layers.

Layout is FlashInfer "NHD" paged: k[layer] and v[layer] are
[num_pages, page_size, num_kv_heads, head_dim] in float8_e4m3fn. K/V are
written with scale 1.0, as the reference does for this checkpoint (it ships no
k_scale/v_scale). Page size is 1 like the reference. Measured 2026-09-18 with
`bench/kv_page_size.py`, tag archive/bench-pre-split-2026-10-04: page 16 would leave decode attention unchanged and
save about 0.45 ms (batch one) to 1.1 ms (batch four) of 16-row verify
attention per round at 10K context, under 1% of the round, so the
token-granular cache stays.
"""

from __future__ import annotations

import torch

from sharar.target import DEFAULT_TARGET, Target

KV_DTYPE = torch.float8_e4m3fn


class KvPool:
    def __init__(self, num_pages: int, device: torch.device, page_size: int = 1, layer_ids: list[int] | None = None, dtype: torch.dtype = KV_DTYPE,
                 target: Target | None = None, padding_rows: int = 0):
        self.target = target or DEFAULT_TARGET
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.num_kv_heads, self.head_dim = self.target.num_kv_heads, self.target.head_dim
        self.layer_ids = layer_ids if layer_ids is not None else list(self.target.attention_layer_ids)
        # Write-only pages past the allocatable ones, one per padded row of a CUDA graph (`padding_pages`).
        self.padding_rows = padding_rows
        self.padding_pages = torch.arange(num_pages, num_pages + padding_rows, dtype=torch.int32, device=device)
        shape = (num_pages + padding_rows, page_size, self.num_kv_heads, self.head_dim)
        self.k = {i: torch.zeros(shape, dtype=dtype, device=device) for i in self.layer_ids}
        self.v = {i: torch.zeros(shape, dtype=dtype, device=device) for i in self.layer_ids}
        # Per-page rows that belong to a token like its K/V and travel with the page
        # through the prefix cache, the cold tier and the session store (Flash-Next's
        # sparse-attention index keys). Registered by the model before pools are shared.
        self.page_aux: dict[str, torch.Tensor] = {}
        # Simple free list; page 0 is reserved as the dummy page for padded batch rows.
        self._free = list(range(num_pages - 1, 0, -1))

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def bytes(self) -> int:
        n = (self.num_pages + self.padding_rows) * self.page_size * self.num_kv_heads * self.head_dim
        kv = n * 2 * len(self.layer_ids) * self.k[self.layer_ids[0]].element_size()
        return kv + sum(t.numel() * t.element_size() for t in self.page_aux.values())

    def register_page_aux(self, name: str, row_shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A [num_pages, *row_shape] tensor indexed by page id like K/V."""
        if name in self.page_aux:
            raise ValueError(f"page aux {name!r} registered twice")
        self.page_aux[name] = torch.zeros(((self.num_pages + self.padding_rows) * self.page_size, *row_shape), dtype=dtype, device=self.device)
        return self.page_aux[name]

    def pad_pages(self, n: int) -> torch.Tensor:
        """Distinct pages for the K/V of n padded graph rows. Padded rows still attend to page 0, but they must
        not write it together: a duplicate-index write keeps an arbitrary row's values, so page 0 and the
        padded rows would differ from replay to replay, and Flash-Next's per-tensor activation scale carries
        that into the real rows (#94)."""
        if n > self.padding_rows:
            raise ValueError(f"{n} padded rows need more than the pool's {self.padding_rows} padding pages")
        return self.padding_pages[:n]

    def alloc(self, n: int) -> torch.Tensor:
        return self.alloc_with_ids(n)[0]

    def alloc_with_ids(self, n: int) -> tuple[torch.Tensor, list[int]]:
        """Allocate n pages; returns the device tensor and the same ids as a
        host list so callers can free without a device-to-host copy. The upload
        is a pinned non-blocking copy so it does not stall the host on GPU work."""
        if n < 0:
            raise ValueError("allocation size cannot be negative")
        if n > len(self._free):
            raise MemoryError(f"KV pool exhausted: need {n} pages, {len(self._free)} free")
        pages = [self._free.pop() for _ in range(n)]
        host = torch.tensor(pages, dtype=torch.int32).pin_memory()
        return host.to(self.device, non_blocking=True), pages

    def free(self, pages: torch.Tensor) -> None:
        self._free.extend(int(p) for p in pages.tolist())

    def free_ids(self, pages: list[int]) -> None:
        self._free.extend(pages)

    def write(self, layer_id: int, pages: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write [T, H, D] bf16 K/V for tokens whose (page, offset) are given
        with page_size 1: pages[t] is the token's page."""
        assert self.page_size == 1
        self.k[layer_id][pages, 0] = k.to(self.dtype)
        self.v[layer_id][pages, 0] = v.to(self.dtype)

    def cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer_id], self.v[layer_id]
