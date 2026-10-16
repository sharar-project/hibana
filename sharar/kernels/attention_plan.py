"""One attention planner per KV geometry.

Every paged attention call in the engine plans a FlashInfer wrapper first: the
target's prefill, decode and verification (eager and graph-bound), both MTP
heads, their draft graphs and the parallel draft. The planner holds the
geometry once (heads, head size, page size, dtypes, scale, sliding window) and
keeps the two host-side rules in one place:

- index pointers are host tensors (`host_indptr`): `plan()` reads them on the
  CPU, so a device tensor would cost a synchronization per plan;
- eager wrappers re-plan through `plan_guard` (a plan must not overwrite the
  pinned buffer of a copy the GPU has not executed yet); graph-bound wrappers
  plan into their own fixed buffers between replays and skip the event wait
  (`guard=False`), as they always did.

The arguments each call passes to FlashInfer are the ones its call site passed
before, so planning and arithmetic are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sharar.kernels import plan_guard


def host_indptr(lengths) -> torch.Tensor:
    """[0, cumsum(lengths)] as an int32 host tensor."""
    return torch.tensor([0] + list(torch.tensor(list(lengths)).cumsum(0).tolist()), dtype=torch.int32)


@dataclass(frozen=True)
class AttentionGeometry:
    q_heads: int
    kv_heads: int
    head_dim: int
    page_size: int = 1
    kv_dtype: torch.dtype = torch.bfloat16
    q_dtype: torch.dtype = torch.bfloat16
    window_left: int | None = None  # passed to FlashInfer only when set (the parallel draft's sliding window)

    @property
    def sm_scale(self) -> float:
        return self.head_dim ** -0.5

    @classmethod
    def of_target(cls, target, page_size: int, kv_dtype: torch.dtype) -> AttentionGeometry:
        return cls(target.num_attention_heads, target.num_kv_heads, target.head_dim, page_size, kv_dtype)


class AttentionPlanner:
    def __init__(self, geometry: AttentionGeometry):
        self.geometry = geometry

    def _call(self, wrapper, args, kwargs, guard: bool) -> None:
        if guard:
            plan_guard.plan(wrapper, *args, **kwargs)
        else:
            wrapper.plan(*args, **kwargs)

    def paged_prefill(self, wrapper, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor,
                      last_page_len: torch.Tensor, *, causal: bool, guard: bool = True, page_size: int | None = None) -> None:
        """Plan a BatchPrefillWithPagedKVCacheWrapper: rows `qo_indptr` against the pages `kv_indices`."""
        g = self.geometry
        kwargs = dict(causal=causal, q_data_type=g.q_dtype, kv_data_type=g.kv_dtype, sm_scale=g.sm_scale)
        if g.window_left is not None:
            kwargs["window_left"] = g.window_left
        self._call(wrapper, (qo_indptr, kv_indptr, kv_indices, last_page_len, g.q_heads, g.kv_heads, g.head_dim,
                             g.page_size if page_size is None else page_size), kwargs, guard)

    def decode(self, wrapper, indptr: torch.Tensor, indices: torch.Tensor, last_page_len: torch.Tensor, *, guard: bool = True) -> None:
        """Plan a BatchDecodeWithPagedKVCacheWrapper: one row per sequence against its pages."""
        g = self.geometry
        self._call(wrapper, (indptr, indices, last_page_len, g.q_heads, g.kv_heads, g.head_dim, g.page_size),
                   dict(pos_encoding_mode="NONE", q_data_type=g.q_dtype, kv_data_type=g.kv_dtype, sm_scale=g.sm_scale), guard)

    def ragged_prefill(self, wrapper, indptr: torch.Tensor) -> None:
        """Plan a BatchPrefillWithRaggedKVCacheWrapper: causal attention over the new rows' own K/V."""
        g = self.geometry
        plan_guard.plan(wrapper, indptr, indptr, g.q_heads, g.kv_heads, g.head_dim,
                        causal=True, sm_scale=g.sm_scale, q_data_type=g.q_dtype, kv_data_type=g.kv_dtype)
