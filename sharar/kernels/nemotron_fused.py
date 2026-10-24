# SPDX-License-Identifier: Apache-2.0
"""Fused decode/verify kernels for Nemotron-3.5-Super's trunk, each bitwise against the chain it replaces
(`tests/gpu/test_nemotron_fused.py`).

- `write_kv`: the attention's K and V rows straight from the q|k|v projection output into the KV pool's pages, cast
  to the pool's dtype on the way. It replaces `.contiguous()` copies of k and v, the pool's two casts and two
  index_put writes (seven launches a layer) with one. The cast is FP32 -> e4m3 round-to-nearest-even as torch's;
  torch maps |x| >= 464 to NaN where this saturates at 448, a value no attention input reaches (it would be NaN
  in every later attention read).

SHARAR_NEMOTRON_FUSED=0 turns every Nemotron fusion off (this module's, the residual norm's), for A/B runs.
"""

from __future__ import annotations


import torch
import triton
import triton.language as tl


ENABLED: bool | None = None  # tests pin it; None: the engine's setting (EngineConfig.nemotron_fused)


def enabled() -> bool:
    """Whether Nemotron's fused kernels run (SHARAR_NEMOTRON_FUSED, read through the engine's config, not at import)."""
    if ENABLED is not None:
        return ENABLED
    from sharar.engine.config import current
    return current().nemotron_fused


@triton.jit
def _write_kv_kernel(QKV, s_row, Pages, K, V, s_page_k, s_page_v, K_OFF: tl.constexpr, V_OFF: tl.constexpr,
                     KV: tl.constexpr):
    t = tl.program_id(0)
    page = tl.load(Pages + t).to(tl.int64)
    cols = tl.arange(0, KV)
    row = QKV + t.to(tl.int64) * s_row
    k = tl.load(row + K_OFF + cols).to(tl.float32)
    v = tl.load(row + V_OFF + cols).to(tl.float32)
    tl.store(K + page * s_page_k + cols, k.to(K.dtype.element_ty))
    tl.store(V + page * s_page_v + cols, v.to(V.dtype.element_ty))


def write_kv(qkv: torch.Tensor, pages: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, q_size: int,
             kv_size: int) -> None:
    """qkv: [T, q | k | v] BF16 rows (unit column stride); pages: int [T]; k_cache, v_cache: one layer's
    [pages, 1, kv heads, head dim] (page size 1, contiguous pages). k_cache[pages[t]] = k row t, likewise v."""
    t = qkv.shape[0]
    if qkv.stride(1) != 1 or k_cache.shape[1] != 1 or k_cache[0].numel() != kv_size or not k_cache[0].is_contiguous():
        raise ValueError("write_kv needs row-major q|k|v rows and page-size-1 pages of kv_size contiguous values")
    if kv_size & (kv_size - 1):
        raise ValueError("write_kv: kv_size must be a power of two")
    if t:
        _write_kv_kernel[(t,)](qkv, qkv.stride(0), pages, k_cache, v_cache, k_cache.stride(0), v_cache.stride(0),
                               q_size, q_size + kv_size, kv_size, num_warps=2)
