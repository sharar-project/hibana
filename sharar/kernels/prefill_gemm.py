"""CUTLASS tactics for the 27B's NVFP4 MLP GEMMs at prefill row counts.

Every one of FlashInfer 0.6.17's 32 CUTLASS FP4 tactics produces bitwise-identical output for these shapes:
the tactics differ in tiling and schedule, not in the K reduction order. The choice therefore only moves time.
FlashInfer's default (heuristic) tactic is up to 1.2x slower than the best one for the gate/up projection from
1,024 rows, and up to 1.65x slower for the down projection at the 3-4K rows a batched prefill reaches
(`profiling/prefill_gemm_tactics.py` measures every tactic and checks the output; `tests/gpu/test_prefill_gemm.py`).
Below the listed row counts the default stays: the measured gains there are within noise.
"""

from __future__ import annotations

from functools import lru_cache

import torch

# (out_features, in_features) of an NVFP4 linear: (fewest rows, tactic). The 27B's MLP gate/up and down projections.
TACTICS = {(34816, 5120): (1024, 24), (5120, 17408): (1536, 25)}


def tactic(m: int, n: int, k: int) -> int | None:
    """The tactic for an [m, k] x [k, n] FP4 GEMM, or None for FlashInfer's default."""
    rule = TACTICS.get((n, k))
    return rule[1] if rule is not None and m >= rule[0] else None


@lru_cache(maxsize=None)
def _runner():
    from flashinfer.gemm.gemm_base import get_gemm_sm120_module_cutlass_fp4
    return get_gemm_sm120_module_cutlass_fp4().cutlass_fp4_gemm_runner()


def fp4_gemm(x_fp4: torch.Tensor, x_sf: torch.Tensor, lin, chosen: int) -> torch.Tensor:
    """`mm_fp4(..., backend="cutlass")` for a BF16 output with the given tactic."""
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf
    out = torch.empty((x_fp4.shape[0], lin.out_features), device=x_fp4.device, dtype=torch.bfloat16)
    workspace = _get_cache_buf("mm_fp4_workspace", DEFAULT_WORKSPACE_SIZE, x_fp4.device)
    # The runner's sixth, eighth and ninth inputs (output dtype, block size, layout flag) are unused by the CUTLASS path.
    _runner().forward([x_fp4, lin.weight.T, x_sf, lin.weight_scale_swizzled.T, lin.alpha, None, out, None, None,
                       workspace], tactic=chosen)
    return out
