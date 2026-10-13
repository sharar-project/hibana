"""GEMM choices for the 27B's prefill: CUTLASS tactics for the NVFP4 MLP GEMMs, cuBLASLt algorithms for the FP8
projections.

Every one of FlashInfer 0.6.17's 32 CUTLASS FP4 tactics produces bitwise-identical output for these shapes:
the tactics differ in tiling and schedule, not in the K reduction order. The choice therefore only moves time.
FlashInfer's default (heuristic) tactic is up to 1.2x slower than the best one for the gate/up projection from
1,024 rows, and up to 1.65x slower for the down projection at the 3-4K rows a batched prefill reaches
(`profiling/prefill_gemm_tactics.py` measures every tactic and checks the output; `tests/gpu/test_prefill_gemm.py`).
Below the listed row counts the default stays: the measured gains there are within noise.

The FP8 projections (attention q/k/v and o, GDN in_qkv, in_z, out) run cuBLASLt through torch._scaled_mm. For a full
2,048-row prefill chunk, every algorithm in FlashInfer's cuBLASLt enumeration gives the same bits as _scaled_mm, and
the listed one is 1-9% faster (about 9 ms per chunk in all). The enumeration is per exact shape, so the choice applies
at 2,048 rows only, and the first call of each shape checks it against _scaled_mm bit for bit (a different cuBLAS may
enumerate differently): on a mismatch that shape keeps _scaled_mm.
"""

from __future__ import annotations

from functools import lru_cache

import torch

# (out_features, in_features) of an NVFP4 linear: (fewest rows, tactic). The 27B's MLP gate/up and down projections.
TACTICS = {(34816, 5120): (1024, 24), (5120, 17408): (1536, 25)}
# (out_features, in_features) of an FP8 linear at FP8_ROWS rows: index into FlashInfer's cuBLASLt enumeration.
FP8_ROWS = 2048
FP8_ALGOS = {(10240, 5120): 2, (6144, 5120): 2, (5120, 6144): 2, (14336, 5120): 0}
_fp8_checked: dict[tuple[int, int], bool] = {}


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


@lru_cache(maxsize=None)
def _fp8_runner():
    from flashinfer.gemm.gemm_base import get_gemm_module
    return get_gemm_module().cublas_fp8_gemm_runner()


def fp8_gemm(x_q: torch.Tensor, lin) -> torch.Tensor | None:
    """The FP8 projection with its chosen cuBLASLt algorithm, or None for torch._scaled_mm."""
    n, k = lin.out_features, lin.weight.shape[1]
    algo = FP8_ALGOS.get((n, k))
    if algo is None or x_q.shape[0] != FP8_ROWS or _fp8_checked.get((n, k)) is False:
        return None
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf
    out = torch.empty((1, FP8_ROWS, n), device=x_q.device, dtype=torch.bfloat16)
    workspace = _get_cache_buf("bmm_fp8_workspace", DEFAULT_WORKSPACE_SIZE, x_q.device)
    _fp8_runner().forward([x_q[None], lin.weight.t()[None], lin.input_scale, lin.weight_scale, out, workspace], tactic=algo)
    if (n, k) not in _fp8_checked:  # once per shape: the same bits as the default dispatch, or keep the default
        reference = torch._scaled_mm(x_q, lin.weight.t(), scale_a=lin.input_scale.reshape(1, 1),
                                     scale_b=lin.weight_scale.reshape(1, 1), out_dtype=torch.bfloat16)
        _fp8_checked[n, k] = torch.equal(out[0].view(torch.int16), reference.view(torch.int16))
        if not _fp8_checked[n, k]:
            return reference
    return out[0]
