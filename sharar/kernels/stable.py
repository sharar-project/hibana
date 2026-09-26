"""Shape-stable prefill numerics.

The target's prefill of a fixed prefix changes with the total row count M
because three kernels choose their arithmetic by M (measured in
target-numerics study, tag archive/pre-cleanup-2026-09-26): cuBLAS's BF16 GEMM for the GDN b/a
projection (six algorithm classes below 4096 rows, rounding flips in most
layers), cuBLASLt's FP8 GEMM for the 5120x6144 output projections (one
different algorithm for M in 27..32) and the gated RMS norm's rows-per-block
heuristic. With ``SHARAR_STABLE_NUMERICS=1`` each is replaced by an
M-invariant choice: a fixed-tile Triton kernel for b/a, zero-row padding over
the FP8 window, a constant rows-per-block. FlashInfer's ragged attention keeps
its length-dependent schedule (classes at 177, 209, 465, 497, 593 and 721
tokens); its effect on earlier rows is rare and documented.
"""
from __future__ import annotations

def enabled() -> bool:
    """SHARAR_STABLE_NUMERICS (EngineConfig.stable_numerics)."""
    from sharar.engine.config import current
    return current().stable_numerics


# (out_features, in_features) -> [(first unstable M, first stable M above it)]
FP8_SCALED_MM_WINDOWS: dict[tuple[int, int], list[tuple[int, int]]] = {(5120, 6144): [(27, 33)]}


def padded_rows(m: int, shape: tuple[int, int]) -> int:
    """Row count to run the FP8 GEMM at so that cuBLASLt keeps its wide-M algorithm."""
    for low, high in FP8_SCALED_MM_WINDOWS.get(shape, ()):
        if low <= m < high:
            return high
    return m
