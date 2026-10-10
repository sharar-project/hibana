"""32x32 specialization of FlashInfer's hierarchical FP16 inverse.

Retains the 128-thread barrier contract; inactive warps still participate.
The 64 path delegates to the unmodified installed implementation.
"""
import cutlass
import cutlass.cute as cute
from flashinfer.gdn_kernels.delta_rule_dsl.collective_inverse_hmma import (
    CollectiveInverse as InstalledInverse,
)


class CollectiveInverse(InstalledInverse):
    @cute.jit
    def run(self, sT: cute.Tensor, barrier_id: cutlass.Int32,
            chunk_size: cutlass.Constexpr):
        if cutlass.const_expr(chunk_size == 64):
            InstalledInverse.run(self, sT, barrier_id)
        else:
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % 128
            t8x8 = cute.flat_divide(sT, (8, 8))
            if thread_idx < 32:
                blk = thread_idx // 8
                self.compute_diagonal_inverse_NxN(
                    t8x8[None, None, blk, blk], thread_idx % 8, 8)
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
            t16x16 = cute.flat_divide(sT, (16, 16))
            if thread_idx < 64:
                blk = thread_idx // 32
                self.blockwise_8x8_to_16x16(t16x16[None, None, blk, blk])
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
            if thread_idx < 32:
                self.blockwise_16x16_to_32x32(sT)
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
