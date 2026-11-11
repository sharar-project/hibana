# Copyright (c) 2025 by FlashInfer team. Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Modified by Sharar: quantizing epilogues.
"""The 27B's prefill residual norms with their quantization fused in (from FlashInfer 0.6.17's CuTe
`FusedAddRMSNormKernel`, `flashinfer/norm/kernels/fused_add_rmsnorm.py`; the epilogues come from the FlashPrefill V2 work).

The residual add, the reduction and the BF16 rounding are FlashInfer's, line for line: CuTe layouts are explicit, so
the added stores leave the arithmetic alone. Two epilogues read the BF16-rounded output in registers instead of
from memory:

- "fp8": the input norm. The BF16 output is still written (the mixer reads it) and also quantized as
  `kernels/fp8_quant.quant_fp8` does: divided by the scale with Triton's `div.full.f32`, clamped, converted.
- "nvfp4": the post-attention norm feeding the MLP. Only the NVFP4 activation and its swizzled block scales are
  written, with `kernels/nvfp4_quant`'s arithmetic (approximate reciprocals, E2M1 round-to-nearest-even).

Both are bitwise equal to the norm followed by the separate quantizer (tests/gpu/test_residual_norm_quant.py).
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import (
    cvt_and_store_8xf32_to_e4m3_hw,
    dsl_user_op,
    get_ptr_as_int64,
    get_sm_version,
    llvm,
    predicate_k,
    rcp_approx_ftz,
    row_reduce_sum_multirow,
)


@dsl_user_op
def clamp_native_fp8(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        Float32.mlir_type, [value.ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, 0fC3E00000; min.f32 $0, $0, 0f43E00000;",
        "=f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def divide_full(value: Float32, scale: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        Float32.mlir_type, [value.ir_value(loc=loc, ip=ip), scale.ir_value(loc=loc, ip=ip)],
        "div.full.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def store_fp8(values: cute.TensorSSA, output: cute.Tensor, scale: cute.Tensor,
              row: Int64, thread: Int32, threads_per_row: cutlass.Constexpr[int],
              vector_size: cutlass.Constexpr[int], blocks: cutlass.Constexpr[int]):
    fragment = cute.make_rmem_tensor(cute.make_layout(cute.size(values.shape)), Float32)
    fragment.store(values)
    lane = thread % threads_per_row
    global_scale = Float32(scale[0])
    for block in cutlass.range_constexpr(blocks):
        column = lane * vector_size + block * threads_per_row * vector_size
        for element in cutlass.range_constexpr(vector_size):
            quantized = divide_full(fragment[block * vector_size + element], global_scale)
            fragment[block * vector_size + element] = clamp_native_fp8(quantized)
        base = block * vector_size
        cvt_and_store_8xf32_to_e4m3_hw(
            fragment[base], fragment[base + 1], fragment[base + 2], fragment[base + 3],
            fragment[base + 4], fragment[base + 5], fragment[base + 6], fragment[base + 7],
            get_ptr_as_int64(output, cute.crd2idx((Int64(row), Int32(column)), output.layout)))


@dsl_user_op
def encode_e2m1(value: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        Int32.mlir_type,
        [Float32(value).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .f32 magnitude;
            .reg .b32 code, bits, sign;
            .reg .pred boundary;
            abs.f32 magnitude, $1;
            min.f32 magnitude, magnitude, 0f40c00000;
            mov.b32 code, 7;
            setp.le.f32 boundary, magnitude, 0f40a00000;
            selp.b32 code, 6, code, boundary;
            setp.lt.f32 boundary, magnitude, 0f40600000;
            selp.b32 code, 5, code, boundary;
            setp.le.f32 boundary, magnitude, 0f40200000;
            selp.b32 code, 4, code, boundary;
            setp.lt.f32 boundary, magnitude, 0f3fe00000;
            selp.b32 code, 3, code, boundary;
            setp.le.f32 boundary, magnitude, 0f3fa00000;
            selp.b32 code, 2, code, boundary;
            setp.lt.f32 boundary, magnitude, 0f3f400000;
            selp.b32 code, 1, code, boundary;
            setp.le.f32 boundary, magnitude, 0f3e800000;
            selp.b32 code, 0, code, boundary;
            mov.b32 bits, $1;
            shr.u32 sign, bits, 28;
            and.b32 sign, sign, 8;
            or.b32 $0, code, sign;
        }
        """,
        "=r,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def store_nvfp4(values: cute.TensorSSA, packed: cute.Tensor, factors: cute.Tensor, scale: cute.Tensor,
                row: Int64, thread: Int32, threads_per_row: cutlass.Constexpr[int],
                vector_size: cutlass.Constexpr[int], blocks: cutlass.Constexpr[int], sf_row_stride: cutlass.Constexpr[int]):
    fragment = cute.make_rmem_tensor(cute.make_layout(cute.size(values.shape)), Float32)
    fragment.store(values)
    lane = thread % threads_per_row
    global_scale = Float32(scale[0])
    reciprocal_scale = rcp_approx_ftz(global_scale)
    reciprocal_six = rcp_approx_ftz(Float32(6.0))
    for block in cutlass.range_constexpr(blocks):
        maximum = Float32(0.0)
        for element in cutlass.range_constexpr(vector_size):
            absolute = abs(fragment[block * vector_size + element])
            if absolute > maximum:
                maximum = absolute
        adjacent = cute.arch.shuffle_sync_bfly(maximum, offset=1)
        if adjacent > maximum:
            maximum = adjacent
        factor = (global_scale * (maximum * reciprocal_six)).to(factors.element_type)
        inverse = Float32(0.0)
        if maximum != Float32(0.0):
            inverse = rcp_approx_ftz(factor.to(Float32) * reciprocal_scale)
        column = lane * vector_size + block * threads_per_row * vector_size
        for pair in cutlass.range_constexpr(vector_size // 2):
            lower = encode_e2m1(fragment[block * vector_size + pair * 2] * inverse)
            upper = encode_e2m1(fragment[block * vector_size + pair * 2 + 1] * inverse)
            packed[row, column // 2 + pair] = (lower | (upper << 4)).to(cutlass.Uint8)
        if lane % 2 == 0:
            group = column // 16
            offset = row // 128 * (128 * sf_row_stride) + row % 32 * 16 + row // 32 % 4 * 4 + group // 4 * 512 + group % 4
            factors[offset] = factor


class ResidualNormKernel(FusedAddRMSNormKernel):
    """FlashInfer's fused residual add + Gemma RMS norm with a quantizing epilogue (`out`: "fp8" or "nvfp4")."""

    def __init__(self, H: int, out: str, sm_version: int):
        super().__init__(cutlass.BFloat16, H, weight_bias=1.0, sm_version=sm_version)
        if out not in ("fp8", "nvfp4") or self.vec_size != 8 or self.cluster_n != 1 or self.threads_per_row % 2:
            raise ValueError(f"unsupported residual norm epilogue {out!r} at H={H}")
        self.out = out
        self.sf_row_stride = (H // 16 + 3) // 4 * 4  # swizzled block-scale columns, padded to 4

    @cute.jit
    def __call__(self, mX: cute.Tensor, mR: cute.Tensor, mW: cute.Tensor, mQ: cute.Tensor, mSF: cute.Tensor,
                 mGS: cute.Tensor, M: Int64, eps: Float32, enable_pdl: cutlass.Constexpr[bool], stream):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(self.threads_per_row, self.rows_per_block, self.vec_size,
                                                            self.num_vec_blocks)
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        self.kernel(mX, mR, mW, mQ, mSF, mGS, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1], block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(), stream=stream, use_pdl=enable_pdl)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        mGS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        # Partition residual (same load tiled copy)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output destinations
        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + residual, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        # Write h to residual (global)
        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        # sharar: the epilogues read the BF16-rounded output, as the separate quantizers read it back from memory.
        if cutlass.const_expr(self.out == "fp8"):
            tXrO.store(y.to(mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)
                store_fp8(y.to(mX.element_type).to(Float32), mQ, mGS, row_coord[0], tidx, threads_per_row,
                          self.vec_size, self.num_vec_blocks)
        else:  # "nvfp4": only the quantized output; nothing reads the BF16 one
            if row_in_bounds:
                store_nvfp4(y.to(mX.element_type).to(Float32), mQ, mSF, mGS, row_coord[0], tidx, threads_per_row,
                            self.vec_size, self.num_vec_blocks, self.sf_row_stride)

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compiled(H: int, out: str, device_index: int):
    kernel = ResidualNormKernel(H, out, get_sm_version(torch.device("cuda", device_index)))
    rows = cute.sym_int(64)
    align = math.gcd(128, H * 2)
    tensor = lambda dtype, shape, a=16: cute.runtime.make_fake_compact_tensor(  # noqa: E731
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=a)
    q = tensor(cutlass.Float8E4M3FN, (rows, H)) if out == "fp8" else tensor(cutlass.Uint8, (rows, H // 2))
    with torch.cuda.device(device_index):
        return cute.compile(kernel, tensor(cutlass.BFloat16, (rows, H), align), tensor(cutlass.BFloat16, (rows, H), align),
                            tensor(cutlass.BFloat16, (H,)), q, tensor(cutlass.Float8E4M3FN, (cute.sym_int(64),)),
                            tensor(Float32, (1,)), Int64(1), Float32(1e-6), True,
                            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")


def supported(x: torch.Tensor, residual: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype == residual.dtype == torch.bfloat16 and x.dim() == 2 and x.shape == residual.shape
            and x.is_contiguous() and residual.is_contiguous() and x.shape[1] % 16 == 0
            and x.numel() < 2**31 and x.data_ptr() != residual.data_ptr())


def residual_norm_fp8(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                      scale: torch.Tensor) -> torch.Tensor:
    """residual += x; x = gemma_rmsnorm(residual) in BF16 (both in place); returns quant_fp8(x, scale)."""
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    _compiled(x.shape[1], "fp8", x.device.index)(x, residual, weight, q, q.view(-1)[:1], scale.view(1), x.shape[0], eps)
    return q


def residual_norm_nvfp4(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                        global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """residual += x in place; returns nvfp4_quantize(gemma_rmsnorm(residual), global_scale). x is not written."""
    from hibana.kernels.nvfp4_quant import _scale_buffer
    m, h = x.shape
    packed = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    factors = _scale_buffer(m, h, x.device)
    _compiled(h, "nvfp4", x.device.index)(x, residual, weight, packed, factors.view(-1), global_scale.view(1), m, eps)
    return packed, factors
