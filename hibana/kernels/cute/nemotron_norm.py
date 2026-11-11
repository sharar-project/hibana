# Copyright (c) 2025 by FlashInfer team. Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Modified by Hibana: separate outputs and the FP8 epilogue.
"""Nemotron's residual norms in one launch: FlashInfer 0.6.17's CuTe `FusedAddRMSNormKernel` (the kernel
`flashinfer.norm.fused_add_rmsnorm` runs on SM121) at weight_bias 0, with outputs of their own and
`residual_norm.py`'s FP8 epilogue.

FlashInfer's kernel works in place (residual += x, x = norm(residual)), so a caller that must keep x or the old
residual clones them first, and the FP8 copy a projection reads is a third launch (`quant_fp8`). Here the
normalized row, the new residual and the FP8 copy go to tensors of the caller's choosing (any of them may be the
input itself): the residual add, the reduction and the BF16 rounding are FlashInfer's line for line, so the outputs
are bitwise those of clone + fused_add_rmsnorm + quant_fp8 (tests/gpu/test_nemotron_fused.py).
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import get_sm_version, predicate_k, row_reduce_sum_multirow

from hibana.kernels.cute.residual_norm import store_fp8


class NemotronNormKernel(FusedAddRMSNormKernel):
    """residual_out = x + residual; out = rmsnorm(residual_out) * weight (BF16); optionally q = quant_fp8(out)."""

    def __init__(self, H: int, fp8: bool, sm_version: int):
        super().__init__(cutlass.BFloat16, H, weight_bias=0.0, sm_version=sm_version)
        if self.vec_size != 8 or self.cluster_n != 1 or self.threads_per_row % 2:
            raise ValueError(f"unsupported Nemotron residual norm at H={H}")
        self.fp8 = fp8

    @cute.jit
    def __call__(self, mX: cute.Tensor, mR: cute.Tensor, mW: cute.Tensor, mO: cute.Tensor, mRO: cute.Tensor,
                 mQ: cute.Tensor, mGS: cute.Tensor, M: Int64, eps: Float32, enable_pdl: cutlass.Constexpr[bool], stream):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(self.threads_per_row, self.rows_per_block, self.vec_size,
                                                            self.num_vec_blocks)
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        self.kernel(mX, mR, mW, mO, mRO, mQ, mGS, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1], block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(), stream=stream, use_pdl=enable_pdl)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mR: cute.Tensor, mW: cute.Tensor, mO: cute.Tensor, mRO: cute.Tensor,
               mQ: cute.Tensor, mGS: cute.Tensor, M: Int64, eps: Float32, enable_pdl: cutlass.Constexpr[bool],
               tv_layout: cute.Layout, tiler_mn: cute.Shape):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
            sR = smem.allocate_tensor(mR.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
        reduction_buffer = smem.allocate_tensor(Float32, cute.make_layout((rows_per_block, warps_per_row)), byte_alignment=4)
        mbar_ptr = None

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        gO = cute.local_tile(mO, tiler_mn, (bidx, cluster_y))
        gRO = cute.local_tile(mRO, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        copy_atom_sync = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits)
        copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits)
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type,
                                                  num_bits_per_copy=copy_bits)
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)
        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)
        # hibana: the outputs' own destinations (FlashInfer stores back into gX and gR)
        tXgO = thr_copy_O.partition_D(gO)
        tRgO = thr_copy_O.partition_D(gRO)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

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
        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(h_sq, threads_per_row, reduction_buffer, mbar_ptr, 1)
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
        cute.arch.barrier()

        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))
        tXrO.store(y.to(mX.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)
            if cutlass.const_expr(self.fp8):  # quant_fp8 of the BF16-rounded row (residual_norm.py's epilogue)
                store_fp8(y.to(mX.element_type).to(Float32), mQ, mGS, row_coord[0], tidx, threads_per_row,
                          self.vec_size, self.num_vec_blocks)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compiled(H: int, fp8: bool, device_index: int):
    kernel = NemotronNormKernel(H, fp8, get_sm_version(torch.device("cuda", device_index)))
    rows = cute.sym_int(64)
    align = math.gcd(128, H * 2)
    tensor = lambda dtype, shape, a=16: cute.runtime.make_fake_compact_tensor(  # noqa: E731
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=a)
    bf16 = lambda: tensor(cutlass.BFloat16, (rows, H), align)  # noqa: E731
    q = tensor(cutlass.Float8E4M3FN, (rows, H)) if fp8 else tensor(cutlass.Float8E4M3FN, (cute.sym_int(64),))
    with torch.cuda.device(device_index):
        return cute.compile(kernel, bf16(), bf16(), tensor(cutlass.BFloat16, (H,)), bf16(), bf16(), q,
                            tensor(Float32, (1,)), Int64(1), Float32(1e-6), True,
                            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")


def supported(x: torch.Tensor, residual: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype == residual.dtype == torch.bfloat16 and x.dim() == 2 and x.shape == residual.shape
            and x.is_contiguous() and residual.is_contiguous() and x.shape[1] % 16 == 0 and x.numel() < 2**31
            and x.data_ptr() != residual.data_ptr())


def residual_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor,
                  residual_out: torch.Tensor, scale: torch.Tensor | None = None,
                  placeholders: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor | None:
    """residual_out = x + residual (BF16); out = rmsnorm(residual_out) * weight in FlashInfer's arithmetic; with
    `scale`, also returns quant_fp8(out, scale). `out` may be x and `residual_out` may be residual (in place); both
    must be contiguous like x. `placeholders` (FP8 [1], FP32 [1], allocated outside any graph capture) stand in for
    the FP8 output and scale of a call without `scale`."""
    m, h = x.shape
    if not (out.is_contiguous() and residual_out.is_contiguous() and out.shape == residual_out.shape == x.shape):
        raise ValueError("residual_norm outputs must be contiguous [rows, H] like x")
    if scale is None:
        if placeholders is None:
            placeholders = (torch.empty(1, dtype=torch.float8_e4m3fn, device=x.device),
                            torch.ones(1, dtype=torch.float32, device=x.device))
        q, gs = placeholders
    else:
        q, gs = torch.empty_like(x, dtype=torch.float8_e4m3fn), scale.view(1)
    if m:
        _compiled(h, scale is not None, x.device.index)(x, residual, weight, out, residual_out, q, gs, m, eps)
    return q if scale is not None else None
