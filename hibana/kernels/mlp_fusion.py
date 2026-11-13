"""27B MLP prefill: the gate/up GEMM with SiLU(gate) * up and the down projection's NVFP4 quantization in its epilogue.

The unfused prefill writes the gate/up GEMM's [M, 2I] BF16 output and reads it back to quantize SiLU(gate) * up for the
down projection (`nvfp4_quant.silu_mul_nvfp4_quantize`): 142 MB each way per 2,048-row chunk on one Spark. Here
FlashInfer's CuTe DSL SM120 block-scaled GEMM (`dense_blockscaled_gemm_sm120_b12x`, 128 x 128 tiles) finishes in
registers instead. The MLP's gate/up rows are interleaved in blocks of PAIRS (`weights.modelopt.interleave_pairs`), so a
tile's columns are 64 gate and the matching 64 up. Each thread scales its accumulators by alpha and rounds them to BF16
(the unfused GEMM's output), takes SiLU(gate) * up in FP32 rounded to BF16, and packs four products to E2M1 with the
group-16 E4M3 scale (amax across four lanes): the bytes the unfused path writes, without the BF16 tensor. One 2,048-row
chunk of a 27B MLP's gate/up and quantization: 2.78 -> 2.31 ms.

The kernel is FlashInfer's source (pinned in uv.lock) with its epilogue and arguments edited at first use. Every edit must
match exactly once; otherwise `available()` is False, prefill keeps the unfused path with a warning, and the GPU test
`test_mlp_fusion.py::test_available` fails."""
from __future__ import annotations

import inspect
import linecache
from functools import lru_cache
from types import FunctionType

import torch

PAIRS = 64  # gate/up interleave: one 128-column tile holds 64 gate columns and their 64 up columns
# Fused against unfused (gate/up GEMM then SiLU-mul quantization), the 27B's I = 17408 on one Spark / 8704 per rank on two:
# 32 rows x0.94 / x0.90, 128 x1.06 / x1.01, 512 x1.33 / x1.26, 2048 x1.20 / x1.30. Graph replays stay unfused.
MIN_ROWS = 128


def available(device: torch.device) -> bool:
    """Whether the fused kernel runs on `device` (compiling it on first use: call at load, not in a request)."""
    return device.type == "cuda" and _kernel(torch.cuda.current_device() if device.index is None else device.index) is not None


def applies(rows: int, gate_up) -> bool:
    """Whether a gate/up call of `rows` rows takes the fused kernel (eager prefill-sized calls; graphs stay unfused)."""
    return (rows >= MIN_ROWS and gate_up.pairs == PAIRS and available(gate_up.weight.device)
            and not torch.cuda.is_current_stream_capturing())


def gate_up_silu_quantize(pre: tuple[torch.Tensor, torch.Tensor], gate_up, global_scale: torch.Tensor):
    """`pre` = the NVFP4 gate/up input (packed [M, K/2], swizzled scales) -> SiLU(gate) * up quantized for the down
    projection with `global_scale`: (packed [M, I/2], swizzled scales), the bytes of `silu_mul_nvfp4_quantize` on the
    unfused GEMM's output."""
    from hibana.kernels.nvfp4_quant import _scale_buffer
    packed, factors = pre
    m, n, k = packed.shape[0], gate_up.out_features, gate_up.in_features
    inter = n // 2
    kernel = _kernel(packed.device.index)
    out = torch.empty((m, inter // 2), dtype=torch.uint8, device=packed.device)
    scales = _scale_buffer(m, inter, packed.device)
    unused = torch.empty((m, inter), dtype=torch.bfloat16, device=packed.device)  # C: its shape bounds the stores
    kernel(packed, gate_up.weight, unused, (m + 127) // 128, n // 128, (k // 16 + 3) // 4, factors.data_ptr(),
           gate_up.weight_scale_swizzled.data_ptr(), gate_up.alpha.reshape(1), out, scales, global_scale.reshape(1))
    return out, scales


# ------------------------------------------------------------------ FlashInfer's kernel, edited

_EPILOGUE = '''                if cutlass.const_expr(not self.swap_ab):
                    acc_mn = _reshape_acc_to_mn(accumulators)
                    identity = cute.make_identity_tensor((self.tile_shape_mnk[0], self.tile_shape_mnk[1]))
                    coords = _reshape_acc_to_mn(thr_mma.partition_C(identity))
                    for acc_m in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
                        for product_group in cutlass.range_constexpr(cute.size(acc_mn.shape[1]) // 8):
                            base = product_group * 4
                            up_offset = cute.size(acc_mn.shape[1]) // 2
                            coord = coords[acc_m, base]
                            other_coord = coords[acc_m, base + 2]
                            global_row = tile_coord_mnl[0] * Int32(self.tile_shape_mnk[0]) + coord[0]
                            global_column = tile_coord_mnl[1] * Int32(64) + coord[1]
                            other_column = tile_coord_mnl[1] * Int32(64) + other_coord[1]
                            # every lane computes: the group-16 maximum shuffles across lanes with a full mask, so a
                            # lane whose row lies past M (the last tile of M % 128 rows) must still take part
                            gate0 = (alpha_value * acc_mn[acc_m, base]).to(self.c_dtype).to(cutlass.Float32)
                            up0 = (alpha_value * acc_mn[acc_m, base + up_offset]).to(self.c_dtype).to(cutlass.Float32)
                            gate1 = (alpha_value * acc_mn[acc_m, base + 1]).to(self.c_dtype).to(cutlass.Float32)
                            up1 = (alpha_value * acc_mn[acc_m, base + 1 + up_offset]).to(self.c_dtype).to(cutlass.Float32)
                            gate2 = (alpha_value * acc_mn[acc_m, base + 2]).to(self.c_dtype).to(cutlass.Float32)
                            up2 = (alpha_value * acc_mn[acc_m, base + 2 + up_offset]).to(self.c_dtype).to(cutlass.Float32)
                            gate3 = (alpha_value * acc_mn[acc_m, base + 3]).to(self.c_dtype).to(cutlass.Float32)
                            up3 = (alpha_value * acc_mn[acc_m, base + 3 + up_offset]).to(self.c_dtype).to(cutlass.Float32)
                            first = fused_product(gate0, up0).to(cutlass.Float32)
                            second = fused_product(gate1, up1).to(cutlass.Float32)
                            third = fused_product(gate2, up2).to(cutlass.Float32)
                            fourth = fused_product(gate3, up3).to(cutlass.Float32)
                            encoded, other_encoded, factor = fused_quant_four(first, second, third, fourth, quant_scale)
                            if global_row < Int32(directC_mnl.shape[0]) and global_column < Int32(directC_mnl.shape[1]):
                                packed_output[(global_row, global_column // 2)] = encoded
                                packed_output[(global_row, other_column // 2)] = other_encoded
                                if global_column % 16 == 0:
                                    group = global_column // 16
                                    groups = output_scales.shape[1]
                                    sf_offset = (global_row // 128) * (128 * groups) + (global_row % 32) * 16 + ((global_row // 32) % 4) * 4 + (group // 4) * 512 + group % 4
                                    flattened = cute.make_tensor(output_scales.iterator, cute.make_layout(cute.size(output_scales)))
                                    flattened[sf_offset] = factor
                    if cutlass.const_expr(self.single_work_tile_per_cta):
                        work_tile = WorkTileInfo(work_tile.tile_idx, cutlass.Boolean(0))
                    else:
                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()

'''
_ARGUMENTS = "        packed_output: cute.Tensor,\n        output_scales: cute.Tensor,\n        global_scale: cute.Tensor,\n"
_KERNEL_EDITS = (
    # C is [M, I]: the GEMM's N is 2I, each tile writes 64 products per row
    ("        c_tensor = cute.make_tensor(\n            mC.iterator,\n            layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),\n        )",
     "        c_tensor = cute.make_tensor(\n            mC.iterator,\n            layout=cute.make_ordered_layout((m, n // 2, l), order=(1, 0, 2)),\n        )"),
    ("        alpha_value = alpha[0].to(cutlass.Float32)\n",
     "        alpha_value = alpha[0].to(cutlass.Float32)\n        quant_scale = global_scale[0]\n"),
    ("        stream: cuda.CUstream,\n        epilogue_op:", "        stream: cuda.CUstream,\n" + _ARGUMENTS + "        epilogue_op:"),
    ("        alpha: cute.Tensor,\n    ):", "        alpha: cute.Tensor,\n" + _ARGUMENTS + "    ):"),
    ("        current_stream,\n        swap_ab:", "        current_stream,\n" + _ARGUMENTS + "        swap_ab:"),
    ("            alpha,\n        ).launch(", "            alpha,\n            packed_output,\n            output_scales,\n            global_scale,\n        ).launch("),
    ("            current_stream,\n            epilogue_op,",
     "            current_stream,\n            packed_output,\n            output_scales,\n            global_scale,\n            epilogue_op,"),
)
_EPILOGUE_START = "                if cutlass.const_expr(not self.swap_ab):\n                    # EPILOGUE\n"
_EPILOGUE_END = "        elif warp_idx == self.tma_load_warp_id:\n"
_COMPILE_EDITS = (
    ("            c_fake = cute.runtime.make_fake_compact_tensor(\n                c_cutlass_dtype,\n                (sym_m, sym_n),",
     "            c_fake = cute.runtime.make_fake_compact_tensor(\n                c_cutlass_dtype,\n                (sym_m, sym_n // 2),"),
    ("        return cute.compile(\n",
     "        packed_fake = cute.runtime.make_fake_compact_tensor(\n"
     "            cutlass.Uint8, (sym_m, sym_n // 4), stride_order=(1, 0), assumed_align=16)\n"
     "        scales_fake = cute.runtime.make_fake_compact_tensor(\n"
     "            cutlass.Float8E4M3FN, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16)\n"
     "        global_scale_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)\n"
     "        return cute.compile(\n"),
    ("            stream_fake,\n            swap_ab,",
     "            stream_fake,\n            packed_fake,\n            scales_fake,\n            global_scale_fake,\n            swap_ab,"),
)


def _edit(source: str, edits) -> str:
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"FlashInfer source changed: {old.strip()[:60]!r}")
        source = source.replace(old, new)
    return source


def _module(source: str, namespace: dict, label: str) -> dict:
    filename = f"<hibana-mlp-fusion-{label}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    owned = dict(namespace)
    exec(compile(source, filename, "exec"), owned)
    return owned


@lru_cache(maxsize=1)
def _activation():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass._mlir.dialects import llvm

    @dsl_user_op
    def product(gate, up, *, loc=None, ip=None):
        exponential = cute.math.exp(-gate, fastmath=False, loc=loc, ip=ip)
        denominator = cutlass.Float32(1.) + exponential
        divided = llvm.inline_asm(
            T.f32(), [gate.ir_value(loc=loc, ip=ip), denominator.ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
        multiplied = llvm.inline_asm(
            T.f32(), [divided, up.ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
        return cutlass.Float32(multiplied).to(cutlass.BFloat16)

    return product


@lru_cache(maxsize=1)
def _quantize_four():
    import cutlass
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass._mlir.dialects import arith, llvm

    @dsl_user_op
    def instruction(operation, values, *, loc=None, ip=None):
        result = llvm.inline_asm(
            T.f32(), [value.ir_value(loc=loc, ip=ip) for value in values], operation,
            "=f," + ",".join("f" for value in values), has_side_effects=False,
            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
        return cutlass.Float32(result)

    @dsl_user_op
    def quant(first, second, third, fourth, scale, *, loc=None, ip=None):
        absolute_first = instruction("abs.f32 $0, $1;", [first], loc=loc, ip=ip)
        absolute_second = instruction("abs.f32 $0, $1;", [second], loc=loc, ip=ip)
        maximum = instruction("max.f32 $0, $1, $2;", [absolute_first, absolute_second], loc=loc, ip=ip)
        absolute_third = instruction("abs.f32 $0, $1;", [third], loc=loc, ip=ip)
        absolute_fourth = instruction("abs.f32 $0, $1;", [fourth], loc=loc, ip=ip)
        other_maximum = instruction("max.f32 $0, $1, $2;", [absolute_third, absolute_fourth], loc=loc, ip=ip)
        maximum = instruction("max.f32 $0, $1, $2;", [maximum, other_maximum], loc=loc, ip=ip)
        for distance in (2, 1):
            neighbor = instruction(f"shfl.sync.bfly.b32 $0, $1, {distance}, 31, -1;", [maximum], loc=loc, ip=ip)
            maximum = instruction("max.f32 $0, $1, $2;", [maximum, neighbor], loc=loc, ip=ip)
        reciprocal_six = instruction("rcp.approx.ftz.f32 $0, $1;", [cutlass.Float32(6.)], loc=loc, ip=ip)
        factor = (scale * (maximum * reciprocal_six)).to(cutlass.Float8E4M3FN)
        reciprocal_scale = instruction("rcp.approx.ftz.f32 $0, $1;", [scale], loc=loc, ip=ip)
        inverse = instruction("rcp.approx.ftz.f32 $0, $1;", [factor.to(cutlass.Float32) * reciprocal_scale], loc=loc, ip=ip)
        zero = cutlass.Float32(0.)
        is_zero = maximum == zero
        inverse = cutlass.Float32(arith.select(is_zero.ir_value(loc=loc, ip=ip),
                                               zero.ir_value(loc=loc, ip=ip), inverse.ir_value(loc=loc, ip=ip),
                                               loc=loc, ip=ip))
        first_scaled = instruction("mul.rn.f32 $0, $1, $2;", [first, inverse], loc=loc, ip=ip)
        second_scaled = instruction("mul.rn.f32 $0, $1, $2;", [second, inverse], loc=loc, ip=ip)
        third_scaled = instruction("mul.rn.f32 $0, $1, $2;", [third, inverse], loc=loc, ip=ip)
        fourth_scaled = instruction("mul.rn.f32 $0, $1, $2;", [fourth, inverse], loc=loc, ip=ip)
        saturation = "{ .reg .b32 sign, bits; .reg .f32 magnitude; mov.b32 sign, $1; and.b32 sign, sign, 0x80000000; abs.f32 magnitude, $1; min.f32 magnitude, magnitude, 0f40c00000; mov.b32 bits, magnitude; or.b32 bits, bits, sign; mov.b32 $0, bits; }"
        first_saturated = instruction(saturation, [first_scaled], loc=loc, ip=ip)
        second_saturated = instruction(saturation, [second_scaled], loc=loc, ip=ip)
        third_saturated = instruction(saturation, [third_scaled], loc=loc, ip=ip)
        fourth_saturated = instruction(saturation, [fourth_scaled], loc=loc, ip=ip)
        packed = llvm.inline_asm(
            T.i32(), [first_saturated.ir_value(loc=loc, ip=ip), second_saturated.ir_value(loc=loc, ip=ip)],
            "{ .reg .b8 encoded; cvt.rn.satfinite.e2m1x2.f32 encoded, $2, $1; cvt.u32.u8 $0, encoded; }",
            "=r,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
        other_packed = llvm.inline_asm(
            T.i32(), [third_saturated.ir_value(loc=loc, ip=ip), fourth_saturated.ir_value(loc=loc, ip=ip)],
            "{ .reg .b8 encoded; cvt.rn.satfinite.e2m1x2.f32 encoded, $2, $1; cvt.u32.u8 $0, encoded; }",
            "=r,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
        return cutlass.Int32(packed).to(cutlass.Uint8), cutlass.Int32(other_packed).to(cutlass.Uint8), factor

    return quant


@lru_cache(maxsize=None)
def _kernel(device_index: int):
    """The compiled kernel for this device, or None where it cannot run (not SM121, or the source edits fail)."""
    if torch.cuda.get_device_capability(device_index) != (12, 1):
        return None
    torch.zeros(1, device=f"cuda:{device_index}")  # a current CUDA context for FlashInfer's occupancy query
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass import utils
        from flashinfer.gemm import gemm_mm_fp4_cute_dsl as wrapper
        from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm120_b12x as native
        source = inspect.getsource(native.DenseGemmKernel)
        if source.count(_EPILOGUE_START) != 1 or source.count(_EPILOGUE_END) != 1:
            raise ValueError("FlashInfer source changed: epilogue bounds")
        before, rest = source.split(_EPILOGUE_START)
        _, after = rest.split(_EPILOGUE_END)
        source = _edit(before + _EPILOGUE + _EPILOGUE_END + after, _KERNEL_EDITS)
        namespace = dict(vars(native), fused_product=_activation(), fused_quant_four=_quantize_four())
        base = _module(source, namespace, "kernel")["DenseGemmKernel"]

        class Kernel(base):
            @staticmethod
            def _compute_grid(output, tile_shape, active_clusters, direct_scheduler, split_k):
                # tiles over the GEMM's [M, 2I], not C's [M, I]
                logical = cute.make_identity_tensor((output.shape[0], output.shape[1] * 2, output.shape[2]))
                tiled = cute.zipped_divide(logical, tiler=cute.slice_(tile_shape, (None, None, 0)))
                params = utils.PersistentTileSchedulerParams(tiled[(0, (None, None, None))].shape, (1, 1, 1),
                                                            swizzle_size=1, raster_along_m=True)
                return params, utils.StaticPersistentTileScheduler.get_grid_shape(params, active_clusters)

        compile_fn = _module(_edit(inspect.getsource(wrapper._make_blockscaled_gemm_compile_fn), _COMPILE_EDITS),
                             wrapper._compile_block_scaled_gemm.__globals__, "compile")
        function = wrapper._compile_block_scaled_gemm
        compile_gemm = FunctionType(function.__code__, compile_fn, function.__name__, function.__defaults__,
                                    function.__closure__)
        compiled, _ = compile_gemm({}, ("hibana_mlp_fusion",), lambda: Kernel(16, (128, 128), (1, 1), tile_k=128,
                                                                                use_m1_non_tma_c=True),
                                   ab_cutlass_dtype=cutlass.Uint8, sf_dtype=cutlass.Float8E4M3FN,
                                   c_cutlass_dtype=cutlass.BFloat16, ab_assumed_align=32, cluster_shape_mn=(1, 1),
                                   swap_ab=False, sf_m=1, sf_n=1, sf_k=1, batch_size=1, device_index=device_index)
    except Exception as exc:  # a FlashInfer or CuTe DSL change (edits, private names, compile errors): fall back
        import warnings
        warnings.warn(f"MLP prefill fusion unavailable ({exc}); prefill runs the unfused gate/up GEMM")
        return None
    return compiled
