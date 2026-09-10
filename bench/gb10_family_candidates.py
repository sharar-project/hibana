"""Candidate hooks for `bench.gb10_family_eval run --candidate bench.gb10_family_candidates:<name>`.

Each hook is called as fn(stage, executor, args) at 'model' (executor is None),
'spec' (after enable_spec, before graph capture) and 'ready' (after warmup).
Patches applied at 'spec' persist for the process so every captured graph and
every eager fallback uses the candidate path.
"""
from contextlib import ExitStack

_STACK = ExitStack()


def exact_bundle(stage, ex, args):
    """cuBLAS NvJet for the 16 attention QKV projections plus CuTe K=512 tiles for
    the 128 MLP GEMMs: validated bitwise-identical on captured operands
    (docs/gb10-verifier16.md), measured +1.4% decode on the code prompt."""
    if stage == 'spec':
        from bench.gb10_matrix import fp4_mlp_candidate, qkv_candidate
        _STACK.enter_context(qkv_candidate())
        _STACK.enter_context(fp4_mlp_candidate())


def qkv_cublas(stage, ex, args):
    if stage == 'spec':
        from bench.gb10_matrix import qkv_candidate
        _STACK.enter_context(qkv_candidate())


def fp4_k512(stage, ex, args):
    if stage == 'spec':
        from bench.gb10_matrix import fp4_mlp_candidate
        _STACK.enter_context(fp4_mlp_candidate())


def top32(stage, ex, args):
    """Widen the DFlash2 conditional lattice to the top-32 candidates per position
    (selector codebooks are gathered per candidate, so any width up to the lattice
    kernel's 32 is supported at negligible extra cost)."""
    if stage == 'spec':
        cfg = ex.spec.proposer.model.cfg
        assert cfg['dflash_config']['selector_top_k'] == 16
        cfg['dflash_config']['selector_top_k'] = 32


# ----------------------------------------------------------------- draft bytes

def _nvfp4_from_bf16(weight, ops):
    """Quantize a BF16 [N, K] matrix to NVFP4 with FlashInfer's quantizer (swizzled
    block scales) and return (packed, sf, weight_scale_2) where weight_scale_2 is the
    dequantization multiplier 1 / global_scale."""
    import torch
    amax = weight.float().abs().amax().clamp_min(1e-12)
    global_scale = (448.0 * 6.0 / amax).float().reshape(1)
    packed, sf = ops.fp4_quantize(weight.contiguous(), global_scale)
    n, k = weight.shape
    n_pad, k_groups = (n + 127) // 128 * 128, (k // 16 + 3) // 4 * 4
    sf = sf.reshape(n_pad, k_groups).view(torch.float8_e4m3fn)
    return packed.view(n, k // 2), sf, (1.0 / global_scale).float().reshape(1)


class _W4A16:
    """BF16-activation x NVFP4-weight linear through flashinfer.gemm.mm_bf16_fp4 (CuTe DSL, SM121)."""

    def __init__(self, weight_bf16):
        import torch
        from flashinfer.gemm import mm_bf16_fp4, prepare_bf16_fp4_weights
        from sharar.kernels import ops
        self.out_features, self.in_features = weight_bf16.shape
        packed, sf, ws = _nvfp4_from_bf16(weight_bf16, ops)
        self.prepared = prepare_bf16_fp4_weights(packed, sf, ws, backend='cute-dsl')
        self.mm = mm_bf16_fp4
        self.bytes = packed.numel() + sf.numel()
        # Startup oracle: the kernel must reproduce x @ dequant(W).T within 1% NRMSE.
        from bench.verifier_numerics import unpack_fp4, unswizzle
        decoded = unpack_fp4(packed, unswizzle(sf, self.out_features, self.in_features // 16)) * ws
        x = torch.randn(16, self.in_features, device=weight_bf16.device, dtype=torch.bfloat16)
        got = self(x).float()
        want = x.float() @ decoded.float().T
        nrmse = float((got - want).norm() / want.norm().clamp_min(1e-30))
        assert torch.isfinite(got).all() and nrmse < 0.01, ('W4A16 oracle mismatch', nrmse)
        self.oracle_nrmse = nrmse

    def __call__(self, x):
        flat = x.reshape(-1, x.shape[-1])
        if not flat.is_contiguous():
            flat = flat.contiguous()
        out = self.mm(flat, *self.prepared, backend='cute-dsl', enable_pdl=True)
        return out.reshape(*x.shape[:-1], self.out_features)


def _convert_draft_body(model, names):
    import types
    converted = {}
    for name in names:
        weight = model.weights.pop(name)
        converted[name] = _W4A16(weight)
    del weight
    native = model._linear
    def linear(self, x, name):
        lin = converted.get(name)
        if lin is not None:
            return lin(x)
        return native(x, name)
    model._linear = types.MethodType(linear, model)
    return converted


def draft_nvfp4_body(stage, ex, args):
    """Draft q/o/gate_up/down as NVFP4 weights with BF16 activations (W4A16).
    Run with --draft-precision bf16 so the original BF16 matrices are quantized
    once; the target is untouched, only proposals can change."""
    if stage != 'spec':
        return
    import torch
    assert args.draft_precision == 'bf16', 'draft_nvfp4_body needs --draft-precision bf16'
    model = ex.spec.proposer.model
    names = [f'layers.{i}.{s}' for i in range(model.cfg['num_hidden_layers'])
             for s in ('self_attn.q_proj.weight', 'self_attn.o_proj.weight', 'mlp.gate_up', 'mlp.down_proj.weight')]
    converted = _convert_draft_body(model, names)
    torch.cuda.synchronize()
    total = sum(v.bytes for v in converted.values())
    worst = max(v.oracle_nrmse for v in converted.values())
    print(f'  draft_nvfp4_body: {len(converted)} matrices, {total / 2**20:.0f} MiB NVFP4, worst oracle NRMSE {worst:.4%}', flush=True)


def draft_nvfp4_body_fc8(stage, ex, args):
    """draft_nvfp4_body plus the 262 MB BF16 feature projection (fc) as dynamic FP8."""
    draft_nvfp4_body(stage, ex, args)
    if stage != 'spec':
        return
    import torch
    from sharar.model.draft_linear import DraftFp8Linear
    from sharar.model.parallel_draft import rms
    model = ex.spec.proposer.model
    fc = DraftFp8Linear.from_bf16(model.weights.pop('fc.weight'))
    def project_features(features):
        joined = torch.cat([features[i] for i in model.features], dim=-1)
        return rms(fc(joined), model.weights['hidden_norm.weight'])
    model.project_features = project_features
    print(f'  draft fc as FP8: {fc.bytes / 2**20:.0f} MiB', flush=True)


def fp8_m16(stage, ex, args):
    """Route every M=16 target FP8 projection through the Triton weight-streaming
    kernel (214-235 GB/s measured, versus 156-214 for torch._scaled_mm). Numerics
    change in the last BF16 bit: judged by the 21-task protocol."""
    if stage != 'spec':
        return
    from bench.gb10_fp8_m16 import fp8_gemm_m16, supported, validate
    from sharar.kernels import ops
    report = validate()
    print(f"  fp8_m16 validation vs _scaled_mm: nrmse {report['nrmse']:.3e}, max_abs {report['max_abs']:.3e}, "
          f"argmax_equal {report['argmax_equal']}, exact {report['exact']}", flush=True)
    assert report['nrmse'] < 1e-3 and report['argmax_equal'], report
    native = ops.fp8_linear

    def linear(x, lin, x_q=None):
        if x_q is None:
            x2 = x.reshape(-1, x.shape[-1])
            x_q = ops.quant_fp8(x2, lin.input_scale)
            lead = x.shape[:-1]
        else:
            lead = x_q.shape[:-1]
        if supported(x_q, lin):
            return fp8_gemm_m16(x_q, lin.weight, lin.alpha).view(*lead, lin.out_features)
        return native(None, lin, x_q).view(*lead, lin.out_features)

    ops.fp8_linear = linear
    _STACK.callback(lambda: setattr(ops, 'fp8_linear', native))


def exact_bundle_fp8_m16(stage, ex, args):
    exact_bundle(stage, ex, args)
    fp8_m16(stage, ex, args)
