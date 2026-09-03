#!/usr/bin/env python3
"""A/B one NVFP4 linear: Sharar's loader+wrapper versus SGLang's helpers.

Runs under tools/dev-shell.sh --with-reference. Uses layer 0 gate_up from the
checkpoint and a real activation (post-norm of the dumped residual).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/sharar")
from sharar import config  # noqa: E402
from sharar.kernels import ops  # noqa: E402
from sharar.model.norm import GemmaRMSNorm  # noqa: E402
from sharar.model.qwen38 import PREFIX  # noqa: E402
from sharar.weights.modelopt import load_nvfp4_linear, swizzle_block_scales  # noqa: E402
from sharar.weights.safetensors import SnapshotWeights  # noqa: E402

device = torch.device("cuda")
snap = config.snapshot_dir()
w = SnapshotWeights(snap)
ref = torch.load(Path(sys.argv[1]) / "dump" / "rand8.pt", map_location="cpu")
h0, r0 = ref["layers"][0]
post_norm = GemmaRMSNorm(w.get(f"{PREFIX}.layers.0.post_attention_layernorm.weight").to(device), config.RMS_NORM_EPS)
x = post_norm(r0.to(device))

mine = load_nvfp4_linear(w, [f"{PREFIX}.layers.0.mlp.gate_proj", f"{PREFIX}.layers.0.mlp.up_proj"], device)
out_mine = ops.nvfp4_linear(x, mine)

# --- SGLang side ---
from sglang.srt.layers.quantization import fp4_utils  # noqa: E402
from sglang.srt.layers.quantization.modelopt_quant import fp4_gemm  # noqa: E402

print("sglang fp4 backend:", fp4_utils.get_fp4_gemm_runner_backend(), "quant backend:", fp4_utils._flashinfer_fp4_quantize_backend)
x_fp4_s, x_sf_s = fp4_utils.fp4_quantize(x, mine.input_scale_inv)
x_fp4_m, x_sf_m = ops.fp4_quantize(x, mine.input_scale_inv)
print("fp4_quantize data equal:", torch.equal(x_fp4_s, x_fp4_m), "sf equal:", torch.equal(x_sf_s, x_sf_m), x_sf_s.dtype, tuple(x_sf_s.shape))
out_s = fp4_gemm(x_fp4_s, mine.weight.T, x_sf_s, mine.weight_scale_swizzled.T, mine.alpha, x.dtype, mine.out_features)
print("sglang fp4_gemm vs mine (same weights):", (out_s.float() - out_mine.float()).abs().max().item())


def rel(a, b):
    a = a.float(); b = b.float().to(a.device)
    d = (a - b).abs(); s = b.abs().mean().item()
    return f"max {d.max().item() / s:.4f} mean {d.mean().item() / s:.5f}"


# --- torch reference with explicit NVFP4 activation quantization ---
FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=device)


def unpack(wt):
    lo = (wt & 0xF).long(); hi = (wt >> 4).long()
    return torch.stack([FP4[lo], FP4[hi]], -1).reshape(wt.shape[0], -1)


raw_scale = torch.cat([w.get(f"{PREFIX}.layers.0.mlp.gate_proj.weight_scale"), w.get(f"{PREFIX}.layers.0.mlp.up_proj.weight_scale")], 0).to(device).float()
ws2 = w.get(f"{PREFIX}.layers.0.mlp.gate_proj.weight_scale_2").float().item()
in_scale = w.get(f"{PREFIX}.layers.0.mlp.gate_proj.input_scale").float().item()
W = unpack(mine.weight)
W = (W.reshape(W.shape[0], -1, 16) * raw_scale.unsqueeze(-1)).reshape(W.shape[0], -1) * ws2
ref_bf16act = x.float() @ W.T
print("mine vs dequant(W) @ x(bf16):", rel(out_mine, ref_bf16act))


def quant_act(xa, input_scale):
    xb = xa.float().reshape(xa.shape[0], -1, 16)
    amax = xb.abs().amax(-1, keepdim=True)
    sf = (amax / 6.0) / input_scale  # e4m3 block scale in units of input_scale
    sf = sf.to(torch.float8_e4m3fn).float()
    scaled = xb / (sf * input_scale).clamp_min(1e-12)
    # round to nearest e2m1 value
    cand = FP4[:8]
    idx = (scaled.abs().unsqueeze(-1) - cand).abs().argmin(-1)
    q = cand[idx] * scaled.sign()
    return (q * sf * input_scale).reshape(xa.shape[0], -1)


xq = quant_act(x, in_scale)
ref_fp4act = xq @ W.T
print("mine vs dequant(W) @ quant(x):", rel(out_mine, ref_fp4act))
print("act-quant-only error (quant(x)@W vs x@W):", rel(ref_fp4act, ref_bf16act))
# swizzle sanity: unswizzled scale round trip
sw = swizzle_block_scales(raw_scale.to(torch.float8_e4m3fn))
print("swizzled scale shape", tuple(sw.shape), "sum equal:", torch.equal(sw.float().sum(), raw_scale.sum()))
