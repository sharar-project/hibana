"""`rms_norm_gated_quant`: the GDN layer's gated RMS norm writing the out projection's FP8 input itself, the arithmetic of
the norm then `quant_fp8` (the two compiled kernels may round a product to BF16 on either side of a tie: a handful of
values in millions move one E4M3 step)."""
import pytest
import torch


@pytest.mark.parametrize("rows", [1, 7, 2048, 5000])
def test_rms_norm_gated_quant(rows):
    from sharar.kernels.fla.layernorm_gated import rms_norm_gated, rms_norm_gated_quant
    from sharar.kernels.fp8_quant import quant_fp8
    torch.manual_seed(rows)
    hv, v = 48, 128
    x = torch.randn(rows, hv * v, device="cuda", dtype=torch.bfloat16) * 3
    z = torch.randn(rows, 2 * hv * v, device="cuda", dtype=torch.bfloat16)[:, hv * v:]  # a strided row, unit columns
    w = torch.randn(hv * v, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.0123, device="cuda")
    ref = quant_fp8(rms_norm_gated(x=x, weight=w, z=z, eps=1e-6, group_size=v, activation="swish"), scale)
    out = rms_norm_gated_quant(x=x, weight=w, z=z, scale=scale, eps=1e-6, group_size=v, activation="swish")
    moved = out.float() != ref.float()
    assert moved.float().mean() < 1e-5, moved.sum()
    # a moved value is the neighbouring E4M3 level: within one step (1/8 of its magnitude, 1/4 below 2**-6)
    assert ((out.float() - ref.float()).abs() <= ref.float().abs() * 0.125 + 2 ** -9)[moved].all()
