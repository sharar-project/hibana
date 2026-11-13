# SPDX-License-Identifier: Apache-2.0
"""The batched expert GPTQ (weights/flashnext_w3_experts.py) against the per-matrix pass it batches
(weights/w3_calibration.quantize), and its output error against the weight-error fit."""
import torch



def _case(c=3, n=256, k=384, rows=2000, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((c, n, k), device="cuda", generator=g) * 0.02
    mix = torch.randn((c, k, k), device="cuda", generator=g) / k ** 0.5 + torch.eye(k, device="cuda")
    x = torch.randn((c, rows, k), device="cuda", generator=g) @ mix * torch.linspace(0.2, 3.0, k, device="cuda")
    return w, x, torch.bmm(x.transpose(1, 2), x)


def test_batched_matches_per_matrix():
    from hibana.kernels import w3
    from hibana.weights.flashnext_w3_experts import dequant_codes, gptq_batched
    from hibana.weights.w3_calibration import quantize
    w, _, h = _case()
    levels = w3._codebook("lloyd")
    grid = (.6, .7, .8, .9, 1.0)
    codes, scales, rows = gptq_batched(w, h.clone(), levels, grid)
    for e in range(w.shape[0]):
        ref, _ = quantize(w[e], h[e], codebook="lloyd", rows_per_chunk=w.shape[1])
        ours = dequant_codes(codes[e], scales[e], rows[e], levels)
        assert (w3.dequantize(ref) - ours).abs().max().item() < 1e-6 + 1e-3 * w[e].abs().max().item()
        assert (w3.codes(ref) == codes[e]).float().mean().item() > 0.999


def test_output_error_below_weight_fit():
    from hibana.kernels import w3, w3_moe
    from hibana.weights.flashnext_w3_experts import dequant_codes, gptq_batched
    w, x, h = _case(seed=1)
    levels = w3._codebook("lloyd")
    codes, scales, rows = gptq_batched(w, h.clone(), levels, w3_moe.RATIOS)
    deq_g = dequant_codes(codes, scales, rows, levels)
    c, n, k = w.shape
    cc, cs, cr = w3_moe.quantize(w.reshape(c * n, k), levels)
    deq_c = dequant_codes(cc, cs, cr, levels).reshape(c, n, k)
    out = lambda d: torch.bmm(x, d.transpose(1, 2)).square().sum().item()
    assert out(deq_g - w) < 0.85 * out(deq_c - w)
