"""Packed weight-only projections: BF16 operands, routed pairs and CUDA graphs."""

import pytest
import torch

from sharar.kernels import ggml_bf16 as kernels
from sharar.model.bolt.linear import PackedLinear


def matrix(quant, n, k, device="cuda"):
    if quant == "NVFP4":
        return PackedLinear(torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8), quant,
                            (torch.rand(n, k // 16, device=device) * .05).to(torch.float8_e4m3fn),
                            torch.tensor(.7, device=device))
    block, size = (256, 66) if quant == "IQ2_XXS" else (32, 34)
    packed = torch.randint(0, 256, (n, k // block, size), device=device, dtype=torch.uint8)
    packed[..., :2] = (torch.rand(n, k // block, 1, device=device).half() / 128).view(torch.uint8)
    return PackedLinear(packed, quant)


@pytest.mark.parametrize("quant", ["IQ2_XXS", "Q8_0", "NVFP4"])
@pytest.mark.parametrize("m", [1, 17, 65])
@pytest.mark.parametrize("relu2", [False, True])
def test_dense_matches_bf16_reference(quant, m, relu2):
    torch.manual_seed(12)
    weight = matrix(quant, 80, 512)
    x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
    if quant == "NVFP4":
        from sharar.weights.modelopt import unpack_fp4
        expected = (torch.mm(x, unpack_fp4(weight.weight, weight.scales).bfloat16().T,
                             out_dtype=torch.float32) * weight.gain).bfloat16()
    else:
        expected = x @ weight.dequantize().bfloat16().T
    if relu2:
        expected = expected.relu().square()
    actual = weight(x, relu2=relu2)
    torch.testing.assert_close(actual, expected, rtol=.016, atol=.004)


def test_routed_pairs_padding_and_combine():
    torch.manual_seed(13)
    up = [matrix("IQ2_XXS", 512, 256) for _ in range(3)]
    down = [matrix("IQ2_XXS", 256, 512) for _ in range(3)]
    w1, w2 = torch.stack([w.weight for w in up]), torch.stack([w.weight for w in down])
    ids = torch.tensor([[1, 2], [0, 1], [3, 3]], dtype=torch.int32, device="cuda")
    weights = torch.tensor([[.2, .8], [.4, .6], [0., 0.]], device="cuda")
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    hidden = kernels.matmul(x, w1, "IQ2_XXS", ids=ids, repeat_input=True, relu2=True)
    pairs = kernels.matmul(hidden, w2, "IQ2_XXS", ids=ids)
    actual = kernels.combine(pairs, weights)
    expected = torch.zeros_like(actual, dtype=torch.float32)
    for t in range(2):
        for choice in range(2):
            e = ids[t, choice].item()
            h = (x[t:t + 1] @ up[e].dequantize().bfloat16().T).relu().square()
            y = h @ down[e].dequantize().bfloat16().T
            expected[t] += y[0].float() * weights[t, choice] * 5
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=.03, atol=.008)
    assert torch.count_nonzero(actual[-1]) == 0
    # Routing and weight changes remain device inputs under graph replay.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        h = kernels.matmul(x, w1, "IQ2_XXS", ids=ids, repeat_input=True, relu2=True)
        p = kernels.matmul(h, w2, "IQ2_XXS", ids=ids)
        captured = kernels.combine(p, weights)
    ids[:2].zero_()
    graph.replay()
    h = kernels.matmul(x, w1, "IQ2_XXS", ids=ids, repeat_input=True, relu2=True)
    expected = kernels.combine(kernels.matmul(h, w2, "IQ2_XXS", ids=ids), weights)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_q8_embedding_reads_only_requested_rows():
    weight = matrix("Q8_0", 80, 512)
    ids = torch.tensor([4, 0, 79, 4], device="cuda")
    expected = weight.dequantize()[ids].bfloat16()
    assert torch.equal(kernels.embedding(ids, weight.weight), expected)
