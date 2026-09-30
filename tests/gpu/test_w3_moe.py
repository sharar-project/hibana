"""The three-bit routed-expert kernel (kernels/w3_moe.py) against its torch reference on the same 3-bit weights,
eager and graph-captured, from one row to sixteen (experts shared by up to sixteen rows)."""

import pytest
import torch

from sharar.kernels import w3_moe
from sharar.weights.expert_sim3 import requantize

E = 16  # the kernel reads only the experts the ids name


@pytest.fixture(scope="module")
def experts():
    g = torch.Generator(device="cuda").manual_seed(5)

    def sim3(rows, k):
        packed = torch.randint(0, 256, (E, rows, k // 2), generator=g, device="cuda", dtype=torch.uint8)
        scales = (torch.rand(E, rows, k // 16, generator=g, device="cuda") * 0.02 + 0.001).to(torch.float8_e4m3fn)
        return requantize(packed, scales, 32)[:2]

    w13, s13 = sim3(2 * w3_moe.INTER, w3_moe.HIDDEN)
    w2, s2 = sim3(w3_moe.HIDDEN, w3_moe.INTER)
    scale2 = torch.rand(E, generator=g, device="cuda") * 0.5 + 0.25
    return w3_moe.from_sim3(w13, s13, scale2, w2, s2, scale2.flip(0))


@pytest.mark.parametrize("m", [1, 2, 5, 16])
def test_kernel_matches_reference(experts, m):
    torch.manual_seed(m)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    x[:, ::97] *= 30  # outliers exercise the activation group scales
    ids = torch.stack([torch.randperm(E, device="cuda")[:w3_moe.TOPK] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn(m, w3_moe.TOPK, device="cuda"), -1)
    expected = w3_moe.reference(experts, x, ids, weights).float()
    actual = w3_moe.w3_moe(experts, x, ids, weights)
    assert float((actual.float() - expected).norm() / expected.norm()) < 4e-3
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = w3_moe.w3_moe(experts, x, ids, weights)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, actual)  # deterministic: no atomics in the sums
