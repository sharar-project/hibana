"""The three-bit routed-expert kernel (kernels/w3_moe.py) against its torch reference on the same 3-bit weights,
eager and graph-captured: one row to prefill-sized calls, experts shared by up to 200 rows (several passes)."""

import pytest
import torch

from sharar.kernels import w3_moe

E = 16  # the kernel reads only the experts the ids name


@pytest.fixture(scope="module")
def experts():
    g = torch.Generator(device="cuda").manual_seed(5)

    def nvfp4(rows, k):
        packed = torch.randint(0, 256, (E, rows, k // 2), generator=g, device="cuda", dtype=torch.uint8)
        scales = (torch.rand(E, rows, k // 16, generator=g, device="cuda") * 0.02 + 0.001).to(torch.float8_e4m3fn)
        return packed, scales

    w13, s13 = nvfp4(2 * w3_moe.INTER, w3_moe.HIDDEN)
    w2, s2 = nvfp4(w3_moe.HIDDEN, w3_moe.INTER)
    scale2 = torch.rand(E, generator=g, device="cuda") * 0.5 + 0.25
    ex, report = w3_moe.calibrate(w13, s13, scale2, w2, s2, scale2.flip(0))
    assert 0.05 < report["w13"] < 0.3 and 0.05 < report["w2"] < 0.3
    return ex


@pytest.mark.parametrize("m", [1, 2, 5, 16, 64, 200])
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


@pytest.mark.parametrize("m", [1, 5, 80])  # 80 rows: 800 pairs, sorted outside the kernel
def test_expert_parallel_halves_sum_to_the_whole(experts, m):
    """Two ranks' expert ranges (expert parallel): each sums only the pairs routed to its experts, and the two
    partial sums add up to the whole layer's (each partial is rounded to BF16 on its own, as before the all-reduce)."""
    from dataclasses import replace
    torch.manual_seed(100 + m)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:w3_moe.TOPK] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn(m, w3_moe.TOPK, device="cuda"), -1)
    half = E // 2

    def part(lo):
        return replace(experts, planes13=experts.planes13[lo:lo + half], scales13=experts.scales13[lo:lo + half],
                       row13=experts.row13[lo:lo + half], planes2=experts.planes2[lo:lo + half],
                       scales2=experts.scales2[lo:lo + half], row2=experts.row2[lo:lo + half], base=lo)

    whole = w3_moe.w3_moe(experts, x, ids, weights).float()
    parts = w3_moe.w3_moe(part(0), x, ids, weights).float() + w3_moe.w3_moe(part(half), x, ids, weights).float()
    assert float((parts - whole).norm() / whole.norm()) < 4e-3
    only = torch.full_like(ids, half)  # every pair on the second rank: the first rank's partial is zero
    assert not w3_moe.w3_moe(part(0), x, only, weights).any()


@pytest.mark.parametrize("m", [17, 200, 1500])  # with 16 experts: 10, 125 and about 940 pairs an expert
@pytest.mark.parametrize("expert_parallel", [False, True])
def test_pass_per_block_gives_the_same_bits(experts, m, expert_parallel, monkeypatch):
    """Wide calls run each 64-pair pass of an expert in its own block (`gemm_pass_kernel`): the same bits as one block
    looping over the passes (`gemm_kernel`), for single passes, several passes, and one rank's expert range."""
    from dataclasses import replace
    torch.manual_seed(200 + m)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:w3_moe.TOPK] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn(m, w3_moe.TOPK, device="cuda"), -1)
    ex = experts
    if expert_parallel:
        half = E // 2
        ex = replace(experts, planes13=experts.planes13[half:], scales13=experts.scales13[half:], row13=experts.row13[half:],
                     planes2=experts.planes2[half:], scales2=experts.scales2[half:], row2=experts.row2[half:], base=half)
    monkeypatch.setattr(w3_moe, "WIDE_ROWS", 10**9)
    looped = w3_moe.w3_moe(ex, x, ids, weights)
    monkeypatch.setattr(w3_moe, "WIDE_ROWS", 1)
    passes = w3_moe.w3_moe(ex, x, ids, weights)
    assert torch.equal(passes.view(torch.int16), looped.view(torch.int16))
