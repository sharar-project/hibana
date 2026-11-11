"""The three-bit routed-expert kernel (kernels/w3_moe.py) against its torch reference on the same 3-bit weights,
eager and graph-captured: one row to prefill-sized calls, experts shared by up to 200 rows (several passes)."""

import pytest
import torch

from hibana.kernels import w3_moe

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


def matches_reference(experts, x, ids, weights):
    """The kernel against the reference, eager and graph-captured."""
    expected = w3_moe.reference(experts, x, ids, weights).float()
    actual = w3_moe.w3_moe(experts, x, ids, weights)
    assert float((actual.float() - expected).norm() / expected.norm()) < 4e-3
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = w3_moe.w3_moe(experts, x, ids, weights)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, actual)  # deterministic: no atomics in the sums


@pytest.mark.parametrize("m", [1, 2, 5, 16, 64, 200])
def test_kernel_matches_reference(experts, m):
    torch.manual_seed(m)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    x[:, ::97] *= 30  # outliers exercise the activation group scales
    ids = torch.stack([torch.randperm(E, device="cuda")[:w3_moe.TOPK] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn(m, w3_moe.TOPK, device="cuda"), -1)
    matches_reference(experts, x, ids, weights)


@pytest.mark.parametrize("k", [1, 5, 9])
@pytest.mark.parametrize("m", [1, 5, 64, 200])  # 200 rows x 5: 1,000 pairs, sorted outside the kernel
def test_fewer_experts_per_row(experts, m, k):
    """ids and weights [M, k], k below 10 (SHARAR_MOE_KEEP alone): the same kernel against the reference, eager and captured."""
    torch.manual_seed(100 * k + m)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:k] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn(m, k, device="cuda"), -1)
    matches_reference(experts, x, ids, weights)


@pytest.mark.parametrize("m", [2, 5, 16, 17, 80])  # 80 rows: 800 slots, sorted outside the kernel
def test_ids_past_the_last_expert_add_nothing(experts, m):
    """Unfilled slots (route_budget without `repeat`) name id 512 at weight 0: no pair, whatever else the row
    holds. Every row shares one expert, as consecutive verify rows often do, with 1 to 9 unfilled slots: the result is
    the reference over the filled slots, and padded rows (every slot unfilled) come out zero."""
    torch.manual_seed(300 + m)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    filled = [1 + r % 9 for r in range(m)]
    ids = torch.full((m, w3_moe.TOPK), w3_moe.EXPERTS, dtype=torch.int32, device="cuda")
    weights = torch.zeros(m, w3_moe.TOPK, device="cuda")
    others = [e for e in range(E) if e != 3]
    for r, n in enumerate(filled):  # expert 3 and n - 1 others, distinct
        ids[r, :n] = torch.tensor([3] + [others[i] for i in torch.randperm(len(others))[:n - 1].tolist()], dtype=torch.int32)
        weights[r, :n] = torch.softmax(torch.randn(n, device="cuda"), -1)
    ids[-1], weights[-1] = w3_moe.EXPERTS, 0.0  # a padded row
    expected = w3_moe.reference(experts, x, torch.where(ids < E, ids, 3), weights).float()
    actual = w3_moe.w3_moe(experts, x, ids, weights)
    assert torch.isfinite(actual.float()).all() and not actual[-1].any()
    assert float((actual.float() - expected).norm() / expected.norm()) < 4e-3


@pytest.mark.parametrize("k", [w3_moe.TOPK, 5])
@pytest.mark.parametrize("m", [1, 5, 80, 200])  # 80 rows x 10 and 200 x 5: sorted outside the kernel
def test_expert_parallel_halves_sum_to_the_whole(experts, m, k):
    """Two ranks' expert ranges (expert parallel): each sums only the pairs routed to its experts, and the two
    partial sums add up to the whole layer's (each partial is rounded to BF16 on its own, as before the all-reduce)."""
    from dataclasses import replace
    torch.manual_seed(100 + m + k)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:k] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn(m, k, device="cuda"), -1)
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


# with 16 experts: about 10, 70, 90, 105, 125 and 940 pairs an expert (passes of one to seven row tiles, several passes)
@pytest.mark.parametrize("k", [w3_moe.TOPK, 5])  # 5: half as many pairs an expert
@pytest.mark.parametrize("m", [17, 112, 144, 168, 200, 1500])
@pytest.mark.parametrize("expert_parallel", [False, True])
def test_pass_per_block_gives_the_same_bits(experts, m, expert_parallel, k, monkeypatch):
    """Wide calls run each pass of up to 112 pairs of an expert in its own block (`gemm_pass_kernel`): the same bits as
    one block looping over 64-pair passes (`gemm_kernel`), for single passes, several passes, and one rank's experts."""
    from dataclasses import replace
    torch.manual_seed(200 + m + k)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:k] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn(m, k, device="cuda"), -1)
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


@pytest.mark.parametrize("k", [w3_moe.TOPK, 5])
@pytest.mark.parametrize("m", [1, 5, 64, 200, 1500])
@pytest.mark.parametrize("expert_parallel", [False, True])
def test_pairs_then_fused_combine_gives_the_same_bits(experts, m, expert_parallel, k):
    """`w3_moe(..., pairs=True)` summed by `moe_combine_pairs` (MoeLayer's path): the bits of the kernel's own combine
    followed by `moe_combine`, with unfilled slots (id 512) and, expert parallel, another rank's pairs left out."""
    from dataclasses import replace

    from hibana.kernels import flashnext_fused as fk
    torch.manual_seed(300 + m + k)
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:k] for _ in range(m)]).to(torch.int32)
    ids[::3, -1] = w3_moe.EXPERTS
    weights = torch.softmax(torch.randn(m, k, device="cuda"), -1)
    shared = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(w3_moe.HIDDEN, device="cuda") * 0.02
    ex = experts
    if expert_parallel:
        half = E // 2
        ex = replace(experts, planes13=experts.planes13[half:], scales13=experts.scales13[half:], row13=experts.row13[half:],
                     planes2=experts.planes2[half:], scales2=experts.scales2[half:], row2=experts.row2[half:], base=half)
    expected = fk.moe_combine(w3_moe.w3_moe(ex, x, ids, weights), shared, x, gate)
    pairs = w3_moe.w3_moe(ex, x, ids, weights, pairs=True)
    actual = fk.moe_combine_pairs(pairs, ids, weights, shared, x, gate, ex.base, ex.experts)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m", [5, 200, 1500])
@pytest.mark.parametrize("expert_parallel", [False, True])
def test_bf16_pairs_are_the_fp32_pairs_rounded(experts, m, expert_parallel):
    """`pairs_bf16` (prefill calls): each pair is the FP32 pair rounded once to BF16, and `moe_combine_pairs` sums
    them like the FP32 pairs."""
    from dataclasses import replace

    from hibana.kernels import flashnext_fused as fk
    torch.manual_seed(400 + m)
    k = w3_moe.TOPK
    x = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:k] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn(m, k, device="cuda"), -1)
    ex = experts
    if expert_parallel:
        half = E // 2
        ex = replace(experts, planes13=experts.planes13[half:], scales13=experts.scales13[half:], row13=experts.row13[half:],
                     planes2=experts.planes2[half:], scales2=experts.scales2[half:], row2=experts.row2[half:], base=half)
    fp32 = w3_moe.w3_moe(ex, x, ids, weights, pairs=True)
    bf16 = w3_moe.w3_moe(ex, x, ids, weights, pairs=True, pairs_bf16=True)
    assert bf16.dtype == torch.bfloat16
    local = ((ids - ex.base >= 0) & (ids - ex.base < ex.experts)).reshape(-1)
    assert torch.equal(bf16[local].view(torch.int16), fp32[local].to(torch.bfloat16).view(torch.int16))
    shared = torch.randn(m, w3_moe.HIDDEN, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(w3_moe.HIDDEN, device="cuda") * 0.02
    expected = fk.moe_combine_pairs(fp32.to(torch.bfloat16).float(), ids, weights, shared, x, gate, ex.base, ex.experts)
    actual = fk.moe_combine_pairs(bf16, ids, weights, shared, x, gate, ex.base, ex.experts)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
