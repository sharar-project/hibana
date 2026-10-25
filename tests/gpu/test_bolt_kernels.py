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
        from sharar.weights.modelopt import unpack_fp4, unswizzle_block_scales
        scales = unswizzle_block_scales(weight.scales.reshape(-1), 80, 512 // 16)
        expected = (torch.mm(x, unpack_fp4(weight.weight, scales).bfloat16().T,
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


@pytest.mark.parametrize("n,k", [(18560, 4096), (4096, 8192)])  # Bolt's Mamba in_proj and out_proj
@pytest.mark.parametrize("m", [1, 8, 32, 40, 129])
def test_nvfp4_bolt_shapes_and_graph_replay(n, k, m):
    """b12x's W4A16 GEMM at verification row counts, and a captured call reading new inputs on replay."""
    torch.manual_seed(14)
    weight = matrix("NVFP4", n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    expected = (x.float() @ weight.dequantize().T).bfloat16()
    torch.testing.assert_close(weight(x), expected, rtol=.016, atol=.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = weight(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    assert torch.equal(captured, weight(x))


@pytest.mark.parametrize("quant", ["Q8_0", "IQ2_XXS"])
def test_block_quant_plan_rows_chunking_and_release(quant):
    """A prepared projection drops its checkpoint copy, runs its exact rows and chunks calls past its capacity."""
    from sharar.model.bolt.linear import PlanRows
    torch.manual_seed(15)
    weight = matrix(quant, 256, 512)
    reference = weight.dequantize().bfloat16()
    weight.prepare(PlanRows(capacity=16, exact=(8,)))
    assert weight.weight is None and weight.bytes > 0
    for m in (8, 16, 40):
        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(weight(x), x @ reference.T, rtol=.016, atol=.004)
    with pytest.raises(ValueError):
        weight.dequantize()


@pytest.mark.parametrize("quant", ["NVFP4", "Q8_0"])
def test_prepared_plans_run_on_another_thread(quant):
    """The engine infers on its own thread: a plan compiled at load must serve a row count no warm-up ran there."""
    import threading
    from sharar.model.bolt.linear import PlanRows
    torch.manual_seed(16)
    weight = matrix(quant, 256, 512)
    reference = weight.dequantize().float()
    weight.prepare(PlanRows(capacity=1024, exact=(8,)))
    x = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16)
    result = {}
    thread = threading.Thread(target=lambda: result.update(y=weight(x)))
    thread.start()
    thread.join()
    expected = x.float() @ reference.T
    assert ((result["y"].float() - expected).norm() / expected.norm()).item() < 5e-3


def test_b12x_iq2_moe_matches_reference_replays_and_runs_on_another_thread():
    """b12x's fused IQ2_XXS MoE: weighted ReLU^2 experts against the BF16 operator, changed routes under graph replay,
    and a row count no warm-up ran, called from another thread (the engine's)."""
    import threading
    from sharar.kernels import b12x_iq2_moe
    from sharar.model.bolt.linear import PlanRows
    torch.manual_seed(17)
    experts, inter, hidden, topk = 16, 512, 256, 4
    up = torch.stack([matrix("IQ2_XXS", inter, hidden).weight for _ in range(experts)])
    down = torch.stack([matrix("IQ2_XXS", hidden, inter).weight for _ in range(experts)])
    reference = (up.clone(), down.clone())
    block = b12x_iq2_moe.Experts(up, down, topk, PlanRows(capacity=64, exact=(1, 8)))
    b12x_iq2_moe.prepare([block])

    def expected(x, ids, weights):
        h = kernels.matmul(x, reference[0], "IQ2_XXS", ids=ids, repeat_input=True, relu2=True)
        return kernels.combine(kernels.matmul(h, reference[1], "IQ2_XXS", ids=ids), weights, scale=1.).float()

    def close(y, want):
        return ((y.float() - want).norm() / want.norm()).item() < .01

    x = torch.randn(8, hidden, device="cuda", dtype=torch.bfloat16) * .5
    ids = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(8)]).int()
    weights = torch.softmax(torch.randn(8, topk, device="cuda"), -1) * 5
    assert close(block(x, ids, weights), expected(x, ids, weights))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = block(x, ids, weights)
    ids.copy_((ids + 3) % experts)
    graph.replay()
    assert close(captured, expected(x, ids, weights))
    result = {}
    rows = torch.randn(13, hidden, device="cuda", dtype=torch.bfloat16) * .5
    routes = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(13)]).int()
    gains = torch.full((13, topk), .5, device="cuda")
    thread = threading.Thread(target=lambda: result.update(y=block(rows, routes, gains)))
    thread.start()
    thread.join()
    assert close(result["y"], expected(rows, routes, gains))


@pytest.mark.parametrize("m", [8, 28])
def test_tuned_launch_configs_take_their_own_plan(m):
    """A row count `model/bolt/tuning.py` lists runs its measured configuration and keeps the heuristic's result."""
    from sharar.model.bolt import tuning
    from sharar.model.bolt.linear import PlanRows
    assert tuning.config("nvfp4", 4096, 8192, m) is not None
    torch.manual_seed(18)
    weight = matrix("NVFP4", 4096, 8192)
    x = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
    expected = (x.float() @ weight.dequantize().T).bfloat16()
    weight.prepare(PlanRows(exact=(m, 3)))
    assert set(weight.tuned) == {m}
    torch.testing.assert_close(weight(x), expected, rtol=.016, atol=.03)


@pytest.mark.parametrize("m,keep,cap", [(8, 6, 20), (32, 4, 48), (8, 6, 0), (8, 9, 512)])
def test_expert_sharing_at_bolt_geometry(m, keep, cap):
    """Super's budgeted routing on Bolt's top 9 of 432 (16 padded lanes): a call reads at most the cap's experts (or
    the guaranteed ones), every real row keeps its top `keep` first and only set members after, in rank order, at
    renormalized sigmoid weights, dropped and padded slots at weight 0 on an expert already read; with every top 9 in the set, route_topk's ids and weights."""
    from sharar.model.nemotron.moe import route_budget, route_topk
    torch.manual_seed(19)
    experts, latent, k = 432, 1024, 9
    fused = torch.randn(m, experts + latent, device="cuda") * 2
    bias = torch.randn(experts, device="cuda") * .1
    slots = torch.tensor([1] * (m // 8 - 1) + [0], dtype=torch.int32, device="cuda")  # the last request padded
    ids, w, lat = route_budget(fused, bias, k, latent, keep, cap, slots, 8, repeat=True)
    real = m - 8
    s = fused[:, :experts].sigmoid()
    order = torch.topk(s + bias, k, dim=-1).indices
    used = ids[:real][w[:real] > 0].unique()
    guaranteed = order[:real, :keep].unique()
    assert len(used) <= max(cap, len(guaranteed)) and torch.isin(guaranteed, used).all()
    for r in range(real):
        live = w[r] > 0
        assert torch.equal(ids[r, :keep].long(), order[r, :keep])
        picked = ids[r][live].long()
        pos = [int((order[r] == e).nonzero()) for e in picked.tolist()]
        assert pos == sorted(pos)
        torch.testing.assert_close(w[r][live], s[r, picked] / s[r, picked].sum(), rtol=1e-6, atol=1e-7)
        assert (ids[r][~live] == ids[r, 0]).all()  # the row's top expert, at weight 0
    assert (w[real:] == 0).all() and (ids[real:] == order[0, 0]).all()
    assert torch.equal(lat, fused[:, experts:].bfloat16())
    if cap >= real * k:
        plain = route_topk(fused, bias, k, latent, slots, 8)
        assert torch.equal(ids, plain[0]) and torch.equal(w, plain[1])
