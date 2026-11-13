"""Grouped IQ2: routing, fused activation, expert parallelism and graph replay."""
from dataclasses import replace
import pytest
import torch
from hibana.kernels import iq2_moe, w3


def packed(quant, e=5, h=256, intermediate=512, base=0):
    size = 74 if quant == "IQ2_XS" else 66
    gen = torch.Generator().manual_seed(345)
    def raw(n, k):
        data = torch.randint(0, 256, (e, n, k // 256, size), dtype=torch.uint8, generator=gen)
        scales = (torch.randn(e, n, k // 256, 1, generator=gen) / 1024).half()
        scales[:, ::11] = 0
        data[..., :2] = scales.view(torch.uint8)
        return data
    return iq2_moe.pack_experts(raw(intermediate, h), raw(h, intermediate), quant, base=base).to("cuda")


@pytest.fixture(scope="module", params=iq2_moe.FORMATS_IQ2)
def experts(request):
    return packed(request.param)


def close(actual, expected):
    assert torch.isfinite(actual).all()
    relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
    assert relative < 0.01, relative.item()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.08, atol=0.04)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 128, 129])
@pytest.mark.parametrize("tile_rows", [16, 32, 64])
def test_grouped_matches_independent_reference(experts, m, tile_rows):
    torch.manual_seed(812 + m)
    x = torch.randn(m, experts.hidden, device="cuda", dtype=torch.bfloat16) * 0.125
    ids = torch.randint(experts.experts, (m, 3), device="cuda", dtype=torch.int32)
    ids[:, 0] = 1  # hot expert, including more than one work tile
    ids[::3, -1] = -1
    weights = torch.randn(m, 3, device="cuda") / 3
    workspace = iq2_moe.make_workspace(experts, m, 3, tile_rows=tile_rows)
    actual = iq2_moe.moe(experts, x, ids, weights, tile_rows=tile_rows, workspace=workspace)
    close(actual, iq2_moe.reference(experts, x, ids, weights))
    assert torch.equal(actual, iq2_moe.moe(experts, x, ids, weights, tile_rows=tile_rows, workspace=workspace))
    q, s = w3.quant_groups(x)
    assert torch.equal(workspace.xq, q.view(torch.uint8))
    assert torch.equal(workspace.xs, s)
    expected_order = torch.cat([torch.where(ids.flatten() == e)[0] for e in range(experts.experts)]).int()
    assert torch.equal(workspace.order[:expected_order.numel()], expected_order)
    count = workspace.count.item()
    covered = []
    for e, start, rows in workspace.work[:count].cpu().tolist():
        assert 1 <= rows <= tile_rows
        indices = workspace.order[start:start + rows]
        assert (ids.flatten()[indices.long()] == e).all()
        covered.extend(indices.cpu().tolist())
    assert sorted(covered) == sorted(expected_order.cpu().tolist())


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_parallel_expert_range_duplicates_and_invalid_ids(experts, dtype):
    ex = replace(experts, up_values=experts.up_values[1:4], up_metadata=experts.up_metadata[1:4],
                 down_values=experts.down_values[1:4], down_metadata=experts.down_metadata[1:4], base=21)
    ids = torch.tensor([[21, 21, 23, 20, 24, -1, torch.iinfo(dtype).min, torch.iinfo(dtype).max]], device="cuda", dtype=dtype)
    weights = torch.tensor([[.1, .2, .3, .4, .5, .6, .7, .8]], device="cuda")
    x = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) * 0.125
    actual = iq2_moe.moe(ex, x, ids, weights)
    close(actual, iq2_moe.reference(ex, x, ids, weights))
    pairs = iq2_moe.moe(ex, x, ids, weights, pairs=True)
    assert torch.count_nonzero(pairs[3:]) == 0
    assert torch.equal(pairs[0], pairs[1])
    combined = torch.zeros_like(x, dtype=torch.float32)
    for k in range(ids.shape[1]):
        combined += weights[:, k:k + 1] * pairs[k:k + 1]
    close(actual, combined.bfloat16())


@pytest.mark.parametrize("tile_rows", [16, 32, 64])
def test_graph_replay_rebuilds_routes_and_skips_stale_pairs(experts, tile_rows):
    m, topk = 17, 9
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16) * 0.125
    ids = torch.randint(0, experts.experts, (m, topk), device="cuda", dtype=torch.int64)
    weights = torch.rand(m, topk, device="cuda")
    workspace = iq2_moe.make_workspace(experts, m, topk, tile_rows=tile_rows)
    out = torch.empty_like(x)
    run = lambda: iq2_moe.moe(experts, x, ids, weights, tile_rows=tile_rows, workspace=workspace, out=out)
    assert run().data_ptr() == out.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ids.remainder_(2); weights.mul_(.5); x.neg_()
    out.fill_(float("nan"))
    graph.replay()
    close(out, iq2_moe.reference(experts, x, ids, weights))
    allocated = torch.cuda.memory_allocated()
    ids.fill_(-1)
    graph.replay(); torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.count_nonzero(out) == 0 and workspace.count.item() == 0


def test_empty_and_unsafe_buffers(experts):
    x = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    weights = torch.ones(1, 1, device="cuda")
    assert iq2_moe.moe(experts, x[:0], ids[:0], weights[:0]).shape == (0, 256)
    with pytest.raises(ValueError, match="overlap"):
        iq2_moe.moe(experts, x, ids, weights, out=x)
    workspace = iq2_moe.make_workspace(experts, 1, 1)
    workspace.xq = x.view(torch.uint8).flatten()[:256].view(1, 256)
    with pytest.raises(ValueError, match="overlap"):
        iq2_moe.moe(experts, x, ids, weights, workspace=workspace)
    with pytest.raises(ValueError, match="routing weights"):
        iq2_moe.moe(experts, x, ids, weights.half())


@pytest.mark.parametrize("quant", iq2_moe.FORMATS_IQ2)
@pytest.mark.parametrize("h,intermediate", [(1024, 1536), (1024, 2048), (1024, 2304), (1024, 2560), (4096, 5376)])
def test_bolt_expert_geometries(quant, h, intermediate):
    ex = packed(quant, e=1, h=h, intermediate=intermediate)
    x = torch.randn(2, h, device="cuda", dtype=torch.bfloat16) * .125
    ids = torch.zeros(2, 1, device="cuda", dtype=torch.int64)
    weights = torch.ones(2, 1, device="cuda")
    close(iq2_moe.moe(ex, x, ids, weights), iq2_moe.reference(ex, x, ids, weights))


@pytest.mark.parametrize("quant", iq2_moe.FORMATS_IQ2)
def test_496_expert_schedule_and_nine_routes(quant):
    one = packed(quant, e=1, h=256, intermediate=256)
    many = replace(one, up_values=one.up_values.expand(496, -1, -1, -1, -1).contiguous(),
                   up_metadata=one.up_metadata.expand(496, -1, -1, -1).contiguous(),
                   down_values=one.down_values.expand(496, -1, -1, -1, -1).contiguous(),
                   down_metadata=one.down_metadata.expand(496, -1, -1, -1).contiguous(), base=1000)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) * .125
    ids = (torch.arange(576, device="cuda").reshape(64, 9) % 496 + 1000).long()
    weights = torch.rand(64, 9, device="cuda")
    actual = iq2_moe.moe(many, x, ids, weights, pairs=True).reshape(64, 9, 256)
    control = iq2_moe.moe(one, x, torch.zeros(64, 1, device="cuda", dtype=torch.int32),
                          torch.ones(64, 1, device="cuda"), pairs=True)
    assert torch.equal(actual, control[:, None, :].expand_as(actual))
