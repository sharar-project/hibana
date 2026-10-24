"""Nemotron's 3-bit latent experts (kernels/w3_moe.py, the non-gated ReLU^2 geometry: x [M, 1024], fc1 = up [2688,
1024], fc2 [1024, 2688], top 22) against the torch reference on the same 3-bit weights, eager and graph-captured:
decode, verify, the host-sorted calls past 640 pairs and the wide prefill passes."""

import pytest
import torch

from sharar.kernels import w3_moe

E, L, I, K = 48, 1024, 2688, 22  # the kernel reads only the experts the ids name


@pytest.fixture(scope="module")
def experts():
    g = torch.Generator(device="cuda").manual_seed(7)

    def nvfp4(rows, k):
        packed = torch.randint(0, 256, (E, rows, k // 2), generator=g, device="cuda", dtype=torch.uint8)
        scales = (torch.rand(E, rows, k // 16, generator=g, device="cuda") * 0.02 + 0.001).to(torch.float8_e4m3fn)
        return packed, scales

    w1, s1 = nvfp4(I, L)
    w2, s2 = nvfp4(L, I)
    scale2 = torch.rand(E, generator=g, device="cuda") * 0.5 + 0.25
    ex, report = w3_moe.calibrate_relu2(w1, s1, scale2, w2, s2, scale2.flip(0))
    assert not w3_moe.gated(ex)
    assert 0.05 < report["w1"] < 0.3 and 0.05 < report["w2"] < 0.3
    return ex


@torch.no_grad()
def reference(experts, x, ids, weights):
    """w3_moe.reference's math (bf16 up, bf16 relu^2, E4M3 groups, FP32 fc2), one expert at a time over its rows."""
    from sharar.kernels import w3
    xd = w3_moe.quant_e4m3(x, "x")
    out = torch.zeros((x.shape[0], L), dtype=torch.float32, device=x.device)
    for e in ids.unique().tolist():
        if e >= experts.experts:
            continue
        rows, slot = (ids == e).nonzero(as_tuple=True)
        w1 = w3.dequantize(experts.matrix(e, True))
        w2 = w3.dequantize(experts.matrix(e, False))
        h = torch.relu((xd[rows] @ w1.T).to(torch.bfloat16).float()).square().to(torch.bfloat16)
        out.index_add_(0, rows, (w3_moe.quant_e4m3(h, "h") @ w2.T) * weights[rows, slot, None])
    return out


def check(experts, x, ids, weights, pairs=False):
    expected = reference(experts, x, ids, weights)
    actual = w3_moe.w3_moe(experts, x, ids, weights)
    assert actual.shape == (x.shape[0], L)
    assert float((actual.float() - expected).norm() / expected.norm()) < 4e-3
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = w3_moe.w3_moe(experts, x, ids, weights)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, actual)  # deterministic: no atomics in the sums
    if pairs:  # the per-pair outputs, summed in selection order, give the sum
        p = w3_moe.w3_moe(experts, x, ids, weights, pairs=True).reshape(x.shape[0], ids.shape[1], L)
        summed = torch.zeros_like(expected)
        for k in range(ids.shape[1]):
            summed += weights[:, k, None] * torch.where((ids[:, k] < experts.experts)[:, None], p[:, k], 0.0)
        assert float((summed - actual.float()).norm() / expected.norm()) < 4e-3


def routing(m, k=K):
    return torch.stack([torch.randperm(E, device="cuda")[:k] for _ in range(m)]).to(torch.int32)


@pytest.mark.parametrize("m", [1, 2, 4, 5, 16, 17, 29, 64, 200, 1100])  # 29 x 22 = 638 pairs sorted in the kernel, 30 rows past
def test_kernel_matches_reference(experts, m):
    torch.manual_seed(m)
    x = torch.randn(m, L, device="cuda", dtype=torch.bfloat16)
    x[:, ::97] *= 30  # outliers exercise the activation group scales
    ids = routing(m)
    weights = torch.rand(m, K, device="cuda") * 5 / K  # renormalized sigmoid weights times routed_scaling_factor
    if m < 1024:
        check(experts, x, ids, weights, pairs=m <= 64)
    else:  # wide passes (WIDE_ROWS): the reference on a sample of rows is enough
        actual = w3_moe.w3_moe(experts, x, ids, weights)
        rows = torch.arange(0, m, 37, device="cuda")
        expected = reference(experts, x[rows], ids[rows], weights[rows])
        assert float((actual[rows].float() - expected).norm() / expected.norm()) < 4e-3


@pytest.mark.parametrize("m", [1, 4, 17, 64])
def test_unfilled_slots_add_nothing(experts, m):
    """A padded row (every slot id 512, weight 0) comes out zero; rows with a few unfilled slots are the reference over
    the filled ones."""
    torch.manual_seed(500 + m)
    x = torch.randn(m, L, device="cuda", dtype=torch.bfloat16)
    ids, weights = routing(m), torch.rand(m, K, device="cuda") / K
    for r in range(m):
        n = 4 + r % 18
        ids[r, n:], weights[r, n:] = w3_moe.EXPERTS, 0.0
    if m > 1:
        ids[-1], weights[-1] = w3_moe.EXPERTS, 0.0
    actual = w3_moe.w3_moe(experts, x, ids, weights)
    expected = reference(experts, x, ids, weights)
    assert torch.isfinite(actual.float()).all()
    if m > 1:
        assert not actual[-1].any()
    assert float((actual.float() - expected).norm() / expected.norm()) < 4e-3


@pytest.mark.parametrize("k", [4, 12])
def test_fewer_experts_per_row(experts, k):
    torch.manual_seed(900 + k)
    x = torch.randn(8, L, device="cuda", dtype=torch.bfloat16)
    check(experts, x, routing(8, k), torch.rand(8, k, device="cuda"))


@pytest.fixture(scope="module")
def block1():
    """Block 1 of the pinned checkpoint twice: NVFP4 experts and (nearest-rounded, Lloyd/zero8) 3-bit ones."""
    import gc
    from sharar.model.nemotron.moe import NemotronMoe, load_experts
    from sharar.model.nemotron.reference import find_snapshot
    from sharar.target import NEMOTRON
    from sharar.weights.safetensors import SnapshotWeights
    try:
        snap = find_snapshot()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    w = SnapshotWeights(snap)
    prefix = "language_model.backbone.layers.1.mixer"
    nv = load_experts(w, prefix, NEMOTRON.num_experts, torch.device("cuda"))
    ex, _ = w3_moe.calibrate_relu2(nv["w1"], nv["s1"], nv["g1"], nv["w2"], nv["s2"], nv["g2"], codebook2="zero8")
    del nv
    blocks = (NemotronMoe(w, prefix, torch.device("cuda"), NEMOTRON), NemotronMoe(w, prefix, torch.device("cuda"), NEMOTRON, w3=ex))
    norm = w.get("language_model.backbone.layers.1.norm.weight").float()
    yield blocks, norm
    del blocks
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("m", [1, 4, 16, 40])
def test_block_with_3bit_experts(block1, m):
    """The block with 3-bit experts takes route + w3_moe at every size (the small-call kernels read NVFP4 experts):
    decode calls with padded rows (slot 0: no expert, a zero routed sum) match the block's own prefill path on the real
    rows, and stay near the NVFP4 block (nearest rounding: cosine > 0.98)."""
    from types import SimpleNamespace
    from sharar.model.nemotron import config as c
    (nvfp4, b3), norm = block1
    g = torch.Generator().manual_seed(m)
    x = (torch.randn(m, c.HIDDEN_SIZE, generator=g) * norm).bfloat16().cuda()
    real = max(1, m - m // 4)
    slots = torch.tensor([1 + i for i in range(real)] + [0] * (m - real), dtype=torch.int32, device="cuda")
    decode = b3(x, SimpleNamespace(mode="decode", slots=slots))
    prefill = b3(x[:real], None)
    assert float((decode[:real].float() - prefill.float()).norm() / prefill.float().norm()) < 1e-2
    cos = torch.nn.functional.cosine_similarity(prefill.float().flatten(), nvfp4(x[:real], None).float().flatten(), dim=0)
    assert float(cos) > 0.98


@pytest.mark.parametrize("m,mode", [(1, "decode"), (4, "decode"), (8, "verify"), (16, "verify")])
def test_small_calls_with_3bit_experts(block1, m, mode):
    """Calls of at most 16 rows keep the small-call kernels around w3_moe (router|fc1 GEMV and routing, shared expert,
    fc2_latent): the router's BF16 latent and ids are route_topk's on its FP32 output, bitwise (so the routed sum is
    w3_moe's on them, padded rows naming no expert), and the block matches route + w3_moe + the NVFP4 shared expert +
    the fc2 addmm to the rounding the NVFP4 block's two paths differ by; a CUDA graph replays it bitwise."""
    import copy
    from types import SimpleNamespace
    from sharar.kernels import nemotron_moe
    from sharar.model.nemotron import config as c
    from sharar.model.nemotron.moe import route_topk
    (_, b3), norm = block1
    g = torch.Generator().manual_seed(300 + m)
    x = (torch.randn(m, c.HIDDEN_SIZE, generator=g) * norm).bfloat16().cuda()
    if mode == "decode":
        s = torch.tensor([1 + i for i in range(m - m // 4)] + [0] * (m // 4), dtype=torch.int32, device="cuda")
        ctx, steps = SimpleNamespace(mode="decode", slots=s), 1
    else:
        s = torch.tensor([3, 0] if m == 8 else [3, 1, 0, 0], dtype=torch.int32, device="cuda")  # 4 rows a request
        ctx, steps = SimpleNamespace(mode="verify", seq_slots=s, verify_steps=4), 4
    with torch.inference_mode():
        fused, ids, wts, _, lat = nemotron_moe.router(x, b3.small_weights(), b3.bias, b3.topk, s, steps, pad_id=E_ALL, latent=True)
        ref_ids, ref_w, ref_lat = route_topk(fused, b3.bias, b3.topk, b3.latent, s, steps, repeat=False)
        assert torch.equal(ids, ref_ids) and torch.equal(wts, ref_w) and torch.equal(lat, ref_lat)
        big = copy.copy(b3)
        big.SMALL_ROWS = 0
        small, wide = b3(x, ctx), big(x, ctx)
        # the shared expert and fc2 round differently on the two paths: the NVFP4 block's small and wide calls differ
        # by 0.9-1.0% too (block 1, M 1-16), and its small path is the closer to the FP32 reference
        assert float((small.float() - wide.float()).norm() / wide.float().norm()) < 1.5e-2
        static = x.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = b3(static, ctx)
        graph.replay()
        assert torch.equal(out, small)


E_ALL = 512  # the id a padded row names with 3-bit experts (no expert)
