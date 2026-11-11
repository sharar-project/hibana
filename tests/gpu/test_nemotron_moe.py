"""Nemotron-3.5-Super's latent MoE block (`hibana/model/nemotron/moe.py`) on block 1 of the pinned checkpoint.

- `NemotronMoe.reference` (FP32, dequantized experts) against Transformers' `NemotronHMoE` carrying the same
  dequantized weights in BF16: the semantics (FP32 router, bias for the choice only, renormalized sigmoid weights times
  5, latent experts, shared expert on x).
- The served block against the reference with its activations rounded to NVFP4 as the kernels round them. Calls of
  at most 16 rows take Sharar's kernels (`kernels/nemotron_moe.py`): cosine >= 0.9999 (0.999997 measured), their router
  kernel's routing bitwise route_topk's, and the budgeted path. Larger calls take FlashInfer's b12x MoE (fused
  router|fc1 GEMM, routing kernel, b12x ReLU², NVFP4 shared expert, fc2 addmm), whose routed sum is checked against
  FlashInfer's `cutlass_fused_moe` on the same pairs (`cutlass_routed`, the test's reference: it is deterministic and
  rounds closer to the reference than b12x); against the FP32 reference about 0.9985, which is NVFP4 activations alone.
- The routing kernel against torch's top-k (the same experts every row), its padded rows, budgeted selection's
  invariants (floors, free riders, a binding cap), a replayed CUDA graph, and the MTP head's block.

Needs the pinned snapshot (skipped otherwise); holds one block's experts (1.5 GiB) and, for the Transformers check,
a BF16 copy of them (5.6 GiB)."""

import copy
import gc
from types import SimpleNamespace

import pytest
import torch

from hibana.model.nemotron import config as c

PREFIX = "language_model.backbone.layers.1.mixer"
B12X_COS = 0.998  # b12x's routed sum against cutlass_fused_moe's on the same pairs (real inputs, blocks 1/8/14: 0.9987-0.9999)
B12X_BLOCK_COS = 0.9995  # a block through b12x against the reference with NVFP4 activations (cutlass: 0.9999)
MTP_PREFIX = "language_model.mtp.layers.1.mixer"


def cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


@pytest.fixture(scope="module")
def weights():
    from hibana.target import NEMOTRON
    from hibana.weights.safetensors import SnapshotWeights
    try:
        snap = NEMOTRON.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return snap, SnapshotWeights(snap)


@pytest.fixture(scope="module")
def moe(weights):
    from hibana.model.nemotron.moe import NemotronMoe
    from hibana.target import NEMOTRON
    m = NemotronMoe(weights[1], PREFIX, torch.device("cuda"), NEMOTRON)
    yield m
    del m
    gc.collect()
    torch.cuda.empty_cache()


def rows(w, m: int, seed: int) -> torch.Tensor:
    """Post-norm-like rows: Gaussian times block 1's norm weight."""
    g = torch.Generator().manual_seed(seed)
    norm = w.get("language_model.backbone.layers.1.norm.weight").float()
    return (torch.randn(m, c.HIDDEN_SIZE, generator=g) * norm).bfloat16().cuda()


def cutlass_routed(moe, lat, ids, w):
    """The routed experts' weighted sum through FlashInfer's `cutlass_fused_moe` (ReLU², the checkpoint's tensors and
    `moe.quant_scales`): the reference the b12x path is checked against."""
    from flashinfer.fused_moe import ActivationType, cutlass_fused_moe
    m = lat.shape[0]
    out = cutlass_fused_moe(
        input=lat, token_selected_experts=ids, token_final_scales=w,
        fc1_expert_weights=moe.w1.view(torch.long), fc2_expert_weights=moe.w2.view(torch.long),
        output_dtype=torch.bfloat16, quant_scales=moe.quant_scales, activation_type=ActivationType.Relu2,
        tune_max_num_tokens=max(1, 1 << (m - 1).bit_length()))
    return out[0] if isinstance(out, (list, tuple)) else out


def ctx_for(mode: str, slots: list[int], steps: int = 1):
    s = torch.tensor(slots, dtype=torch.int32, device="cuda")
    return SimpleNamespace(mode="decode", slots=s) if mode == "decode" else SimpleNamespace(mode="verify", seq_slots=s, verify_steps=steps)


def test_reference_matches_transformers(weights, moe):
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMoE
    from hibana.kernels.w3_moe import nvfp4_dequant, unswizzle_batched
    import json
    snap, w = weights
    cfg = {k: v for k, v in json.loads((snap / "config.json").read_text())["llm_config"].items() if k != "time_step_limit"}
    with torch.device("meta"):
        hf = NemotronHMoE(NemotronHConfig(**cfg))
    hf = hf.to_empty(device="cuda").bfloat16()
    with torch.no_grad():
        hf.gate.weight.copy_(moe.gate_fc1[:moe.experts])
        hf.gate.e_score_correction_bias = moe.bias.clone()
        hf.fc1_latent_proj.weight.copy_(moe.gate_fc1[moe.experts:])
        hf.fc2_latent_proj.weight.copy_(moe.fc2)
        hf.shared_experts.up_proj.weight.copy_(moe.shared_up.dequantize())
        hf.shared_experts.down_proj.weight.copy_(moe.shared_down.dequantize())
        s1 = unswizzle_batched(moe.s1, moe.inter, moe.latent // 16)
        s2 = unswizzle_batched(moe.s2, moe.latent, moe.inter // 16)
        for lo in range(0, moe.experts, 64):  # FP32 dequantization a slice at a time
            hf.experts.up_proj[lo:lo + 64] = nvfp4_dequant(moe.w1[lo:lo + 64], s1[lo:lo + 64], moe.g1[lo:lo + 64])
            hf.experts.down_proj[lo:lo + 64] = nvfp4_dequant(moe.w2[lo:lo + 64], s2[lo:lo + 64], moe.g2[lo:lo + 64])
        x = rows(w, 48, 0)
        got = hf(x[None])[0]
    ref = moe.reference(x)
    assert cos(ref, got) > 0.99999
    assert float((ref - got.float()).norm() / ref.norm()) < 0.006  # BF16 weights and activations on the HF side
    del hf
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("m", [1, 4, 9, 16, 37, 300])
def test_block_matches_reference(weights, moe, m):
    x = rows(weights[1], m, m)
    with torch.inference_mode():
        out = moe(x, None)
        assert out.dtype == torch.bfloat16 and out.shape == x.shape
        b12x = m > moe.SMALL_ROWS  # b12x rounds less exactly than cutlass and Sharar's kernels
        assert cos(out, moe.reference(x, quantize_activations=True)) > (B12X_BLOCK_COS if b12x else 0.9999)
        assert cos(out, moe.reference(x)) > 0.998


@pytest.mark.parametrize("m", [1, 7, 64, 2048])
def test_routing_is_torch_topk(weights, moe, m):
    from hibana.model.nemotron.moe import route_topk
    x = rows(weights[1], m, 100 + m)
    fused = torch.mm(x, moe.gate_fc1.T, out_dtype=torch.float32)
    ids, w, lat = route_topk(fused, moe.bias, moe.topk, moe.latent)
    s = fused[:, :moe.experts].sigmoid()
    choice = s + moe.bias
    ref = torch.topk(choice, moe.topk, dim=-1)
    # the same choice scores in the same descending order: the same experts up to exact FP32 ties of s + bias (its
    # spacing near 5 is 5e-7, so ties occur, a few rows in 2048), which the kernel breaks to the lower expert
    assert torch.equal(choice.gather(1, ids.long()), ref.values)
    assert (ids.long() == ref.indices).all(-1).float().mean() > 0.99
    want = s.gather(1, ids.long())
    torch.testing.assert_close(w, want / (want.sum(-1, keepdim=True) + 1e-20), rtol=1e-6, atol=1e-7)
    assert torch.equal(lat, fused[:, moe.experts:].bfloat16())


@pytest.mark.parametrize("m,k,slots,steps", [(16, 22, None, 1), (12, 22, [5, 0, 9], 4), (8, 6, None, 1)])
def test_small_router_routes_as_route_topk(weights, moe, m, k, slots, steps):
    """The small-call router kernel's own routing (and its padded rows, and a reduced k) is route_topk's, bitwise."""
    from hibana.kernels import nemotron_moe
    from hibana.model.nemotron.moe import route_topk
    x = rows(weights[1], m, 200 + m)
    s = None if slots is None else torch.tensor(slots, dtype=torch.int32, device="cuda")
    fused, ids, w, _ = nemotron_moe.router(x, moe.small_weights(), moe.bias, k, s, steps)
    torch.testing.assert_close(fused, torch.mm(x, moe.gate_fc1.T, out_dtype=torch.float32), rtol=0, atol=1e-4)
    ref_ids, ref_w, _ = route_topk(fused, moe.bias, k, moe.latent, s, steps)
    assert torch.equal(ids, ref_ids) and torch.equal(w, ref_w)


@pytest.mark.parametrize("m", [1, 9, 64, 600])
def test_b12x_routed_matches_cutlass(weights, moe, m):
    """FlashInfer's b12x MoE (calls past 16 rows) on the checkpoint's tensors: the routed sum against
    cutlass_fused_moe's on the same pairs, same magnitude, eager and in a graph replay."""
    from hibana.model.nemotron.moe import route_topk
    x = rows(weights[1], m, 700 + m)
    fused = torch.mm(x, moe.gate_fc1.T, out_dtype=torch.float32)
    ids, w, lat = route_topk(fused, moe.bias, moe.topk, moe.latent)
    b12x = moe.b12x
    with torch.inference_mode():
        ref = cutlass_routed(moe, lat, ids, w).float()
        got = b12x(lat, ids, w)
        assert cos(got, ref) > B12X_COS
        assert abs(float(got.float().norm() / ref.norm()) - 1) < 0.03  # 0.984-1.015 of cutlass's on real inputs, 0.98 on Gaussian rows
        if m <= 64:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = b12x(lat, ids, w)
            g.replay()
            # b12x is not deterministic (two eager calls agree to 0.9997-0.99995): the replay against cutlass as eager
            assert cos(out, ref) > B12X_COS


def test_small_budget_path(weights, moe):
    """Budgeted selection on a small verify call: route_budget's ids through the small kernels (the plan from them)
    equal the large-call path's experts (the routed sum through `cutlass_routed`), to the W4A4 bound."""
    x = rows(weights[1], 8, 400)
    m = copy.copy(moe)
    m.keep, m.budget, m.per_request = 6, 64, 0
    ctx = ctx_for("verify", [3, 2], 4)
    big = copy.copy(m)
    big.SMALL_ROWS = 0
    big.routed = lambda lat, ids, w: cutlass_routed(big, lat, ids, w)
    with torch.inference_mode():
        assert cos(m(x, ctx), big(x, ctx)) > 0.9999


def test_padded_rows_repeat_row_zero(weights, moe):
    """Graph padding (dummy slot 0) names row 0's top expert at weight 0; real rows route as without slots."""
    from hibana.model.nemotron.moe import route_topk
    x = rows(weights[1], 3 * 4, 7)
    fused = torch.mm(x, moe.gate_fc1.T, out_dtype=torch.float32)
    plain = route_topk(fused, moe.bias, moe.topk, moe.latent)
    slots = torch.tensor([5, 0, 9], dtype=torch.int32, device="cuda")  # request 1 padded, verify steps 4
    ids, w, lat = route_topk(fused, moe.bias, moe.topk, moe.latent, slots, 4)
    real = torch.tensor([True] * 4 + [False] * 4 + [True] * 4, device="cuda")
    assert torch.equal(ids[real], plain[0][real]) and torch.equal(w[real], plain[1][real])
    assert (ids[~real] == plain[0][0, 0]).all() and (w[~real] == 0).all()
    assert torch.equal(lat, plain[2])


@pytest.mark.parametrize("m,cap", [(4, 44), (16, 64), (16, 128)])
def test_budget_invariants(weights, moe, m, cap):
    """The call reads at most the cap's experts (or the guaranteed ones when they exceed it); every real row keeps its
    guaranteed top experts first and takes only experts in the set, from its top 22, in rank order; weights are
    its sigmoid scores renormalized; padded rows read nothing new."""
    from hibana.model.nemotron.moe import route_budget
    x = rows(weights[1], m, 1000 + m)
    fused = torch.mm(x, moe.gate_fc1.T, out_dtype=torch.float32)
    keep = max(4, min(8, cap // m))
    slots = torch.tensor([1] * (m - 1) + [0], dtype=torch.int32, device="cuda")  # the last row padded
    ids, w, lat = route_budget(fused, moe.bias, moe.topk, moe.latent, keep, cap, slots, 1)
    s = fused[:, :moe.experts].sigmoid()
    order = torch.topk(s + moe.bias, moe.topk, dim=-1).indices
    used = ids[w > 0].unique()
    guaranteed = order[:m - 1, :keep].unique()
    assert len(used) <= max(cap, len(guaranteed))
    assert torch.isin(guaranteed, used).all()
    for r in range(m - 1):
        live = w[r] > 0
        assert torch.equal(ids[r, :keep].long(), order[r, :keep])
        picked = ids[r][live].long()
        pos = [int((order[r] == e).nonzero()) for e in picked.tolist()]
        assert pos == sorted(pos)
        assert torch.isin(order[r][torch.isin(order[r], used)][:moe.topk], picked).all()  # all its in-set candidates
        torch.testing.assert_close(w[r][live], s[r, picked] / s[r, picked].sum(), rtol=1e-6, atol=1e-7)
        assert (ids[r][~live] == ids[r, 0]).all()
    assert (w[m - 1] == 0).all() and (ids[m - 1] == order[0, 0]).all()
    assert torch.equal(lat, fused[:, moe.experts:].bfloat16())


@pytest.mark.parametrize("mode,m", [("decode", 4), ("verify", 8)])
def test_graph_replay_matches_eager(weights, moe, mode, m):
    x = rows(weights[1], m, 50 + m)
    ctx = ctx_for("decode", [3, 1, 2, 0]) if mode == "decode" else ctx_for("verify", [4, 0], m // 2)
    with torch.inference_mode():
        eager = moe(x, ctx)
        static = x.clone()
        moe(static, ctx)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = moe(static, ctx)
        static.copy_(rows(weights[1], m, 999))
        g.replay()
        other = moe(static, ctx)
        static.copy_(x)
        g.replay()
        assert torch.equal(out, eager)
    # padded requests' rows still come out finite (their expert slots weigh 0; the shared expert runs on them)
    assert torch.isfinite(other).all()


def test_mtp_block(weights):
    from hibana.model.nemotron.moe import NemotronMoe
    from hibana.target import NEMOTRON
    w = weights[1]
    mtp = NemotronMoe(w, MTP_PREFIX, torch.device("cuda"), NEMOTRON)
    assert mtp.expert_bytes == 512 * mtp.bytes_per_expert and mtp.bytes > mtp.expert_bytes
    x = rows(w, 5, 3)
    with torch.inference_mode():
        assert cos(mtp(x), mtp.reference(x, quantize_activations=True)) > 0.9999
    del mtp
    gc.collect()
    torch.cuda.empty_cache()


def test_bytes(moe):
    # packed NVFP4 (2688 x 1024 / 2 twice) and their E4M3 block scales per expert
    assert moe.bytes_per_expert == 2 * (2688 * 1024 // 2) + 2 * (2688 * 1024 // 16)
    assert moe.expert_bytes == 512 * moe.bytes_per_expert
    assert moe.bytes == moe.expert_bytes + moe.dense_bytes


@pytest.mark.parametrize("m,floor,cap", [(12, 16, 0), (24, 16, 0), (24, 12, 0), (24, 16, 400), (12, 8, 90)])
def test_floor_and_free_riders(weights, moe, m, floor, cap):
    """MoE sharing with SHARAR_MOE_KEEP=F (verify rows, 6 a request, the last request padded): every real row keeps its
    top F; the call's set is the union of those floors, grown by the cap (when larger) with the highest-scoring experts;
    each real row also takes its other top-22 experts that are in the set, in rank order; weights are its sigmoid
    scores renormalized over its kept experts; padded rows take nothing."""
    x = rows(weights[1], m, 1300 + m)
    fused = torch.mm(x, moe.gate_fc1.T, out_dtype=torch.float32)
    mm = copy.copy(moe)
    mm.keep, mm.budget, mm.per_request = floor, cap, 0
    reqs = m // 6
    ctx = ctx_for("verify", list(range(1, reqs)) + [0], 6)
    ids, w, lat = mm.route(fused, ctx)
    s = fused[:, :moe.experts].sigmoid()
    order = torch.topk(s + moe.bias, moe.topk, dim=-1).indices
    real = m - 6
    floors = set(order[:real, :floor].flatten().tolist())
    used = set(ids[:real][w[:real] > 0].tolist())
    assert floors <= used
    assert used == floors if cap <= len(floors) else len(used) <= cap
    for r in range(real):
        live = w[r] > 0
        want = [e for i, e in enumerate(order[r].tolist()) if i < floor or e in used]
        assert ids[r][live].tolist() == want
        torch.testing.assert_close(w[r][live], s[r, want] / s[r, want].sum(), rtol=1e-6, atol=1e-7)
    assert (w[real:] == 0).all()
    assert torch.equal(lat, fused[:, moe.experts:].bfloat16())


@pytest.mark.parametrize("m,floor,cap", [(12, 16, 60), (24, 16, 160), (24, 12, 128)])
def test_binding_cap(weights, moe, m, floor, cap):
    """A cap the floors' union exceeds: each real row is guaranteed its top `fallback` (cap / rows within [8, floor)),
    and the set fills to the cap by routing weight over the rows' top 22; rows take their top-22 experts in it, in rank
    order, renormalized."""
    x = rows(weights[1], m, 1700 + m)
    fused = torch.mm(x, moe.gate_fc1.T, out_dtype=torch.float32)
    mm = copy.copy(moe)
    mm.keep, mm.budget, mm.per_request = floor, cap, 0
    g = mm.fallback(m, 6)
    reqs = m // 6
    ids, w, _ = mm.route(fused, ctx_for("verify", list(range(1, reqs)) + [0], 6))
    s = fused[:, :moe.experts].sigmoid()
    order = torch.topk(s + moe.bias, moe.topk, dim=-1).indices
    real = m - 6
    union_f = set(order[:real, :floor].flatten().tolist())
    used = set(ids[:real][w[:real] > 0].tolist())
    if len(union_f) <= cap:
        assert union_f <= used and len(used) <= cap
    else:
        assert 0 < g < floor
        union_g = set(order[:real, :g].flatten().tolist())
        assert union_g <= used and len(used) == max(cap, len(union_g))
    for r in range(real):
        live = w[r] > 0
        want = [e for e in order[r].tolist() if e in used]
        assert ids[r][live].tolist() == want
        torch.testing.assert_close(w[r][live], s[r, want] / s[r, want].sum(), rtol=1e-6, atol=1e-7)
    assert (w[real:] == 0).all()
