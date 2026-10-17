"""Prefill-sized hyper-connection sites (kernels/hyper_prefill.py) against an FP64 reference of the unquantized math,
next to the decode kernels on the same inputs."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
from sharar.kernels import hyper_prefill as hp  # noqa: E402
from sharar.model.flashnext.hyper import HyperConnection, WideNorm  # noqa: E402

H, HC, R, EPS = 2560, 4, 320, 1e-6


def _site(g):
    norm = (torch.randn(HC * H, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    down = (torch.randn(R, HC * H, device="cuda", generator=g) / (HC * H) ** 0.5).to(torch.bfloat16)
    up = (torch.randn(HC * H, R, device="cuda", generator=g) / R ** 0.5).to(torch.bfloat16)
    inj = (torch.randn(HC, HC * H, device="cuda", generator=g) / (HC * H) ** 0.5).to(torch.bfloat16)
    return HyperConnection(norm, down, up, inj, HC, H, EPS, precision="fp8"), (norm, down, up, inj)


def _norm64(x, w):
    g = x.double().reshape(x.shape[0], HC, H)
    return (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + EPS)).reshape(x.shape[0], -1) * (1 + w.double())


def _rel(x, ref):
    return float((x.double() - ref).norm() / ref.norm())


@pytest.mark.parametrize("t", [hp.ROWS, 1500, 4096])
def test_site_tracks_the_reference_like_the_decode_kernels(t):
    """combine (site a) then mix (site b): the wide path's hyper' and block input are at least as close to FP64 as
    the decode kernels', and the wide path is taken from `ROWS` rows."""
    g = torch.Generator(device="cuda").manual_seed(t)
    a, araw = _site(g)
    b, braw = _site(g)
    hyper = (torch.randn(t, HC * H, device="cuda", generator=g)
             * torch.tensor([1.0, 3.0, 0.5, 8.0], device="cuda").repeat_interleave(H)).to(torch.bfloat16)
    block = (torch.randn(t, H, device="cuda", generator=g) * 2).to(torch.bfloat16)
    na = _norm64(hyper, araw[0])
    ref_h = (hyper.double().reshape(t, HC, H) + block.double()[:, None] * (2 * torch.sigmoid(na @ araw[3].double().T / HC))[..., None]).reshape(t, -1)
    nb = _norm64(ref_h, braw[0])
    ref_m = (torch.sigmoid(F.silu(nb @ braw[1].double().T / HC) @ braw[2].double().T) * nb).reshape(t, HC, H).mean(1)

    _, res = a.mix(hyper)
    assert isinstance(res[1], WideNorm)
    wide_h, normed = a.combine(block, res, then=b)
    assert isinstance(normed, WideNorm)
    wide_m, _ = b.mix(wide_h, normed)

    from sharar.kernels import flashnext_fused as fk  # the decode kernels' chain on the same rows
    n_a, _ = fk.grouped_gemma_rms(hyper, a.norm, EPS, H, amax=True)
    old_h, n_b, parts_b = fk.inject_combine_norm(hyper, block, a.inject(n_a), HC, H, b.norm, EPS)
    old_m = fk.gate_mean(b.up(b.down(n_b, amax_parts=parts_b), act=fk.ACT_SILU), n_b, HC, H)

    assert _rel(wide_h, ref_h) <= _rel(old_h, ref_h) * 1.05
    assert _rel(wide_m, ref_m) <= _rel(old_m, ref_m) * 1.05
    assert _rel(wide_m, ref_m) < 1e-2


def test_wide_path_is_graph_capturable_and_deterministic():
    g = torch.Generator(device="cuda").manual_seed(1)
    a, _ = _site(g)
    b, _ = _site(g)
    t = hp.ROWS
    hyper = torch.randn(t, HC * H, device="cuda", generator=g).to(torch.bfloat16)
    block = torch.randn(t, H, device="cuda", generator=g).to(torch.bfloat16)

    def run():
        _, res = a.mix(hyper)
        h, n = a.combine(block, res, then=b)
        return h, b.mix(h, n)[0]

    eager = run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    for x, y in zip(eager, captured):
        assert torch.equal(x.view(torch.int16), y.view(torch.int16))


def test_last_combine_without_a_next_site():
    """`combine` with no next site (the last layer) adds the block output with the wide residual's inject logits."""
    g = torch.Generator(device="cuda").manual_seed(2)
    a, _ = _site(g)
    t = hp.ROWS
    hyper = torch.randn(t, HC * H, device="cuda", generator=g).to(torch.bfloat16)
    block = torch.randn(t, H, device="cuda", generator=g).to(torch.bfloat16)
    _, res = a.mix(hyper)
    out = a.combine(block, res)
    inj = 2 * torch.sigmoid(res[1].inj)
    ref = (hyper.float().reshape(t, HC, H) + block.float()[:, None] * inj[..., None]).reshape(t, -1)
    assert torch.allclose(out.float(), ref, atol=0.05, rtol=1e-2)
