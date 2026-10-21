"""Draft-only 3-bit and 2-bit lm_heads: picks map to global ids through padded rows, the low-bit
copy picks what the FP8 head picks when the winner is clear, and the 2-bit GEMM computes its matrix."""
import pytest
import torch



def _fp8_head(n: int, k: int):
    from sharar.model.flashnext.linear import TrunkLinear
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda")
    w = (w / w.norm(dim=-1, keepdim=True) * 8).to(torch.bfloat16)
    return TrunkLinear.from_bf16(w, "fp8"), w


@pytest.mark.parametrize("m", [1, 3, 16, 17, 40, 64])
@pytest.mark.parametrize("n,k", [(256, 512), (2048, 2560)])
def test_two_bit_gemm_matches_its_dequantized_matrix(n, k, m):
    from sharar.kernels import w3
    torch.manual_seed(n + m)
    packed = w3.pack(torch.randn(n, k, device="cuda") * 0.02, "lloyd2", fit_scales=True)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, s = w3.quant_groups(x)
    ref = (q.float() * s.float().repeat_interleave(32, dim=1)) @ w3.dequantize(packed).T
    y = w3.w3_linear(x, packed).float()
    assert ((y - ref).norm() / ref.norm()).item() < 5e-3


@pytest.mark.parametrize("fmt", ["w3", "w2"])
@pytest.mark.parametrize("rows", [1, 4])
def test_low_bit_head_full_vocabulary(rows, fmt):
    from sharar.proposers.draft_head import ShortlistHead
    lm_head, w = _fp8_head(4096, 2560)
    want = torch.randint(0, 4096, (rows,), device="cuda")
    x = w[want]  # each row's own weight row wins by a wide margin
    head = ShortlistHead(lm_head, fmt=fmt)
    assert head.bytes == head.w3.bytes > 0 and head.w3.bits == int(fmt[1])
    assert torch.equal(head.argmax(x), want)


@pytest.mark.parametrize("fmt", ["w3", "w2"])
def test_low_bit_head_reduced_vocabulary_pads_and_maps(fmt):
    from sharar.proposers.draft_head import ShortlistHead
    lm_head, w = _fp8_head(4096, 2560)
    ids = torch.randperm(4096, device="cuda")[:1000]  # not a multiple of 128: padded to 1024
    head = ShortlistHead(lm_head, ids, fmt)
    assert head.ids.numel() == 1024 and head.w3.out_features == 1024
    inside = ids[:4]
    assert torch.equal(head.argmax(w[inside]), inside)


@pytest.mark.parametrize("fmt", ["w3", "w2"])
def test_low_bit_head_takes_more_rows_than_the_kernel(fmt):
    from sharar.proposers.draft_head import ShortlistHead
    lm_head, w = _fp8_head(1024, 2560)
    want = torch.randint(0, 1024, (150,), device="cuda")
    assert torch.equal(ShortlistHead(lm_head, fmt=fmt).argmax(w[want]), want)


@pytest.mark.parametrize("fmt", ["w3", "w2"])
def test_rescored_shortlist_picks_the_fp8_heads_argmax(fmt):
    from sharar.proposers.draft_head import ShortlistHead
    # random rows over random weights are near-ties everywhere: the 2-bit copy's candidates miss more of them
    # (the served head's top 16 hold the FP8 argmax on 99.8% of recorded draft steps)
    floor = {"w3": 0.95, "w2": 0.9}[fmt]
    lm_head, _ = _fp8_head(8192, 2560)
    torch.manual_seed(1)
    x = torch.randn(64, 2560, device="cuda").to(torch.bfloat16)  # random rows: many near-ties for the low-bit logits
    exact = (x.float() @ lm_head.weight.float().T).argmax(-1)
    plain = ShortlistHead(lm_head, fmt=fmt).argmax(x)
    rescored = ShortlistHead(lm_head, fmt=fmt, rescore=16).argmax(x)
    assert (rescored == exact).float().mean() >= (plain == exact).float().mean()
    assert (rescored == exact).float().mean() > floor
    ids = torch.arange(0, 8192, 3, device="cuda")
    sub = ShortlistHead(lm_head, ids, fmt, 16).argmax(x)
    assert torch.isin(sub, ids).all()
    assert (sub == ids[(x.float() @ lm_head.weight[ids].float().T).argmax(-1)]).float().mean() > floor
