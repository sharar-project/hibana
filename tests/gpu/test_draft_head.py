"""Draft-only 3-bit lm_head: picks map to global ids through padded rows, and the 3-bit
copy picks what the FP8 head picks when the winner is clear."""
import pytest
import torch



def _fp8_head(n: int, k: int):
    from sharar.model.flashnext.linear import TrunkLinear
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda")
    w = (w / w.norm(dim=-1, keepdim=True) * 8).to(torch.bfloat16)
    return TrunkLinear.from_bf16(w, "fp8"), w


@pytest.mark.parametrize("rows", [1, 4])
def test_w3_head_full_vocabulary(rows):
    from sharar.proposers.draft_head import ShortlistHead
    lm_head, w = _fp8_head(4096, 2560)
    want = torch.randint(0, 4096, (rows,), device="cuda")
    x = w[want]  # each row's own weight row wins by a wide margin
    head = ShortlistHead(lm_head, fmt="w3")
    assert head.bytes == head.w3.bytes > 0
    assert torch.equal(head.argmax(x), want)


def test_w3_head_reduced_vocabulary_pads_and_maps():
    from sharar.proposers.draft_head import ShortlistHead
    lm_head, w = _fp8_head(4096, 2560)
    ids = torch.randperm(4096, device="cuda")[:1000]  # not a multiple of 128: padded to 1024
    head = ShortlistHead(lm_head, ids, "w3")
    assert head.ids.numel() == 1024 and head.w3.out_features == 1024
    inside = ids[:4]
    assert torch.equal(head.argmax(w[inside]), inside)


def test_w3_head_takes_more_rows_than_the_kernel():
    from sharar.proposers.draft_head import ShortlistHead
    lm_head, w = _fp8_head(1024, 2560)
    want = torch.randint(0, 1024, (150,), device="cuda")
    assert torch.equal(ShortlistHead(lm_head, fmt="w3").argmax(w[want]), want)


def test_rescored_shortlist_picks_the_fp8_heads_argmax():
    from sharar.proposers.draft_head import ShortlistHead
    lm_head, _ = _fp8_head(8192, 2560)
    torch.manual_seed(1)
    x = torch.randn(64, 2560, device="cuda").to(torch.bfloat16)  # random rows: many near-ties for the 3-bit logits
    exact = (x.float() @ lm_head.weight.float().T).argmax(-1)
    plain = ShortlistHead(lm_head, fmt="w3").argmax(x)
    rescored = ShortlistHead(lm_head, fmt="w3", rescore=16).argmax(x)
    assert (rescored == exact).float().mean() >= (plain == exact).float().mean()
    assert (rescored == exact).float().mean() > 0.95
    ids = torch.arange(0, 8192, 3, device="cuda")
    sub = ShortlistHead(lm_head, ids, "w3", 16).argmax(x)
    assert torch.isin(sub, ids).all()
    assert (sub == ids[(x.float() @ lm_head.weight[ids].float().T).argmax(-1)]).float().mean() > 0.95
