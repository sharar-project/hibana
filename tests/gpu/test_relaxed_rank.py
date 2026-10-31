"""kernels/relaxed_rank.rank_gap against the PyTorch statistics it replaces, ties and skipped cells included."""
import pytest
import torch

from sharar.kernels.relaxed_rank import SKIPPED, rank_gap


@pytest.mark.parametrize('vocab', [248320, 1000, 5])
def test_rank_gap_matches_the_reference(vocab):
    g = torch.Generator(device='cuda').manual_seed(vocab)
    logits = (torch.randn((40, vocab), device='cuda', generator=g) * 4).round().to(torch.bfloat16)  # many ties
    rows = torch.randint(0, 40, (37,), device='cuda', generator=g)
    tokens = torch.randint(0, vocab, (37,), device='cuda', generator=g)
    rows[::5] = -1
    above, gap = rank_gap(logits, rows, tokens)
    kept = rows >= 0
    picked = logits[rows[kept]]
    value = picked.gather(-1, tokens[kept, None]).squeeze(-1).float()
    assert torch.equal(above[kept], (picked > value[:, None]).sum(-1).int())
    assert torch.equal(gap[kept], picked.max(-1).values.float() - value)
    assert (above[~kept] == SKIPPED).all() and torch.isinf(gap[~kept]).all()
