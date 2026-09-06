import math

import pytest
import torch

from bench.attention_reference import causal_attention


def test_causal_alignment_includes_prefix_and_current_row_but_not_future():
    q, k = torch.zeros(2, 2, 2), torch.zeros(4, 1, 2)
    v = torch.tensor([[[1., 3]], [[3., 5]], [[5., 7]], [[7., 9]]])
    expected = torch.tensor([[[3., 5], [3., 5]], [[4., 6], [4., 6]]], dtype=torch.float64)
    assert torch.equal(causal_attention(q, k, v), expected)


def test_grouped_head_mapping_and_accumulation_against_scalar_reference():
    generator = torch.Generator().manual_seed(413)
    q = torch.randn(3, 4, 2, generator=generator).bfloat16()
    k = torch.randn(5, 2, 2, generator=generator).bfloat16()
    v = torch.randn(5, 2, 2, generator=generator).bfloat16()
    expected = torch.empty(3, 4, 2, dtype=torch.float64)
    for row in range(3):
        for head in range(4):
            scores = [sum(float(q[row, head, d]) * float(k[pos, head // 2, d]) for d in range(2)) / math.sqrt(2)
                      for pos in range(3 + row)]
            exp = [math.exp(s - max(scores)) for s in scores]
            for d in range(2):
                expected[row, head, d] = sum(e * float(v[pos, head // 2, d]) for pos, e in enumerate(exp)) / sum(exp)
    assert torch.allclose(causal_attention(q, k, v), expected, rtol=1e-14, atol=1e-14)


def test_impossible_query_geometry_is_rejected():
    with pytest.raises(ValueError, match='geometry'):
        causal_attention(torch.ones(8, 2, 2), torch.ones(4, 1, 2), torch.ones(4, 1, 2))
