"""The reference's sparse-attention selection against SGLang's own torch
reference operators (vendored below from sglang v0.5.20,
srt/layers/attention/qsa/{mqa,kernel,metadata}.py, Apache-2.0), at small
budgets where selection actually drops blocks."""
import math

import pytest
import torch

from sharar.model.flashnext.reference import qsa_token_mask


# ---- vendored SGLang torch references (single sequence) ----
def torch_qsa_mqa_prefill(q, k, row_starts, row_ends):
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    columns = torch.arange(k.shape[0]).unsqueeze(0)
    valid = (columns >= row_starts.reshape(-1, 1)) & (columns < row_ends.reshape(-1, 1))
    return logits.masked_fill(~valid, -float("inf"))


def qsa_fast_topk_cpu(logits, row_starts, row_ends, topk):
    lengths, starts = (row_ends - row_starts).to(torch.int32), row_starts.to(torch.int32)
    output = torch.full((logits.shape[0], topk), -1, dtype=torch.int32)
    for row in range(logits.shape[0]):
        start, length = int(starts[row]), int(lengths[row])
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(logits[row, start:start + length], width).indices.to(torch.int32)
    return output


def torch_expand_qsa_block_indices(block_indices, query_positions, sequence_lengths, compress_ratio, token_topk):
    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    rows = block_indices.shape[0]
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where((expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)), expanded, torch.full_like(expanded, -1))
    tail_offsets = torch.arange(compress_ratio - 1, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = torch.div(visible_tokens, compress_ratio, rounding_mode="floor") * compress_ratio
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (tail < sequence_lengths.unsqueeze(1))
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))
    return torch.cat([expanded, tail], dim=1)


def sglang_mask(q, keys, budget, ratio):
    t = q.shape[0]
    positions = torch.arange(t)
    g = keys.shape[0]
    row_starts = torch.zeros(t, dtype=torch.int32)
    row_ends = torch.minimum((positions + 1) // ratio, torch.tensor(g)).to(torch.int32)
    logits = torch_qsa_mqa_prefill(q, keys[:, None, :], row_starts, row_ends)
    blocks = qsa_fast_topk_cpu(logits, row_starts, row_ends, topk=budget // ratio)
    tokens = torch_expand_qsa_block_indices(blocks, positions, torch.full((t,), t), ratio, budget)
    mask = torch.zeros((t, t), dtype=torch.bool)
    for i in range(t):
        chosen = tokens[i][tokens[i] >= 0]
        mask[i, chosen] = True
    return mask


@pytest.mark.parametrize("t,budget,ratio,seed", [(200, 32, 4, 0), (203, 32, 4, 1), (130, 16, 2, 2), (64, 256, 4, 3), (5, 32, 4, 4)])
def test_selection_matches_sglang(t, budget, ratio, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(t, 4, 16, generator=g)
    keys = torch.randn(t // ratio, 16, generator=g)
    ours, theirs = qsa_token_mask(q, keys, budget, ratio), sglang_mask(q, keys, budget, ratio)
    assert torch.equal(ours, theirs)
    assert not torch.triu(ours, diagonal=1).any()  # causal
    if t // ratio <= budget // ratio:
        assert torch.equal(ours, torch.tril(torch.ones(t, t, dtype=torch.bool)))  # dense below the budget
    else:
        assert ours.sum(1).max() <= budget + ratio - 1
