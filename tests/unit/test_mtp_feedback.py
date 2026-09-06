"""Feedback must cover exactly the processed inputs, with owned cache endpoints."""
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.contracts import ProposerContext, VerifiedFeatures
from sharar.proposers.mtp import MtpProposer, MtpState


@pytest.mark.parametrize('counts', [(1,), (4,), (1, 4, 2, 3)])
def test_feedback_token_hidden_alignment_and_stopped_rows(counts):
    calls = []
    head = SimpleNamespace(bytes=0, write_kv=lambda *args: calls.append([x.clone() for x in args]))
    proposer = MtpProposer(head, ProposerContext(torch.device('cpu'), 4, 512))
    requests, outputs = [], []
    norm = torch.arange(len(counts) * 4 * 2).reshape(len(counts), 4, 2).float()
    for row, count in enumerate(counts):
        start = 7 + row * 10
        outputs.append(list(range(100 + row * 10, 100 + row * 10 + count)))
        requests.append(SimpleNamespace(proposer_state=MtpState(draft_pos=start, pending_token=50 + row,
                        pending_hidden=torch.full((1, 2), -1. - row)), committed_len=start + count,
                        finished=row % 2 == 0, pages=torch.arange(512).int() + row * 512,
                        checkpoints={start + 1: 1}, cache_auxiliary={}))
    proposer.accepted_batch(requests, outputs, VerifiedFeatures(norm))
    assert len(calls) == len(counts)
    for row, (req, count, args) in enumerate(zip(requests, counts, calls)):
        tokens, hidden, positions, pages = args
        start = 7 + row * 10
        assert tokens.tolist() == [50 + row] + outputs[row][:-1]
        assert torch.equal(hidden, torch.cat([torch.full((1, 2), -1. - row), norm[row, :count - 1]]))
        assert positions.tolist() == list(range(start, start + count))
        assert torch.equal(pages, req.pages[start:start + count])
        state = req.proposer_state
        assert state.draft_pos == req.committed_len
        assert state.pending_token == (None if req.finished else outputs[row][-1])
        for position, entry in req.cache_auxiliary.items():
            assert torch.equal(entry[proposer.cache_identity][1], norm[row, position - start - 1])
    saved = requests[0].cache_auxiliary[requests[0].committed_len][proposer.cache_identity][1].clone()
    norm.zero_()
    assert torch.equal(requests[0].cache_auxiliary[requests[0].committed_len][proposer.cache_identity][1], saved)
