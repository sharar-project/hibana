"""Feedback must cover exactly the processed inputs, with owned cache endpoints.

Updated from main (drafter metadata in `Snapshot.auxiliary`): decoding places no snapshot, so feedback writes no rows;
the reply end gets the row its release inserts (`prepare_release`), an owned copy of the last committed one."""
from types import SimpleNamespace

import pytest
import torch

from sharar.cache.snapshot import Snapshot
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
                        private={}))
    proposer.accepted_batch(requests, outputs, VerifiedFeatures(norm))
    assert len(calls) == len(counts)
    ends = []
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
        end = Snapshot(9, req.committed_len)
        proposer.prepare_release(req, end)
        assert end.auxiliary[proposer.cache_identity][0] == req.committed_len
        assert torch.equal(end.auxiliary[proposer.cache_identity][1], norm[row, count - 1])
        ends.append(end)
    saved = ends[0].auxiliary[proposer.cache_identity][1].clone()
    norm.zero_()
    assert torch.equal(ends[0].auxiliary[proposer.cache_identity][1], saved)


class _RowHead:
    """write_kv with no cross-row arithmetic: each page keeps (token, hidden row, position)."""
    hidden_width = 2

    def __init__(self):
        self.bytes, self.kv = 0, {}

    def write_kv(self, tokens, hidden, positions, pages):
        for t, h, p, page in zip(tokens.tolist(), hidden.tolist(), positions.tolist(), pages.tolist()):
            self.kv[page] = (t, h, p)


@pytest.mark.parametrize('counts', [(1,), (5,), (1, 5, 2, 3), (3, 1, 0)])
def test_device_feedback_matches_host_feedback(counts):
    """accepted_prepare + feedback_device + accepted_publish (device acceptance) commit the rows and the
    pending input the host's accepted_batch commits; padded bucket rows (count 0) write only page 0."""
    s = 5
    real = [c for c in counts if c]
    norm = torch.randn(len(counts), s, 2).to(torch.bfloat16)  # the verify rows
    heads, results = {}, {}
    for path in ('host', 'device'):
        head = heads[path] = _RowHead()
        proposer = MtpProposer(head, ProposerContext(torch.device('cpu'), 4, 512), k=s - 1)
        requests, outputs, tokens, pages = [], [], [], []
        for row, count in enumerate(counts):
            start = 7 + row * 10 if count else 0
            drafts = [200 + row * 10 + j for j in range(s - 1)]
            req = SimpleNamespace(proposer_state=MtpState(draft_pos=start, pending_token=50 + row,
                                  pending_hidden=torch.full((1, 2), -1. - row, dtype=torch.bfloat16)), committed_len=start,
                                  finished=row == 1, pages=torch.arange(512).int() + 1 + row * 512,
                                  private={})
            tokens += [50 + row] + drafts
            pages.append(req.pages[start:start + s] if count else torch.zeros(s, dtype=torch.int32))
            if count:
                requests.append(req)
                outputs.append((drafts[:count - 1] + [900 + row])[:count])
        if path == 'host':
            for req, out in zip(requests, outputs):
                req.committed_len += len(out)
            proposer.accepted_batch(requests, outputs, VerifiedFeatures(norm[:len(real)]))
        else:
            starts, _ = proposer.accepted_prepare(requests)
            b = len(counts)
            proposer.feedback_device(None, torch.tensor(starts + [0] * (b - len(starts))), torch.zeros(b, dtype=torch.int64),
                                     torch.tensor(counts), s, norm=norm.reshape(b * s, 2), tokens=torch.tensor(tokens),
                                     pages=torch.cat(pages))
            for req, out in zip(requests, outputs):
                req.committed_len += len(out)
            proposer.accepted_publish(requests, outputs)
        results[path] = requests
    assert {p: v for p, v in heads['device'].kv.items() if p} == heads['host'].kv
    for h, d in zip(results['host'], results['device']):
        sh, sd = h.proposer_state, d.proposer_state
        assert (sh.draft_pos, sh.pending_token) == (sd.draft_pos, sd.pending_token)
        assert torch.equal(sh.last_hidden, sd.last_hidden)
        assert (sh.pending_hidden is None) == (sd.pending_hidden is None)
        if sh.pending_hidden is not None:
            assert torch.equal(sh.pending_hidden, sd.pending_hidden)
        eh, ed = Snapshot(1, h.committed_len), Snapshot(2, d.committed_len)
        proposer.prepare_release(h, eh)
        proposer.prepare_release(d, ed)
        (ph, rh), = eh.auxiliary.values()
        (pd, rd), = ed.auxiliary.values()
        assert ph == pd == h.committed_len and torch.equal(rh, rd)
