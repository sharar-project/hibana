"""Checkpoint MTP adapter. Draft KV/hidden state and graphs stay here."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sharar.engine.contracts import FeatureRequest, ProposerContext, ProposerRequirements


@dataclass
class MtpState:
    draft_pos: int = 0
    last_hidden: torch.Tensor | None = None
    pending_token: int | None = None
    pending_hidden: torch.Tensor | None = None


class MtpProposer:
    def __init__(self, head, context: ProposerContext, k: int = 3):
        self.head, self.context, self.k = head, context, k
        self.requirements = ProposerRequirements("mtp", FeatureRequest(final_norm=True), persistent_bytes=head.bytes)
        self.graphs = None

    def _state(self, req) -> MtpState:
        if req.proposer_state is None:
            req.proposer_state = MtpState()
        if not isinstance(req.proposer_state, MtpState):
            raise TypeError("request belongs to another proposer")
        return req.proposer_state

    def ingest_prefill(self, req, new_ids, start, result):
        state = self._state(req)
        norm_rows = result["norm"]
        n, end = len(new_ids), start + len(new_ids)
        first = max(1, start)
        if start > 0 and state.last_hidden is None:
            first = start + 1
        if first < end:
            toks = torch.tensor(new_ids[first - start:], dtype=torch.int64, device=self.context.device)
            if first == start:
                hid = torch.cat([state.last_hidden.unsqueeze(0), norm_rows[:n - 1]], dim=0)
            else:
                hid = norm_rows[first - start - 1:n - 1]
            pos = torch.arange(first, end, dtype=torch.int64, device=self.context.device)
            pages = req.pages[first:end]
            kv_pages = req.pages[1:end] if first == 1 or state.draft_pos >= first else req.pages[first:end]
            self.head.forward(toks, hid, pos, pages, [end - first], [kv_pages])
            state.draft_pos = end
        else:
            state.draft_pos = max(state.draft_pos, first)
        state.last_hidden = norm_rows[-1]

    def start(self, req, token):
        state = self._state(req)
        state.pending_token = token
        state.pending_hidden = state.last_hidden.unsqueeze(0)

    def ready(self, req):
        return isinstance(req.proposer_state, MtpState) and req.proposer_state.pending_token is not None

    def propose(self, requests, pages, inputs, *, graphs):
        hids, kv = [], []
        for i, req in enumerate(requests):
            state = self._state(req)
            pos = req.context_len - 1
            if state.draft_pos != pos or state.pending_token != req.output_ids[-1]:
                raise ValueError("draft pending input does not match committed prefix")
            hids.append(state.pending_hidden)
            kv.append(torch.cat([req.pages[1:pos], pages[i, :1]]))
        hidden, token = self._step(inputs.tokens, torch.cat(hids), inputs.positions, pages[:, 0].contiguous(), kv, graphs)
        drafts = [token]
        for j in range(2, self.k + 1):
            kv = [torch.cat([r.pages[1:r.context_len - 1], pages[i, :j]]) for i, r in enumerate(requests)]
            hidden, token = self._step(drafts[-1], hidden, inputs.positions + j - 1, pages[:, j - 1].contiguous(), kv, graphs)
            drafts.append(token)
        return torch.stack(drafts, dim=1)

    def _step(self, tokens, hidden, positions, pages, kv, graphs):
        if graphs and self.graphs is not None:
            hidden, token = self.graphs.run(tokens, hidden, positions, pages, kv)
            return hidden.clone(), token.clone()  # graph outputs are borrowed until next replay
        hidden = self.head.forward(tokens, hidden, positions, pages, [1] * len(kv), kv)
        return hidden, self.head.logits(hidden).argmax(-1)

    def accepted(self, req, emitted, features, row):
        state = self._state(req)
        state.pending_token = None if req.finished else int(emitted[-1])
        state.pending_hidden = None if req.finished else features.norm[row, len(emitted) - 1:len(emitted)]
        state.draft_pos = req.committed_len

    def release(self, req):
        req.proposer_state = None

    def capture_graphs(self, log=None):
        from sharar.proposers.mtp_graphs import DraftGraphs
        self.graphs = DraftGraphs(self)
        self.graphs.capture(log=log)
