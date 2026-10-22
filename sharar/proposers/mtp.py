"""Checkpoint MTP adapter. Draft KV/hidden state and graphs stay here."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sharar.target import DEFAULT_TARGET

from sharar.engine.config import current
from sharar.engine.contracts import Capabilities, FeatureRequest, ProposerContext, ProposerRequirements
from sharar.engine.rejection import DraftColumns, Proposal


@dataclass
class MtpState:
    draft_pos: int = 0
    last_hidden: torch.Tensor | None = None
    pending_token: int | None = None
    pending_hidden: torch.Tensor | None = None
    valid_prefix: bool = True


def draft_argmax(head, hidden: torch.Tensor) -> torch.Tensor:
    """The head's draft token through its draft head (`proposers/draft_head.py`): the target's own head by default."""
    return head.draft_head.argmax(head.head_input(hidden))


def draft_choice(head, hidden: torch.Tensor, with_confidence: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`draft_argmax`, and with `with_confidence` the draft head's [rows, 2] confidence in it (SHARAR_TRACE_CONFIDENCE)."""
    if not with_confidence:
        return draft_argmax(head, hidden), None
    return head.draft_head.argmax_confidence(head.head_input(hidden))


class MtpProposer:
    def __init__(self, head, context: ProposerContext, k: int = 3):
        self.head, self.context, self.k = head, context, k
        # Drafts are the head's argmax: deterministic proposals, so sampled requests verify them
        # exactly as point masses (engine/rejection.py), like the parallel drafts.
        self.requirements = ProposerRequirements("mtp", FeatureRequest(final_norm=True), persistent_bytes=head.bytes,
                                                 greedy_only=False, max_depth=8,
                                                 capabilities=Capabilities(device_feedback=True, draft_head=getattr(head, "DRAFT_HEADS", False),
                                                                           short_drafts=True, stored_payload=True))
        self.graphs = None
        # The draft's confidence per step for the trace round records (SHARAR_TRACE_CONFIDENCE); read once
        # here, the draft graphs capture the extra output.
        self.with_confidence = current().trace_confidence
        self._fb = None  # device acceptance buffers (`_feedback_buffers`)
        self._host_tokens = torch.empty(context.max_batch * (k + 1), dtype=torch.int64,
                                        pin_memory=context.device.type == "cuda")
        self._upload_done = torch.cuda.Event() if context.device.type == "cuda" else None
        self._upload_pending = False
        # Each drafted column's host copy, behind its step (DraftColumns): Flash-Next reads a draft
        # token's PLE rows while the next step runs.
        self._host_drafts = torch.empty((k, context.max_batch), dtype=torch.int64, pin_memory=context.device.type == "cuda")
        self._draft_events = [torch.cuda.Event() for _ in range(k)] if context.device.type == "cuda" else None
        target = getattr(head, 'target', DEFAULT_TARGET)
        self.target = target
        self.cache_identity = (target.model_id, target.model_revision, "native-mtp", "final_norm-v1", "bf16-kv-target-forced-v1")
        self.cache_hits = self.cache_fallbacks = 0

    def auxiliary_codec(self):
        """Export/import of this proposer's cache auxiliary, (position, last hidden row), for the SSD tier: a restored
        request drafts from where the stored one stopped (with its draft KV, `stored_pages`) instead of never again."""
        def export(value):
            return value  # (position, hidden row on the device): the scalar first, as the tier stores them

        def import_(payload, *, slot=None):
            position, hidden = payload  # hidden is a view into the tier's staging buffer, which the next restore reuses
            return (position, hidden.clone())

        return self.cache_identity, export, import_

    def stored_pages(self):
        """The draft layer's KV, indexed by the target's page ids: stored with the target's KV blocks, so a restored
        prefix keeps the draft's attention history too."""
        kv = self.head.kv
        return [(f"draft.{j}.{t}", pool) for j, lid in enumerate(kv.layer_ids) for t, pool in (("k", kv.k[lid]), ("v", kv.v[lid]))]

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
        if start and state.last_hidden is None:
            cached = req.match.node.auxiliary.get(self.cache_identity) if req.match is not None else None
            if cached is not None and cached[0] == start:
                state.last_hidden, state.draft_pos = cached[1], start
                self.cache_hits += 1
            else:
                state.valid_prefix = False
                self.cache_fallbacks += 1
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
            self.head.write_kv(toks, hid, pos, pages)
            state.draft_pos = end
        else:
            state.draft_pos = max(state.draft_pos, first)
        state.last_hidden = norm_rows[-1].clone()
        self._checkpoint(req, start, end, norm_rows)

    def _checkpoint(self, req, start, end, norm_rows):
        state = self._state(req)
        # Interior state checkpoints plus only the latest end feature.
        req.cache_auxiliary = {p: value for p, value in req.cache_auxiliary.items() if p in req.checkpoints}
        if state.valid_prefix:
            for position in (*req.checkpoints, end):
                if start < position <= end:
                    req.cache_auxiliary[position] = {self.cache_identity: (position, norm_rows[position - start - 1].clone())}

    def start(self, req, token):
        state = self._state(req)
        state.pending_token = token
        state.pending_hidden = state.last_hidden.unsqueeze(0)

    def ready(self, req):
        return isinstance(req.proposer_state, MtpState) and req.proposer_state.valid_prefix and req.proposer_state.pending_token is not None

    def propose(self, requests, pages, inputs, *, graphs, depth=None):
        """[batch, depth] drafts, one sequential head step each (depth: the round's verify depth, else k)."""
        hids, kv = [], []
        for i, req in enumerate(requests):
            state = self._state(req)
            pos = req.context_len - 1
            if state.draft_pos != pos or state.pending_token != req.output_ids[-1]:
                raise ValueError("draft pending input does not match committed prefix")
            hids.append(state.pending_hidden)
            kv.append(torch.cat([req.pages[1:pos], pages[i, :1]]))
        hidden, token, conf = self._step(inputs.tokens, torch.cat(hids), inputs.positions, pages[:, 0].contiguous(), kv, graphs)
        drafts, confs = [token], [conf]
        self._column(0, token)
        for j in range(2, (self.k if depth is None else depth) + 1):
            kv = [torch.cat([r.pages[1:r.context_len - 1], pages[i, :j]]) for i, r in enumerate(requests)]
            hidden, token, conf = self._step(drafts[-1], hidden, inputs.positions + j - 1, pages[:, j - 1].contiguous(), kv, graphs)
            drafts.append(token)
            confs.append(conf)
            self._column(j - 1, token)
        columns = DraftColumns(self._host_drafts, self._draft_events, len(requests)) if self._draft_events is not None else None
        confidence = torch.stack(confs, dim=1) if self.with_confidence else None
        return Proposal(torch.stack(drafts, dim=1), confidence=confidence, columns=columns)

    def _column(self, j: int, token: torch.Tensor) -> None:
        if self._draft_events is not None:
            self._host_drafts[j, :token.shape[0]].copy_(token, non_blocking=True)
            self._draft_events[j].record()

    def _step(self, tokens, hidden, positions, pages, kv, graphs):
        if graphs and self.graphs is not None:
            hidden, token, conf = self.graphs.run(tokens, hidden, positions, pages, kv)
            hidden, token, conf = hidden.clone(), token.clone(), conf.clone() if conf is not None else None  # graph outputs are borrowed until next replay
        else:
            hidden = self.head.forward(tokens, hidden, positions, pages, [1] * len(kv), kv)
            token, conf = draft_choice(self.head, hidden, self.with_confidence)
        return hidden, token, conf

    def accepted_batch(self, requests, emitted, features):
        # The normal acceptance download completes the preceding upload. Keep
        # the staging lifetime safe for direct callers too, without a device
        # barrier on the serving path.
        if self._upload_pending and not self._upload_done.query():
            self._upload_done.synchronize()
        if not 0 < len(requests) == len(emitted) <= self.context.max_batch:
            raise ValueError("invalid MTP feedback batch")
        offset = 0
        for row, (req, tokens) in enumerate(zip(requests, emitted)):
            state = self._state(req)
            count = len(tokens)
            if (not 1 <= count <= self.k + 1 or state.pending_token is None
                    or state.pending_hidden is None or not state.valid_prefix
                    or state.draft_pos != req.committed_len - count
                    or features.norm is None or features.norm.shape[0] <= row
                    or features.norm.shape[1] < count):
                raise ValueError("MTP feedback does not follow the pending input")
            self._host_tokens[offset] = state.pending_token
            for j, token in enumerate(tokens[:-1], 1):
                self._host_tokens[offset + j] = token
            offset += count
        device_tokens = self._host_tokens[:offset].to(self.context.device, non_blocking=True)
        if self._upload_done is not None:
            self._upload_done.record()
            self._upload_pending = True
        offset = 0
        for row, (req, tokens) in enumerate(zip(requests, emitted)):
            self._accepted(req, tokens, features, row, device_tokens[offset:offset + len(tokens)])
            offset += len(tokens)

    def _accepted(self, req, emitted, features, row, inputs):
        state = self._state(req)
        count = len(emitted)
        start, end = req.committed_len - count, req.committed_len
        norm = features.norm[row, :count]
        hidden = torch.cat([state.pending_hidden, norm[:count - 1]])
        positions = torch.arange(start, end, dtype=torch.int64, device=self.context.device)
        # Preserve each row's projection shape/rounding. Accepted draft KV must
        # be target-conditioned, including the fully accepted bonus input.
        self.head.write_kv(inputs, hidden, positions, req.pages[start:end])
        state.last_hidden = norm[-1].clone()
        self._checkpoint(req, start, end, norm)
        state.pending_token = None if req.finished else int(emitted[-1])
        state.pending_hidden = None if req.finished else features.norm[row, len(emitted) - 1:len(emitted)]
        state.draft_pos = req.committed_len

    # ---- device-resident acceptance (engine/accept_graphs.py, SHARAR_DEVICE_ACCEPT=1) ----
    def _feedback_buffers(self) -> dict:
        """Static rows a captured acceptance pass reads and writes: each request's pending hidden (staged
        by accepted_prepare), its verify rows and the row its accepted block ends on."""
        if self._fb is None:
            b, w, d = self.context.max_batch, self.head.hidden_width, self.context.device
            self._fb = {"pending": torch.zeros((b, w), dtype=torch.bfloat16, device=d),
                        "rows": torch.zeros((b, self.k + 1, w), dtype=torch.bfloat16, device=d),
                        "last": torch.zeros((b, w), dtype=torch.bfloat16, device=d)}
        return self._fb

    def accepted_prepare(self, requests):
        """Validate the batch and stage its pending hidden rows before a captured feedback pass.
        Returns (starts, ring slots): the committed prefix each block continues; MTP has no rings."""
        if not 1 <= len(requests) <= self.context.max_batch:
            raise ValueError("invalid MTP feedback batch")
        states = [self._state(req) for req in requests]
        if any(s.pending_token is None or s.pending_hidden is None or not s.valid_prefix or s.draft_pos != req.committed_len
               for s, req in zip(states, requests)):
            raise ValueError("MTP feedback does not follow the pending input")
        self._feedback_buffers()["pending"][:len(requests)].copy_(torch.cat([s.pending_hidden for s in states]))
        return [req.committed_len for req in requests], [0] * len(requests)

    def feedback_device(self, features, starts, ring_slots, counts, steps, *, norm=None, tokens=None, pages=None):
        """`_accepted` for every request of a bucket in one pass, on the device.

        Static shapes for graph capture: norm [b*steps, W] the verify rows, tokens and pages [b*steps] the
        verify block's inputs (the pending token, then the drafts) and pages, starts/counts [b] int64.
        Every row's draft KV is projected and rows past a request's count land on the dummy page 0.
        The projection sees all b*steps rows, the host path only a request's accepted rows, so with an
        FP8 draft layer (one dynamic activation scale per call) the committed KV matches the host's to
        FP8 rounding, not bitwise; drafts, and so round boundaries, can drift."""
        fb = self._feedback_buffers()
        b = starts.shape[0]
        rows = norm[:b * steps].view(b, steps, -1)
        hidden = torch.cat([fb["pending"][:b].unsqueeze(1), rows[:, :steps - 1]], dim=1).reshape(b * steps, -1)
        offsets = torch.arange(steps, device=starts.device)
        positions = (starts[:, None] + offsets[None, :]).reshape(-1)
        valid = (offsets[None, :] < counts[:, None]).reshape(-1)
        write = torch.where(valid, pages[:b * steps], torch.zeros_like(pages[:b * steps]))
        self.head.write_kv(tokens[:b * steps], hidden, positions, write)
        fb["rows"][:b, :steps].copy_(rows)
        fb["last"][:b].copy_(rows[torch.arange(b, device=starts.device), (counts - 1).clamp(min=0)])

    def accepted_publish(self, requests, emitted):
        """The host half of `_accepted` after the device feedback pass landed: pending input, checkpoints."""
        fb = self._feedback_buffers()
        last = fb["last"][:len(requests)].clone()  # the buffers belong to the next round's pass
        for i, (req, tokens) in enumerate(zip(requests, emitted)):
            state = self._state(req)
            count = len(tokens)
            if not 1 <= count <= self.k + 1 or state.draft_pos != req.committed_len - count:
                raise ValueError("MTP feedback does not follow the pending input")
            state.last_hidden = last[i]
            self._checkpoint(req, req.committed_len - count, req.committed_len, fb["rows"][i, :count])
            state.pending_token = None if req.finished else int(tokens[-1])
            state.pending_hidden = None if req.finished else last[i:i + 1]
            state.draft_pos = req.committed_len

    def release(self, req):
        req.proposer_state = None

    def prepare_release(self, req):
        pass  # features are retained at checkpoint creation/accepted commit

    def capture_graphs(self, log=None, depths=None):
        """One draft step's graphs serve every depth (`depths`, the verify graphs' pairs, needs nothing more)."""
        from sharar.proposers.mtp_graphs import DraftGraphs
        self.graphs = DraftGraphs(self)
        self.graphs.capture(log=log)
