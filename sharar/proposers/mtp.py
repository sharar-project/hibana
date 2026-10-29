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


def draft_argmax(head, hidden: torch.Tensor) -> torch.Tensor:
    """The head's draft token through its draft head (`proposers/draft_head.py`): the target's own head by default."""
    return head.draft_head.argmax(head.head_input(hidden))


def draft_candidates(head, hidden: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The draft head's r best tokens for each row: (ids int64 [rows, r], FP32 logits [rows, r]); with a 2-bit or 3-bit
    shortlist head, its candidates rescored by the target's FP8 rows (`draft_head.ShortlistHead.candidates`)."""
    return head.draft_head.candidates(head.head_input(hidden), r)


def draft_rows(requests, depth: int, width: int, tau: float, device):
    """One round's per-row inputs of `draft_laws` on CUDA, one upload each: params float32 [b, 4] (temperature x tau,
    top_k within the candidates or 0, top_p, greedy) and Philox rng int64 [depth, b, 2], step j at the request's
    (seed, ROUND_STRIDE * round + DRAFT + j) (engine/sampling_rng.py)."""
    from sharar.engine import sampling_rng
    params = sampling_rng.rows("mtp-draft-params", [[max(q.params.temperature, 1e-6) * tau, q.params.top_k if 0 < q.params.top_k < width else 0,
                                                     q.params.top_p, float(q.params.greedy)] for q in requests], torch.float32, device)
    base = sampling_rng.offsets(requests, sampling_rng.DRAFT)
    rng = sampling_rng.rows("mtp-draft-rng", [[q.rng_seed, o + j] for j in range(depth) for q, o in zip(requests, base)], torch.int64, device)
    return params, rng.view(depth, len(requests), 2)


def draft_laws(ids, logits, requests, tau: float, rows=None, j: int = 0):
    """Sampled MTP drafts for step j: each row draws from its request's law (temperature x tau, top-k with ties, top-p,
    as `rejection.distribution`) over its candidates, ids / logits [b, r]; greedy rows take the best candidate.
    Returns (tokens [b], laws [b, r]): the drafts and the realized law q each was drawn from, which the verifier tests
    them against exactly. CUDA (`rows` from `draft_rows`): one Philox draw per row, kernels/draft_selection.sampled_lattice
    (DFlash2's sampled walk) with one step; else each request's generator."""
    b, r = ids.shape
    if ids.is_cuda:
        from sharar.kernels.draft_selection import sampled_lattice
        params, rng = rows
        tokens, laws = sampled_lattice(ids.view(b, 1, r), logits.float().contiguous().view(b, 1, 1, r).expand(b, 1, r, r), params, rng[j])
        return tokens[:, 0], laws[:, 0]
    from sharar.engine.rejection import small_laws
    laws = small_laws(logits.float() / tau, [q.params for q in requests])
    u = torch.stack([torch.rand(1, device=ids.device, generator=q.generator) for q in requests])  # [b, 1]
    cdf = laws.cumsum(-1)
    local = torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous(), right=True).clamp_max(r - 1)
    return ids.gather(1, local).squeeze(1), laws


def draft_choice(head, hidden: torch.Tensor, with_confidence: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`draft_argmax`, and with `with_confidence` the draft head's [rows, 2] confidence in it (SHARAR_TRACE_CONFIDENCE)."""
    if not with_confidence:
        return draft_argmax(head, hidden), None
    return head.draft_head.argmax_confidence(head.head_input(hidden))


def draft_outputs(head, hidden: torch.Tensor, with_confidence: bool, width: int):
    """`draft_choice`, and with SHARAR_MTP_SAMPLE (`width` > 0) the draft head's candidates from the same head pass:
    the draft is then their first, the head's best (its argmax, ties aside; a shortlist head ranks max(rescore, width)
    candidates by the FP8 rows), unless the confidence trace reads the head's own logits. Returns (token, confidence,
    (ids, logits) [rows, width] or None)."""
    if not width:
        return (*draft_choice(head, hidden, with_confidence), None)
    ids, values = draft_candidates(head, hidden, width)
    token, conf = draft_choice(head, hidden, True) if with_confidence else (ids[..., 0], None)
    return token, conf, (ids, values)


class MtpProposer:
    def __init__(self, head, context: ProposerContext, k: int = 3):
        self.head, self.context, self.k = head, context, k
        # Greedy batches draft the head's argmax: deterministic proposals, verified as point masses (engine/rejection.py).
        # With SHARAR_MTP_SAMPLE a batch with a sampled request draws each draft from the request's law over the head's
        # rescored candidates (`draft_laws`) and hands that law to the verifier, which accepts against it exactly: the
        # acceptance rate rises from p(argmax) to sum(min(p, q)) with the emitted law unchanged.
        self.requirements = ProposerRequirements("mtp", FeatureRequest(final_norm=True), persistent_bytes=head.bytes,
                                                 greedy_only=False, max_depth=8,
                                                 capabilities=Capabilities(device_feedback=True, draft_head=getattr(head, "DRAFT_HEADS", False),
                                                                           short_drafts=True, stored_payload=True))
        self.graphs = None
        # The draft's confidence per step for the trace round records (SHARAR_TRACE_CONFIDENCE); read once
        # here, the draft graphs capture the extra output.
        self.with_confidence = current().trace_confidence
        cfg = current()
        self.sample = cfg.mtp_sample  # read once here: the draft graphs capture the candidates' outputs
        self.sample_width = cfg.mtp_sample_candidates
        self.draft_temperature = cfg.mtp_draft_temperature
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
            # The snapshot the request resumed from: its lock keeps it (and this row) until this chunk is ingested.
            snap = req.match.snapshot if req.match is not None else None
            cached = snap.auxiliary.get(self.cache_identity) if snap is not None else None
            if cached is not None and cached[0] == start:
                state.last_hidden, state.draft_pos = cached[1], start
                self.cache_hits += 1
            else:
                # No row for the resume position (a snapshot from an SSD store written without it): the chunk's own
                # first row stands in for it, so one draft KV row is approximate and the request keeps drafting.
                state.last_hidden, state.draft_pos = norm_rows[0], start
                self.cache_fallbacks += 1
        first = max(1, start)
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
        """The row each of this prefill forward's snapshots (the request's private ones, in (start, end]) resumes the
        draft from: the final-norm row of the position before it."""
        for position, snap in getattr(req, "private", {}).items():
            if start < position <= end:
                snap.auxiliary[self.cache_identity] = (position, norm_rows[position - start - 1].clone())

    def start(self, req, token):
        state = self._state(req)
        state.pending_token = token
        state.pending_hidden = state.last_hidden.unsqueeze(0)

    def ready(self, req):
        return isinstance(req.proposer_state, MtpState) and req.proposer_state.pending_token is not None

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
        sampled = self.sample and any(not r.params.greedy for r in requests)
        laws, supports = [], []
        if sampled and self.context.device.type == "cuda":
            self._rows = draft_rows(requests, self.k if depth is None else depth, self.sample_width, self.draft_temperature, self.context.device)
        hidden, token, conf, cand = self._step(inputs.tokens, torch.cat(hids), inputs.positions, pages[:, 0].contiguous(), kv, graphs, sampled)
        if sampled:
            token = self._draw(cand, requests, 0, laws, supports)
        drafts, confs = [token], [conf]
        self._column(0, token)
        for j in range(2, (self.k if depth is None else depth) + 1):
            kv = [torch.cat([r.pages[1:r.context_len - 1], pages[i, :j]]) for i, r in enumerate(requests)]
            hidden, token, conf, cand = self._step(drafts[-1], hidden, inputs.positions + j - 1, pages[:, j - 1].contiguous(), kv, graphs, sampled)
            if sampled:
                token = self._draw(cand, requests, j - 1, laws, supports)
            drafts.append(token)
            confs.append(conf)
            self._column(j - 1, token)
        columns = DraftColumns(self._host_drafts, self._draft_events, len(requests)) if self._draft_events is not None else None
        confidence = torch.stack(confs, dim=1) if self.with_confidence else None
        if sampled:
            return Proposal(torch.stack(drafts, dim=1), torch.stack(laws, dim=1), torch.stack(supports, dim=1), confidence=confidence, columns=columns)
        return Proposal(torch.stack(drafts, dim=1), confidence=confidence, columns=columns)

    def _draw(self, cand, requests, j, laws, supports):
        """Step j's sampled draft from its candidates, its law and support appended for the verifier."""
        ids, values = cand
        token, law = draft_laws(ids, values, requests, self.draft_temperature, getattr(self, "_rows", None), j)
        laws.append(law)
        supports.append(ids)
        return token

    def _column(self, j: int, token: torch.Tensor) -> None:
        if self._draft_events is not None:
            self._host_drafts[j, :token.shape[0]].copy_(token, non_blocking=True)
            self._draft_events[j].record()

    def _step(self, tokens, hidden, positions, pages, kv, graphs, sampled=False):
        """One draft step: (hidden, argmax, confidence, candidates). Candidates (ids, logits) [b, r] only for a sampled
        batch, else None."""
        cand = None
        if graphs and self.graphs is not None:
            hidden, token, conf = self.graphs.run(tokens, hidden, positions, pages, kv)
            hidden, token, conf = hidden.clone(), token.clone(), conf.clone() if conf is not None else None  # graph outputs are borrowed until next replay
            if sampled:
                cand = tuple(t.clone() for t in self.graphs.candidates(tokens.shape[0]))
        else:
            hidden = self.head.forward(tokens, hidden, positions, pages, [1] * len(kv), kv)
            token, conf, cand = draft_outputs(self.head, hidden, self.with_confidence, self.sample_width if self.sample else 0)
            cand = cand if sampled else None
        return hidden, token, conf, cand

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
                    or state.pending_hidden is None
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
        if any(s.pending_token is None or s.pending_hidden is None or s.draft_pos != req.committed_len
               for s, req in zip(states, requests)):
            raise ValueError("MTP feedback does not follow the pending input")
        self._feedback_buffers()["pending"][:len(requests)].copy_(torch.cat([s.pending_hidden for s in states]))
        return [req.committed_len for req in requests], [0] * len(requests)

    def feedback_device(self, features, starts, ring_slots, counts, steps, *, norm=None, tokens=None, pages=None):
        """`_accepted` for every request of a bucket in one pass, on the device.

        Static shapes for graph capture: norm [b*steps, W] the verify rows, tokens and pages [b*steps] the
        verify block's inputs (the pending token, then the drafts) and pages, starts/counts [b] int64.
        Every row's draft KV is projected and rows past a request's count land on the draft cache's padding
        pages, one each: written together to page 0, a duplicate-index write kept an arbitrary row, the padded
        draft rows that attend to page 0 changed from run to run and Flash-Next's per-tensor activation scale
        carried it into the real rows' drafts. The projection sees all b*steps rows, the host path only a
        request's accepted rows, so with an FP8 draft layer (one dynamic activation scale per call) the committed
        KV matches the host's to FP8 rounding, not bitwise; drafts, and so round boundaries, can drift."""
        fb = self._feedback_buffers()
        b = starts.shape[0]
        rows = norm[:b * steps].view(b, steps, -1)
        hidden = torch.cat([fb["pending"][:b].unsqueeze(1), rows[:, :steps - 1]], dim=1).reshape(b * steps, -1)
        offsets = torch.arange(steps, device=starts.device)
        positions = (starts[:, None] + offsets[None, :]).reshape(-1)
        valid = (offsets[None, :] < counts[:, None]).reshape(-1)
        spare = self.head.kv.padding_pages
        dummy = spare[:b * steps].to(pages.dtype) if spare.numel() >= b * steps else torch.zeros_like(pages[:b * steps])
        write = torch.where(valid, pages[:b * steps], dummy)
        self.head.write_kv(tokens[:b * steps], hidden, positions, write)
        fb["rows"][:b, :steps].copy_(rows)
        fb["last"][:b].copy_(rows[torch.arange(b, device=starts.device), (counts - 1).clamp(min=0)])

    def accepted_publish(self, requests, emitted):
        """The host half of `_accepted` after the device feedback pass landed: the pending input."""
        fb = self._feedback_buffers()
        last = fb["last"][:len(requests)].clone()  # the buffers belong to the next round's pass
        for i, (req, tokens) in enumerate(zip(requests, emitted)):
            state = self._state(req)
            count = len(tokens)
            if not 1 <= count <= self.k + 1 or state.draft_pos != req.committed_len - count:
                raise ValueError("MTP feedback does not follow the pending input")
            state.last_hidden = last[i]
            state.pending_token = None if req.finished else int(tokens[-1])
            state.pending_hidden = None if req.finished else last[i:i + 1]
            state.draft_pos = req.committed_len

    def release(self, req):
        req.proposer_state = None

    def prepare_release(self, req, snapshot):
        """The row the reply end (`snapshot`, the request's own slot at its committed length) resumes the draft from:
        the final-norm row of the position before it, the last one the request committed."""
        state = req.proposer_state
        if isinstance(state, MtpState) and state.last_hidden is not None and state.draft_pos == snapshot.position:
            # its own copy: last_hidden can be a row of a round's [batch, hidden] buffer, which the snapshot would pin
            snapshot.auxiliary[self.cache_identity] = (snapshot.position, state.last_hidden.clone())

    def capture_graphs(self, log=None, depths=None):
        """One draft step's graphs serve every depth (`depths`, the verify graphs' pairs, needs nothing more)."""
        from sharar.proposers.mtp_graphs import DraftGraphs
        self.graphs = DraftGraphs(self)
        self.graphs.capture(log=log)
