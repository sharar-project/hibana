"""Speculative decoding with the checkpoint's MTP head (chain drafts, greedy accept).

One round for a batch of greedy requests, each with context length L and an
unprocessed last token x at position P = L - 1:
  1. draft: ingest the pending (token, target hidden) pairs the draft has not
     seen (1 or 2 rows), then chain k-1 single-token steps feeding its own
     hidden back, producing y_1..y_k;
  2. verify: one target forward over [x, y_1..y_k] per request (attention over
     the prefix plus k+1 fresh pages; GDN from the committed state without
     updating it, recording the state after every token);
  3. accept: a = number of leading drafts whose target argmax matches; emit
     y_1..y_a and the target's bonus token z_a; commit the state after a+1
     tokens (intermediate SSM state a, conv window a), keep pages 0..a, free
     the rest; queue the rows the draft still has to ingest for the next
     round.
Everything is exact with respect to the target's greedy choice at each
position: a draft is accepted only where the target itself would have
produced it.
"""

from __future__ import annotations

import time

import torch

from sharar import config
from sharar.memory.state_pool import CONV_DIM, CONV_STATE_LEN
from sharar.engine.context import ForwardContext
from sharar.engine.request import Request
from sharar.memory.kv_pool import KV_DTYPE
from sharar.model.mtp import MtpHead


class SpecDecoder:
    def __init__(self, executor, mtp: MtpHead, k: int = 3):
        self.ex = executor
        self.mtp = mtp
        self.k = k
        self.steps = k + 1
        self.device = executor.device
        n = executor.max_batch
        hv, vd, kd = config.LINEAR_NUM_VALUE_HEADS, config.LINEAR_VALUE_HEAD_DIM, config.LINEAR_KEY_HEAD_DIM
        layer_ids = executor.state_pool.layer_ids
        # Verify outputs, stacked over the GDN layers like the state pool so the
        # accept step commits a request's state with one strided copy per kind.
        self.inter_ssm_all = torch.zeros((len(layer_ids), n, self.steps, hv, vd, kd), dtype=torch.float32, device=self.device)
        self.inter_ssm = {lid: self.inter_ssm_all[j] for j, lid in enumerate(layer_ids)}
        self.conv_windows_all = torch.zeros((len(layer_ids), n, self.steps, CONV_DIM, CONV_STATE_LEN), dtype=torch.bfloat16, device=self.device)
        self.conv_windows = {lid: self.conv_windows_all[j] for j, lid in enumerate(layer_ids)}
        self.verify_graphs = None
        self.draft_graphs = None
        # Set by benchmarks: synchronize around the draft/verify/accept phases
        # to attribute time. Off in serving so the host can queue graphs ahead.
        self.timing = False
        # Pinned staging for the per-round host -> device upload: rows are
        # pending token, its position, state slot; one non-blocking copy.
        self._host_rows = torch.empty((3, n), dtype=torch.int64).pin_memory()
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0
        self.accept_seconds = 0.0
        self.emitted = 0
        self.draft_seconds = 0.0
        self.verify_seconds = 0.0

    def capture_graphs(self, log=None) -> None:
        from sharar.engine.spec_graphs import DraftGraphs, VerifyGraphs

        self.verify_graphs = VerifyGraphs(self)
        self.verify_graphs.capture(log=log)
        self.draft_graphs = DraftGraphs(self)
        self.draft_graphs.capture(log=log)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * 4 for t in self.inter_ssm.values())

    def reset_stats(self) -> None:
        self.rounds = self.proposed = self.accepted = self.emitted = 0
        self.draft_seconds = self.verify_seconds = self.accept_seconds = 0.0

    def stats(self) -> dict:
        return {
            "rounds": self.rounds, "proposed": self.proposed, "accepted": self.accepted, "emitted": self.emitted,
            "accept_rate": self.accepted / max(1, self.proposed), "tokens_per_round": self.emitted / max(1, self.rounds),
            "draft_ms_per_round": 1000 * self.draft_seconds / max(1, self.rounds),
            "verify_ms_per_round": 1000 * self.verify_seconds / max(1, self.rounds),
            "accept_ms_per_round": 1000 * self.accept_seconds / max(1, self.rounds),
        }

    # ---- prefill hooks ----
    def ingest_prefill(self, req: Request, new_ids: list[int], start: int, norm_rows: torch.Tensor) -> None:
        """After a target prefill chunk covering absolute positions
        [start, start+len(new_ids)), feed the draft every position i >= 1 in
        the chunk with (x_i, h_{i-1}). h_{start-1} comes from the previous
        chunk; position 0 has no predecessor and is skipped; a cached prefix
        is skipped too (its hidden states are gone), which only lowers draft
        quality for that request."""
        n = len(new_ids)
        end = start + n
        first = max(1, start)
        if start > 0 and getattr(req, "spec_last_hidden", None) is None:
            first = start + 1  # no hidden for position start-1 (cache hit)
        if first < end:
            toks = torch.tensor(new_ids[first - start :], dtype=torch.int64, device=self.device)
            if first == start:
                hid = torch.cat([req.spec_last_hidden.unsqueeze(0), norm_rows[: n - 1]], dim=0)
            else:
                hid = norm_rows[first - start - 1 : n - 1]
            pos = torch.arange(first, end, dtype=torch.int64, device=self.device)
            pages = req.pages[first:end]
            kv_pages = req.pages[1:end] if first == 1 or getattr(req, "draft_pos", 0) >= first else req.pages[first:end]
            self.mtp.forward(toks, hid, pos, pages, [end - first], [kv_pages])
            req.draft_pos = end  # next position the draft must ingest
        else:
            req.draft_pos = max(getattr(req, "draft_pos", 0), first)
        req.spec_last_hidden = norm_rows[-1]

    def start(self, req: Request, token: int) -> None:
        """Called once the prompt is fully prefilled and its first token sampled."""
        req.spec_pending_tokens = [token]
        req.spec_pending_hidden = req.spec_last_hidden.unsqueeze(0)

    def ready(self, req: Request) -> bool:
        return bool(getattr(req, "spec_pending_tokens", None))

    # ---- one round ----
    @torch.inference_mode()
    def round(self, reqs: list[Request]) -> None:
        ex = self.ex
        k, s_len = self.k, self.steps
        b_sz = len(reqs)
        d = self.device
        timing = self.timing
        if timing:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        new_pages_flat, new_ids = ex.kv_pool.alloc_with_ids(b_sz * s_len)
        new_pages = new_pages_flat.view(b_sz, s_len)
        # ---- draft step 1: ingest the pending row of every request ----
        # After a round each request has exactly one pending row (the last
        # emitted token with the target's hidden at that position), so this
        # step is the same single-row draft as steps 2..k and uses the same
        # CUDA graph. The eager path remains for graphs disabled.
        hids, kv_pages = [], []
        host = self._host_rows
        for i, r in enumerate(reqs):
            P = r.context_len - 1
            pend = r.spec_pending_tokens
            assert len(pend) == 1 and P == r.draft_pos and pend[0] == r.output_ids[-1], (pend, P, r.draft_pos)
            host[0, i] = pend[0]
            host[1, i] = P
            host[2, i] = r.slot
            hids.append(r.spec_pending_hidden)
            kv_pages.append(torch.cat([r.pages[1:P], new_pages[i, :1]]))
        rows = host[:, :b_sz].to(d, non_blocking=True)  # the round's only upload
        toks_t, poss_t, slots_t = rows[0], rows[1], rows[2].to(torch.int32)
        if self.draft_graphs is not None:
            d_last, nxt = self.draft_graphs.run(toks_t, torch.cat(hids), poss_t, new_pages[:, 0].contiguous(), kv_pages)
            d_last = d_last.clone()
            drafts = [nxt.clone()]
        else:
            dh = self.mtp.forward(toks_t, torch.cat(hids), poss_t, new_pages[:, 0].contiguous(), [1] * b_sz, kv_pages)
            d_last = dh
            drafts = [self.mtp.logits(d_last).argmax(-1)]  # y_1 for each request
        # ---- draft steps 2..k ----
        for j in range(2, k + 1):
            pos_j = poss_t + (j - 1)
            kv = [torch.cat([r.pages[1 : r.context_len - 1], new_pages[i, :j]]) for i, r in enumerate(reqs)]
            if self.draft_graphs is not None:
                d_last, nxt = self.draft_graphs.run(drafts[-1], d_last, pos_j, new_pages[:, j - 1].contiguous(), kv)
                d_last = d_last.clone()
                drafts.append(nxt.clone())
            else:
                d_last = self.mtp.forward(drafts[-1], d_last, pos_j, new_pages[:, j - 1].contiguous(), [1] * b_sz, kv)
                drafts.append(self.mtp.logits(d_last).argmax(-1))
        y = torch.stack(drafts, dim=1)  # [B, k]
        if timing:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        # ---- verify ----
        # The verified block starts with the last emitted token, which is the
        # pending token uploaded above.
        tokens_v = torch.cat([toks_t.unsqueeze(1), y], dim=1).reshape(-1)
        positions = torch.cat([torch.arange(r.context_len - 1, r.context_len - 1 + s_len, device=d) for r in reqs])
        kv_indptr = [0]
        kv_all = []
        for i, r in enumerate(reqs):
            kv_all.append(torch.cat([r.pages, new_pages[i]]))
            kv_indptr.append(kv_indptr[-1] + kv_all[-1].shape[0])
        if self.verify_graphs is not None:
            norm_rows, z_flat, ctx = self.verify_graphs.run(tokens_v, positions, new_pages_flat, slots_t, kv_all)
            norm = norm_rows.view(b_sz, s_len, -1).clone()
            z = z_flat.view(b_sz, s_len).clone()
            self._accept(reqs, y, z, norm, ctx, new_pages, new_ids, t0, t1)
            return
        cu = torch.arange(0, b_sz * s_len + 1, s_len, dtype=torch.int32, device=d)
        ctx = ForwardContext(
            mode="verify", kv_pool=ex.kv_pool, state_pool=ex.state_pool, positions=positions,
            pages=new_pages_flat, seq_len=b_sz * s_len,
            cu_seqlens=cu, seq_slots=slots_t,
            seq_prefix_lens=[r.context_len - 1 for r in reqs], seq_lens=[s_len] * b_sz,
            verify_steps=s_len, verify_scratch=torch.arange(b_sz, dtype=torch.int32, device=d), inter_ssm=self.inter_ssm,
            verify_conv_windows=self.conv_windows,
        )
        ex._paged_prefill.plan(
            torch.arange(0, b_sz * s_len + 1, s_len, dtype=torch.int32), torch.tensor(kv_indptr, dtype=torch.int32), torch.cat(kv_all), torch.ones(b_sz, dtype=torch.int32),
            config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM, 1,
            causal=True, q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=config.HEAD_DIM**-0.5,
        )
        ctx.paged_prefill_wrapper = ex._paged_prefill
        out = ex.model.forward(tokens_v, ctx)
        norm = out["norm"].view(b_sz, s_len, -1)
        z = ex.model.logits(out["norm"]).argmax(-1).view(b_sz, s_len)  # target choice after each token
        self._accept(reqs, y, z, norm, ctx, new_pages, new_ids, t0, t1)

    def _accept(self, reqs, y, z, norm, ctx, new_pages, new_ids, t0, t1) -> None:
        if self.timing:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        ex = self.ex
        k, s_len = self.k, self.steps
        # ---- accept ----
        match = (y == z[:, :k])  # draft j matches target choice after token j-1
        acc_t = torch.cumprod(match.int(), dim=1).sum(dim=1)
        # One host transfer for drafts, target choices and accepted counts.
        packed = torch.cat([y, z, acc_t.unsqueeze(1).to(y.dtype)], dim=1).tolist()
        y_l = [row[:k] for row in packed]
        z_l = [row[k : 2 * k + 1] for row in packed]
        acc = [row[-1] for row in packed]
        for i, r in enumerate(reqs):
            a = acc[i]
            # Commit the state after the accepted prefix: all GDN layers at once.
            ex.state_pool.ssm_all[:, r.slot].copy_(self.inter_ssm_all[:, i, a])
            ex.state_pool.conv_all[:, r.slot].copy_(self.conv_windows_all[:, i, a])
            r.pages = torch.cat([r.pages, new_pages[i, : a + 1]])
            if a + 1 < s_len:
                ex.kv_pool.free_ids(new_ids[i * s_len + a + 1 : (i + 1) * s_len])
            emitted = y_l[i][:a] + [z_l[i][a]]
            for tok in emitted:
                if r.finished:
                    break
                r.push(int(tok))
            self.proposed += k
            self.accepted += a
            self.emitted += len(emitted)
            r.spec_rounds += 1
            r.spec_proposed += k
            r.spec_accepted += a
            # Next round drafts from the last emitted token and the target's
            # hidden state at that position. (Re-ingesting every accepted
            # position with target hiddens, EAGLE's "draft extend", was tried
            # and measured no acceptance gain here, so the draft KV written
            # from the chain hiddens is kept.)
            P = r.context_len - len(emitted) - 1  # position of x_last this round
            r.spec_pending_tokens = [int(emitted[a])]
            r.spec_pending_hidden = norm[i, a : a + 1]
            r.draft_pos = P + a + 1
        self.rounds += 1
        if self.timing:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        self.draft_seconds += t1 - t0
        self.verify_seconds += t2 - t1
        self.accept_seconds += t3 - t2
