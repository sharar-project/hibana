"""Proposer-independent target verification and stop-aware acceptance.

Draft adapters supply candidates. StateManager owns reservations and commits;
this driver owns target verify buffers, graph replay and acceptance metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from sharar.engine.context import ForwardContext
from sharar.engine.request import Request
from sharar.memory.kv_pool import KV_DTYPE
from sharar.engine.contracts import Proposer, StepInputs, VerifiedFeatures
from sharar.engine.rejection import Proposal, distribution, plausible, verify_rows
from sharar.engine.accept_graphs import AcceptGraphs, ENABLED as DEVICE_ACCEPT


@dataclass
class Verification:
    requests: list[Request]
    proposal: Proposal
    sampled: bool
    tokens: torch.Tensor
    positions: torch.Tensor
    pages: torch.Tensor
    slots: torch.Tensor
    kv_pages: list[torch.Tensor]
    started: float
    prepared: float


class SpecDecoder:
    aux_windows: dict = {}  # class attribute: test decoders skip __init__

    def __init__(self, executor, proposer: Proposer, state_strategy: str = "snapshots"):
        if state_strategy not in ("snapshots", "reconstruct"):
            raise ValueError("unknown verify state strategy")
        self.ex = executor
        self.proposer = proposer
        self.k = proposer.k
        self.steps = proposer.k + 1
        self.verify_depth = self.k
        self.verify_depths = (self.k,)  # benchmarks may explicitly capture narrower widths
        self.width_graphs = {}
        self.device = executor.device
        n = executor.max_batch
        ssm_shape = executor.state_pool.ssm_all.shape[2:]
        conv_shape = executor.state_pool.conv_all.shape[2:]
        layer_ids = executor.state_pool.layer_ids
        # Verify outputs, stacked over the GDN layers like the state pool so the
        # accept step commits a request's state with one strided copy per kind.
        self.state_strategy = state_strategy
        self.replay = None
        if state_strategy == "reconstruct":
            from sharar.kernels.fla.accepted_state import AcceptedState
            self.replay = AcceptedState(executor.state_pool, n, self.steps, executor.target.linear_num_key_heads)
            self.inter_ssm_all, self.inter_ssm = None, {}
        else:
            self.inter_ssm_all = torch.zeros((len(layer_ids), n, self.steps, *ssm_shape), dtype=torch.float32, device=self.device)
            self.inter_ssm = {lid: self.inter_ssm_all[j] for j, lid in enumerate(layer_ids)}
        self.conv_windows_all = torch.zeros((len(layer_ids), n, self.steps, *conv_shape), dtype=torch.bfloat16, device=self.device)
        self.conv_windows = {lid: self.conv_windows_all[j] for j, lid in enumerate(layer_ids)}
        # Auxiliary per-slot state (StatePool.aux) after each verify step, same lifecycle as the conv windows.
        self.aux_windows = {name: torch.zeros((n, self.steps, *t.shape[1:]), dtype=t.dtype, device=self.device)
                            for name, t in getattr(executor.state_pool, "aux", {}).items()}
        self.verify_graphs = None
        # Device-resident acceptance for all-greedy graph batches (engine/accept_graphs.py);
        # SHARAR_DEVICE_ACCEPT=0 keeps the host path for every round.
        self.device_accept = DEVICE_ACCEPT
        # Set by benchmarks: synchronize around the draft/verify/accept phases
        # to attribute time. Off in serving so the host can queue graphs ahead.
        self.timing = False
        # Pinned staging for the per-round host -> device upload: rows are
        # pending token, its position, state slot; one non-blocking copy.
        self._host_rows = torch.empty((3, n), dtype=torch.int64, pin_memory=self.device.type == "cuda")
        # Per-row relaxed-acceptance thresholds (top-k, logit margin), uploaded only when a request asks for them.
        self._host_relax = torch.empty((2, n), dtype=torch.float32, pin_memory=self.device.type == "cuda")
        self.relaxed_accepts = 0  # emitted tokens accepted by the relaxed rule (not the argmax)
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0
        self.accept_seconds = 0.0
        self.accept_decide_seconds = self.accept_commit_seconds = self.accept_feedback_seconds = 0.0  # timing=True only
        self.emitted = 0
        self.draft_seconds = 0.0
        self.verify_seconds = 0.0

    def capture_graphs(self, log=None) -> None:
        from sharar.engine.spec_graphs import VerifyGraphs

        for depth in self.verify_depths:
            graphs = VerifyGraphs(self, depth)
            graphs.capture(log=log)
            self.width_graphs[depth] = graphs
        self.verify_graphs = self.width_graphs[self.k]
        self.proposer.capture_graphs(log=log)
        if AcceptGraphs.supported(self):  # captured whenever possible so the path can be switched at run time
            for graphs in self.width_graphs.values():
                graphs.accept = AcceptGraphs(self, graphs)
                graphs.accept.capture(log=log)

    @property
    def bytes(self) -> int:
        return (sum(t.numel() * t.element_size() for t in (self.inter_ssm_all, self.conv_windows_all, *self.aux_windows.values()) if t is not None)
                + (self.replay.bytes if self.replay is not None else 0))

    def reset_stats(self) -> None:
        self.rounds = self.proposed = self.accepted = self.emitted = self.relaxed_accepts = 0
        self.draft_seconds = self.verify_seconds = self.accept_seconds = 0.0
        self.accept_decide_seconds = self.accept_commit_seconds = self.accept_feedback_seconds = 0.0

    def stats(self) -> dict:
        r = max(1, self.rounds)
        out = {
            "rounds": self.rounds, "proposed": self.proposed, "accepted": self.accepted, "emitted": self.emitted,
            "accept_rate": self.accepted / max(1, self.proposed), "tokens_per_round": self.emitted / r,
            "state_strategy": self.state_strategy, "verify_buffer_bytes": self.bytes,
            "verify_depth": self.verify_depth, "verify_depths": self.verify_depths,
            "round_ms": 1000 * (self.draft_seconds + self.verify_seconds + self.accept_seconds) / r,
            "relaxed_accepts": self.relaxed_accepts,
        }
        if self.timing:  # phase split is only meaningful with the synchronizing timers on
            out.update({
                "draft_ms_per_round": 1000 * self.draft_seconds / r,
                "verify_ms_per_round": 1000 * self.verify_seconds / r,
                "accept_ms_per_round": 1000 * self.accept_seconds / r,
                "accept_split_ms": [round(1000 * x / r, 1) for x in (self.accept_decide_seconds, self.accept_commit_seconds, self.accept_feedback_seconds)],
            })
        return out

    def ingest_prefill(self, req, ids, start, result) -> None:
        self.proposer.ingest_prefill(req, ids, start, result)

    def start(self, req, token) -> None:
        self.proposer.start(req, token)
        self._trace_features(req, req.committed_len - min(req.committed_len, getattr(self.proposer, "window", 0)), req.committed_len)

    def _trace_features(self, req, start: int, end: int) -> None:
        """Copy the projected context rows the draft just received into the request's trace recorder."""
        recorder = req.trace_features
        if recorder is None or end <= start:
            return
        rows_for = getattr(self.proposer, "projected_rows", None)
        if rows_for is None or not self.proposer.ready(req) and not req.finished:
            return
        try:
            recorder.append(rows_for(req, start, end - start), start)
        except ValueError:
            recorder.valid = False

    def ready(self, req) -> bool:
        return self.proposer.ready(req)

    # ---- one round ----
    @torch.inference_mode()
    def round(self, reqs: list[Request]) -> None:
        batch = self.prepare(reqs)
        self.verify(batch)

    def prepare(self, reqs: list[Request]) -> Verification:
        """Reserve and draft one target verification batch."""
        ex = self.ex
        k = getattr(self, 'verify_depth', self.k)
        if k not in getattr(self, 'verify_depths', (self.k,)):
            raise ValueError('uncaptured verification depth')
        s_len = k + 1
        b_sz = len(reqs)
        d = self.device
        timing = self.timing
        if timing:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        new_pages_flat = ex.state.reserve(reqs, self.steps)
        new_pages = new_pages_flat.view(b_sz, self.steps)
        host = self._host_rows
        for i, r in enumerate(reqs):
            if r.committed_len != r.context_len - 1:
                raise ValueError("pending input does not follow the committed prefix")
            host[0, i] = r.output_ids[-1]
            host[1, i] = r.committed_len
            host[2, i] = r.slot
        rows = host[:, :b_sz].to(d, non_blocking=True)
        toks_t, poss_t, slots_t = rows[0], rows[1], rows[2].to(torch.int32)
        inputs = StepInputs(toks_t, poss_t, slots_t)
        proposed = self.proposer.propose(reqs, new_pages, inputs, graphs=ex.use_graphs)
        proposal = proposed if isinstance(proposed, Proposal) else Proposal(proposed)
        y = proposal.tokens
        sampled = any(not r.params.greedy for r in reqs)
        if (y.shape != (b_sz, self.k) or y.dtype != torch.int64 or y.device.type != self.device.type
                or (self.device.index is not None and y.device.index != self.device.index)):
            raise ValueError("proposer must return int64 [batch, k] on the target device")
        y = y[:, :k]
        proposal = Proposal(y, proposal.probabilities[:, :k] if proposal.probabilities is not None else None,
                            proposal.support[:, :k] if proposal.support is not None else None)
        new_pages_flat = new_pages[:, :s_len].reshape(-1)
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
            kv_all.append(ex.state.step_pages(r)[:r.committed_len + s_len])
            kv_indptr.append(kv_indptr[-1] + kv_all[-1].shape[0])
        return Verification(list(reqs), proposal, sampled, tokens_v, positions, new_pages_flat,
                            slots_t, kv_all, t0, t1)

    def verify(self, batch: Verification) -> None:
        b_sz, k = batch.proposal.tokens.shape
        s_len = k + 1
        if self.ex.use_graphs and self.verify_graphs is not None:
            graphs = self.width_graphs.get(k, self.verify_graphs)
            ple_rows = self.ex._ple_rows_block(batch.requests, batch.proposal.tokens.tolist()) if getattr(self.ex, "_ple", None) is not None else None
            norm_rows, z_flat, ctx, layers, logits = graphs.run(batch.tokens, batch.positions, batch.pages,
                                                               batch.slots, batch.kv_pages, sampled=batch.sampled, ple_rows=ple_rows)
            accept = getattr(graphs, "accept", None)
            if accept is not None and not batch.sampled and self.device_accept and accept.eligible(batch.requests):
                if self.timing:
                    torch.cuda.synchronize()
                accept.round(batch, batch.proposal.tokens, self._target_topk(batch.requests, logits, b_sz, s_len),
                             batch.started, batch.prepared)
                return
            norm = norm_rows.view(b_sz, s_len, -1).clone() if self.proposer.requirements.features.final_norm else None
            layers = {lid: value.view(b_sz, s_len, -1).clone() for lid, value in layers.items()} if layers else None
            z = z_flat.view(b_sz, s_len, -1) if batch.sampled else z_flat.view(b_sz, s_len).clone()
            relaxed = None if batch.sampled else self._relaxed_matches(batch.requests, logits, batch.proposal.tokens, b_sz, s_len)
            self._accept_batch(batch, z, VerifiedFeatures(norm, layers), topk=self._target_topk(batch.requests, logits, b_sz, s_len), relaxed=relaxed)
            return
        ctx = self.context(batch, self.ex._paged_prefill)
        self.finish(batch, self.ex.model.forward(batch.tokens, ctx))

    def _relaxed_matches(self, reqs, logits, y, b_sz: int, s_len: int):
        """[b, k] bool: drafted token within the request's top-k and logit margin of the target's argmax.

        None when no request in the batch asked for relaxed acceptance, so the
        default path is untouched. Rows of exact requests get top-k 1 and margin 0,
        which reduces to the argmax test."""
        if not any(r.params.relaxed for r in reqs):
            return None
        k = y.shape[1]
        host = self._host_relax
        for i, r in enumerate(reqs):
            host[0, i] = float(r.params.accept_top_k)
            host[1, i] = float(r.params.accept_margin)
        thresholds = host[:, :b_sz].to(self.device, non_blocking=True)
        rows = logits.view(b_sz, s_len, -1)[:, :k]  # row j predicts drafted token y_j
        ly = rows.gather(-1, y.unsqueeze(-1)).squeeze(-1).float()
        lmax = rows.max(-1).values.float()
        rank = (rows > ly.unsqueeze(-1)).sum(-1).float()
        return (rank < thresholds[0].unsqueeze(1)) & ((lmax - ly) <= thresholds[1].unsqueeze(1))

    TOPK = 16

    def _target_topk(self, reqs, logits, b_sz: int, s_len: int):
        """Top-k target logits per verified row, only when a request in the batch records traces."""
        if not any(r.trace_topk is not None for r in reqs):
            return None
        values, ids = logits.view(b_sz, s_len, -1).topk(self.TOPK, dim=-1)
        return ids.to(torch.int32), values.to(torch.bfloat16)

    def context(self, batch: Verification, wrapper) -> ForwardContext:
        """Plan the eager verification attention wrapper."""
        b_sz, k = batch.proposal.tokens.shape
        s_len, d = k + 1, self.device
        cu_host = torch.arange(0, b_sz * s_len + 1, s_len, dtype=torch.int32)
        ctx = ForwardContext(
            feature_request=self.proposer.requirements.features,
            mode="verify", kv_pool=self.ex.kv_pool, state_pool=self.ex.state_pool, positions=batch.positions,
            pages=batch.pages, seq_len=b_sz * s_len,
            cu_seqlens=cu_host.to(d), seq_slots=batch.slots,
            seq_prefix_lens=[r.context_len - 1 for r in batch.requests], seq_lens=[s_len] * b_sz,
            verify_steps=s_len, verify_scratch=torch.arange(b_sz, dtype=torch.int32, device=d), inter_ssm=self.inter_ssm,
            verify_replay=self.replay.layer_inputs if self.replay is not None else {},
            verify_conv_windows=self.conv_windows, verify_aux_windows=self.aux_windows,
        )
        if getattr(self.ex, "_ple", None) is not None:
            ctx.ple_rows = self.ex._ple_rows_block(batch.requests, batch.proposal.tokens.tolist())
        indptr = [0]
        for pages in batch.kv_pages:
            indptr.append(indptr[-1] + pages.shape[0])
        wrapper.plan(cu_host, torch.tensor(indptr, dtype=torch.int32), torch.cat(batch.kv_pages),
                     torch.ones(b_sz, dtype=torch.int32), self.ex.target.num_attention_heads, self.ex.target.num_kv_heads,
                     self.ex.target.head_dim, 1, causal=True, q_data_type=torch.bfloat16,
                     kv_data_type=KV_DTYPE, sm_scale=self.ex.target.head_dim**-0.5)
        ctx.paged_prefill_wrapper = wrapper
        return ctx

    def finish(self, batch: Verification, out) -> None:
        """Accept owned target output; preserve the common stop/commit/feedback path."""
        b_sz, k = batch.proposal.tokens.shape
        s_len = k + 1
        norm = out["norm"].view(b_sz, s_len, -1) if self.proposer.requirements.features.final_norm else None
        layers = {lid: value.view(b_sz, s_len, -1) for lid, value in out["features"].items()} if "features" in out else None
        logits = self.ex.model.logits(out["norm"])
        z = logits.view(b_sz, s_len, -1) if batch.sampled else logits.argmax(-1).view(b_sz, s_len)
        relaxed = None if batch.sampled else self._relaxed_matches(batch.requests, logits, batch.proposal.tokens, b_sz, s_len)
        self._accept_batch(batch, z, VerifiedFeatures(norm, layers), topk=self._target_topk(batch.requests, logits, b_sz, s_len), relaxed=relaxed)

    def _accept_batch(self, batch, z, features, topk=None, relaxed=None):
        self._accept(batch.requests, batch.proposal.tokens, z, features, batch.started, batch.prepared,
                     proposal=batch.proposal if batch.sampled else None, topk=topk, relaxed=relaxed)

    def _accept(self, reqs, y, z, features: VerifiedFeatures, t0, t1, *, proposal=None, topk=None, relaxed=None) -> None:
        if self.timing:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        ex = self.ex
        k, s_len = y.shape[1], y.shape[1] + 1
        # ---- accept ----
        if proposal is None:
            exact = (y == z[:, :k])
            match = exact if relaxed is None else (exact | relaxed)
            prefix = torch.cumprod(match.int(), dim=1)
            acc_t = prefix.sum(dim=1)
            # Relaxed accepts inside the accepted prefix: drafted tokens the target
            # did not rank first. Downloaded with everything else, one sync.
            flags_t = (prefix.bool() & ~exact) if relaxed is not None else torch.zeros_like(exact)
            packed = torch.cat([y, z, acc_t.unsqueeze(1).to(y.dtype), flags_t.to(y.dtype)], dim=1).tolist()
            y_l = [row[:k] for row in packed]
            z_l = [row[k:2 * k + 1] for row in packed]
            acc = [row[2 * k + 1] for row in packed]
            flags = [row[2 * k + 2:] for row in packed]  # block offsets accepted by the relaxed rule
            used = [sum(f) for f in flags]
        else:
            # Sampled batch: acceptance and the correction/bonus token are
            # computed on the device per row (vectorized rejection sampling,
            # `rejection.verify_rows`) and downloaded once with the drafts.
            acc_rows, corr_rows, target_rows, flag_rows = [], [], [], []
            for i, req in enumerate(reqs):
                if req.params.greedy:
                    target_ids = z[i].argmax(-1)
                    match = (y[i] == target_ids[:k]).to(torch.int64)
                    a_t = torch.cumprod(match, dim=0).sum()
                    acc_rows.append(a_t)
                    corr_rows.append(target_ids.index_select(0, a_t.reshape(1)).reshape(()))
                    target_rows.append(target_ids)  # greedy rows keep every argmax for traces
                    flag_rows.append(torch.zeros_like(match))
                else:
                    p = distribution(z[i], req.params)
                    relaxed_i = plausible(y[i], p, req.params.accept_top_k, req.params.accept_margin) if req.params.relaxed else None
                    a_t, c_t, rx_t = verify_rows(y[i], p, proposal.probabilities[i] if proposal.probabilities is not None else None,
                                                 proposal.support[i] if proposal.support is not None else None, req.generator,
                                                 relaxed=relaxed_i)
                    acc_rows.append(a_t)
                    corr_rows.append(c_t)
                    flag_rows.append(rx_t.to(torch.int64))
                    target_rows.append(torch.full((s_len,), -1, dtype=y.dtype, device=y.device).index_put_((a_t.reshape(1),), c_t.reshape(1)))
            packed = torch.cat([y, torch.stack(acc_rows).unsqueeze(1).to(y.dtype), torch.stack(corr_rows).unsqueeze(1).to(y.dtype),
                                torch.stack(target_rows).to(y.dtype), torch.stack(flag_rows).to(y.dtype)], dim=1).tolist()
            y_l = [row[:k] for row in packed]
            acc = [row[k] for row in packed]
            # Sampled rows carry only their correction at index a (-1 elsewhere).
            z_l = [row[k + 2:k + 2 + s_len] for row in packed]
            flags = [row[k + 2 + s_len:] for row in packed]  # block offsets accepted only by the relaxed rule
            used = [sum(f) for f in flags]
        outputs = []
        for i, r in enumerate(reqs):
            a = acc[i]
            emitted = (y_l[i][:a] + [z_l[i][a]])[:r.params.max_new_tokens - len(r.output_ids)]
            if not r.params.ignore_eos:
                for j, tok in enumerate(emitted):
                    if tok in r.params.stop_token_ids:
                        emitted = emitted[:j + 1]
                        break
            count = len(emitted)
            assert count > 0 and not r.finished
            outputs.append(emitted)
            if r.trace_rounds is not None:
                # Greedy rows carry every target argmax; sampled rows only
                # realize the correction token, so their z is omitted. `rx`
                # counts relaxed (non-argmax) accepts in the emitted prefix.
                record = {"a": a, "y": y_l[i], "z": z_l[i] if proposal is None or r.params.greedy else None, "c": z_l[i][a]}
                if used[i]:
                    # block offsets (0 = first drafted token) emitted under the relaxed rule; the trainer masks them
                    record["rx"] = [j for j, f in enumerate(flags[i][:count]) if f]
                r.trace_rounds.append(record)
        if self.timing:
            torch.cuda.synchronize()
            t_accepted = time.perf_counter()
        replay = getattr(self, "replay", None)
        if replay is not None:
            # StateManager validates every stop-truncated count before one
            # upload/kernel updates the private committed slots directly.
            ex.state.commit_reconstructed(reqs, [len(o) for o in outputs], replay, self.conv_windows_all, aux_windows=self.aux_windows,
                                          verified_steps=s_len)
        for i, (r, emitted) in enumerate(zip(reqs, outputs)):
            a, count = acc[i], len(emitted)
            # The last emitted token is still unprocessed. Commit exactly the
            # inputs that produced the visible outputs, including early stops.
            if replay is None:
                ex.state.commit(r, count, ssm=self.inter_ssm_all[:, i, count - 1],
                                conv=self.conv_windows_all[:, i, count - 1],
                                aux={name: win[i, count - 1] for name, win in self.aux_windows.items()})
            for tok in emitted:
                r.push(int(tok))
            self.proposed += self.k
            self.accepted += a
            self.emitted += len(emitted)
            relaxed_here = min(used[i], count)  # stop truncation can cut accepted tokens
            self.relaxed_accepts += relaxed_here
            r.spec_relaxed += relaxed_here
            r.spec_rounds += 1
            r.spec_proposed += self.k
            r.spec_accepted += a
            r.spec_emitted += count
        if self.timing:
            torch.cuda.synchronize()
            t_committed = time.perf_counter()
            self.accept_decide_seconds += t_accepted - t2
            self.accept_commit_seconds += t_committed - t_accepted
        self.proposer.accepted_batch(reqs, outputs, features)
        if self.timing:
            torch.cuda.synchronize()
            self.accept_feedback_seconds += time.perf_counter() - t_committed
        for i, (r, emitted) in enumerate(zip(reqs, outputs)):
            if r.trace_features is not None:
                self._trace_features(r, r.committed_len - len(emitted), r.committed_len)
            if r.trace_topk is not None and topk is not None:
                count = len(emitted)
                r.trace_topk.append(topk[0][i, :count], topk[1][i, :count], r.committed_len - count)
        self.rounds += 1
        if self.timing:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        self.draft_seconds += t1 - t0
        self.verify_seconds += t2 - t1
        self.accept_seconds += t3 - t2
