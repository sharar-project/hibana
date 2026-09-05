"""Proposer-independent greedy target verification and stop-aware acceptance.

Draft adapters supply candidates. StateManager owns reservations and commits;
this driver owns target verify buffers, graph replay and acceptance metrics.
"""

from __future__ import annotations

import time

import torch

from sharar import config
from sharar.engine.context import ForwardContext
from sharar.engine.request import Request
from sharar.memory.kv_pool import KV_DTYPE
from sharar.engine.contracts import Proposer, StepInputs, VerifiedFeatures


class SpecDecoder:
    def __init__(self, executor, proposer: Proposer):
        self.ex = executor
        self.proposer = proposer
        self.k = proposer.k
        self.steps = proposer.k + 1
        self.device = executor.device
        n = executor.max_batch
        ssm_shape = executor.state_pool.ssm_all.shape[2:]
        conv_shape = executor.state_pool.conv_all.shape[2:]
        layer_ids = executor.state_pool.layer_ids
        # Verify outputs, stacked over the GDN layers like the state pool so the
        # accept step commits a request's state with one strided copy per kind.
        self.inter_ssm_all = torch.zeros((len(layer_ids), n, self.steps, *ssm_shape), dtype=torch.float32, device=self.device)
        self.inter_ssm = {lid: self.inter_ssm_all[j] for j, lid in enumerate(layer_ids)}
        self.conv_windows_all = torch.zeros((len(layer_ids), n, self.steps, *conv_shape), dtype=torch.bfloat16, device=self.device)
        self.conv_windows = {lid: self.conv_windows_all[j] for j, lid in enumerate(layer_ids)}
        self.verify_graphs = None
        # Set by benchmarks: synchronize around the draft/verify/accept phases
        # to attribute time. Off in serving so the host can queue graphs ahead.
        self.timing = False
        # Pinned staging for the per-round host -> device upload: rows are
        # pending token, its position, state slot; one non-blocking copy.
        self._host_rows = torch.empty((3, n), dtype=torch.int64, pin_memory=self.device.type == "cuda")
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0
        self.accept_seconds = 0.0
        self.emitted = 0
        self.draft_seconds = 0.0
        self.verify_seconds = 0.0

    def capture_graphs(self, log=None) -> None:
        from sharar.engine.spec_graphs import VerifyGraphs

        self.verify_graphs = VerifyGraphs(self)
        self.verify_graphs.capture(log=log)
        self.proposer.capture_graphs(log=log)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.inter_ssm_all, self.conv_windows_all))

    def reset_stats(self) -> None:
        self.rounds = self.proposed = self.accepted = self.emitted = 0
        self.draft_seconds = self.verify_seconds = self.accept_seconds = 0.0

    def stats(self) -> dict:
        r = max(1, self.rounds)
        out = {
            "rounds": self.rounds, "proposed": self.proposed, "accepted": self.accepted, "emitted": self.emitted,
            "accept_rate": self.accepted / max(1, self.proposed), "tokens_per_round": self.emitted / r,
            "round_ms": 1000 * (self.draft_seconds + self.verify_seconds + self.accept_seconds) / r,
        }
        if self.timing:  # phase split is only meaningful with the synchronizing timers on
            out.update({
                "draft_ms_per_round": 1000 * self.draft_seconds / r,
                "verify_ms_per_round": 1000 * self.verify_seconds / r,
                "accept_ms_per_round": 1000 * self.accept_seconds / r,
            })
        return out

    def ingest_prefill(self, req, ids, start, result) -> None:
        self.proposer.ingest_prefill(req, ids, start, result)

    def start(self, req, token) -> None:
        self.proposer.start(req, token)

    def ready(self, req) -> bool:
        return self.proposer.ready(req)

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
        new_pages_flat = ex.state.reserve(reqs, s_len)
        new_pages = new_pages_flat.view(b_sz, s_len)
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
        y = self.proposer.propose(reqs, new_pages, inputs, graphs=ex.use_graphs)
        if (y.shape != (b_sz, k) or y.dtype != torch.int64 or y.device.type != self.device.type
                or (self.device.index is not None and y.device.index != self.device.index)):
            raise ValueError("proposer must return int64 [batch, k] on the target device")
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
            kv_all.append(ex.state.step_pages(r))
            kv_indptr.append(kv_indptr[-1] + kv_all[-1].shape[0])
        if ex.use_graphs and self.verify_graphs is not None:
            norm_rows, z_flat, ctx, layers = self.verify_graphs.run(tokens_v, positions, new_pages_flat, slots_t, kv_all)
            norm = norm_rows.view(b_sz, s_len, -1).clone() if self.proposer.requirements.features.final_norm else None
            layers = {lid: value.view(b_sz, s_len, -1).clone() for lid, value in layers.items()} if layers else None
            z = z_flat.view(b_sz, s_len).clone()
            self._accept(reqs, y, z, VerifiedFeatures(norm, layers), t0, t1)
            return
        cu = torch.arange(0, b_sz * s_len + 1, s_len, dtype=torch.int32, device=d)
        ctx = ForwardContext(
            feature_request=self.proposer.requirements.features,
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
        norm = out["norm"].view(b_sz, s_len, -1) if self.proposer.requirements.features.final_norm else None
        layers = {lid: value.view(b_sz, s_len, -1) for lid, value in out["features"].items()} if "features" in out else None
        z = ex.model.logits(out["norm"]).argmax(-1).view(b_sz, s_len)  # target choice after each token
        self._accept(reqs, y, z, VerifiedFeatures(norm, layers), t0, t1)

    def _accept(self, reqs, y, z, features: VerifiedFeatures, t0, t1) -> None:
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
            emitted = (y_l[i][:a] + [z_l[i][a]])[:r.params.max_new_tokens - len(r.output_ids)]
            if not r.params.ignore_eos:
                for j, tok in enumerate(emitted):
                    if tok in r.params.stop_token_ids:
                        emitted = emitted[:j + 1]
                        break
            count = len(emitted)
            assert count > 0 and not r.finished
            # The last emitted token is still unprocessed. Commit exactly the
            # inputs that produced the visible outputs, including early stops.
            ex.state.commit(r, count, ssm=self.inter_ssm_all[:, i, count - 1],
                            conv=self.conv_windows_all[:, i, count - 1])
            for tok in emitted:
                r.push(int(tok))
            self.proposed += k
            self.accepted += a
            self.emitted += len(emitted)
            r.spec_rounds += 1
            r.spec_proposed += k
            r.spec_accepted += a
            r.spec_emitted += count
            self.proposer.accepted(r, emitted, features, i)
        self.rounds += 1
        if self.timing:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        self.draft_seconds += t1 - t0
        self.verify_seconds += t2 - t1
        self.accept_seconds += t3 - t2
