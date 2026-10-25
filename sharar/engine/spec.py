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
from sharar.engine.contracts import Proposer, StepInputs, VerifiedFeatures
from sharar.engine.debug import debug_sync
from sharar.engine.depth import RowBudget
from sharar.distributed import all_reduce_max, get_tp_info
from sharar.engine import graph_set, sampling_rng
from sharar.engine.rejection import Proposal, distribution, plausible, verify_batch, verify_rows
from sharar.engine import accept_graphs
from sharar.engine.accept_graphs import AcceptGraphs


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
    # Constrained requests (engine/grammar.py): the drafted tokens on their way to the host, (pinned tensor, event),
    # copied right after drafting so the grammar walk runs while the GPU verifies.
    drafted_host: tuple | None = None
    # A packed verification (RaggedVerifyGraphs): (bucket, rows, every sequence's rows, unpacked width, each request's
    # acceptance limit), the bucket's slots and every sequence's KV pages, padded sequences included.
    ragged: tuple | None = None
    slots_all: torch.Tensor | None = None
    kv_all: list | None = None


class SpecDecoder:
    aux_windows: dict = {}  # class attribute: test decoders skip __init__

    @property
    def verify_depth(self) -> int:
        """The depth a fixed policy verifies (benchmarks set it to a captured narrower width)."""
        policy = getattr(self, "depth_policy", None)
        return policy.depth if policy is not None else self.k

    @verify_depth.setter
    def verify_depth(self, depth: int) -> None:
        from sharar.engine.depth import FixedDepth
        self.depth_policy = FixedDepth(depth)

    def __init__(self, executor, proposer: Proposer, state_strategy: str = "snapshots"):
        if state_strategy not in ("snapshots", "reconstruct"):
            raise ValueError("unknown verify state strategy")
        self.ex = executor
        self.proposer = proposer
        self.k = proposer.k
        self.steps = proposer.k + 1
        self.verify_depth = self.k
        self.verify_depths = (self.k,)  # benchmarks may explicitly capture narrower widths
        from sharar.engine.config import current
        from sharar.engine.depth import parse_depths
        from sharar.engine.graphs import capture_sizes
        policy = parse_depths(current().spec_depths, capture_sizes(executor.max_batch), self.k)
        if policy is not None:  # depth per batch bucket: verify graphs only where a bucket uses the depth
            self.depth_policy = policy
            self.verify_depths = policy.verify_depths
        if current().spec_step_cost and proposer.requirements.capabilities.short_drafts:
            # Adaptive depth per request: the depth above is the cap, and every depth below it verifies.
            from sharar.engine.depth import AcceptanceDepth
            self.depth_policy = AcceptanceDepth(self.depth_policy, current().spec_step_cost)
            self.verify_depths = self.depth_policy.verify_depths
        if current().spec_policy not in ("fixed", "budget"):
            raise ValueError("SHARAR_SPEC_POLICY must be fixed or budget")
        if current().spec_policy == "budget" and proposer.requirements.capabilities.scored_drafts:
            # Adaptive verification (engine/depth.RowBudget): the drafts' scores decide what a round verifies, in
            # packed graphs (a depth per request) for a target without side inputs that accepts on the host (the
            # 27B), else in padded graphs (one depth for the batch; Flash-Next). The block is drafted whole: one
            # forward costs about the same at any depth, and its draft graphs stay one per bucket.
            from sharar.engine.depth import ragged_budgets
            packed = not accept_graphs.enabled() and getattr(executor, "extras", None) is None
            sizes = capture_sizes(executor.max_batch)
            caps = policy.depths if policy is not None else None  # a depth table caps each bucket's depth
            self.depth_policy = RowBudget(self.k, sizes, packed_rows=ragged_budgets(sizes, self.steps) if packed else None,
                                          caps=caps)
            proposer.score_drafts = True
            self.short_drafts = False  # else the proposer's capability (`drafts_short`)
            self.verify_depths = self.depth_policy.verify_depths
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
        quantized = getattr(executor.state_pool, "window", None) is not None
        if quantized and state_strategy != "reconstruct":
            raise ValueError("a quantized recurrent state (SHARAR_SSM_STATE=int8/int4) commits through the verify ring: use --verify-state reconstruct")
        if state_strategy == "reconstruct":
            if quantized:  # one record per verified token; the commit appends or folds them
                from sharar.kernels.ssm_window import WindowState
                self.replay = WindowState(executor.state_pool, n, self.steps, executor.target.linear_num_key_heads)
            elif executor.target.recurrence == "mamba2":  # replays the verified x, B and dt
                from sharar.kernels.mamba2 import AcceptedState
                self.replay = AcceptedState(executor.state_pool, n, self.steps, executor.target.linear_num_key_heads)
            else:  # the commit replays with the arithmetic the model's verify uses
                from sharar.kernels.fla.accepted_state import AcceptedState
                cuda = any(getattr(layer.mixer, "cuda_verify", False) for layer in getattr(executor.model, "layers", ()))
                self.replay = AcceptedState(executor.state_pool, n, self.steps, executor.target.linear_num_key_heads, cuda=cuda)
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
        self.device_accept = accept_graphs.enabled()
        # Set by benchmarks: synchronize around the draft/verify/accept phases
        # to attribute time. Off in serving so the host can queue graphs ahead.
        self.timing = False
        self._round_timings: list[tuple] = []  # finished rounds' (batch, rows, events, tokens, sampled) for the depth policy
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
        from sharar.engine.graphs import capture_sizes
        from sharar.engine.spec_graphs import VerifyGraphs

        policy = self.depth_policy
        if isinstance(policy, RowBudget) and policy.packed:
            from sharar.engine.spec_graphs import RaggedVerifyGraphs
            self.ragged_graphs = RaggedVerifyGraphs(self, policy.budgets)
            self.ragged_graphs.capture(log=log)
        # deepest first: a shallower depth's graphs fit in the blocks a deeper one's freed (they share a pool)
        for depth in sorted(self.verify_depths, reverse=True):
            graphs = VerifyGraphs(self, depth, sizes=policy.sizes_for(depth, capture_sizes(self.ex.max_batch)))
            graphs.capture(log=log)
            self.width_graphs[depth] = graphs
        self.width_graphs = dict(sorted(self.width_graphs.items()))
        self.verify_graphs = (self.width_graphs.get(self.k) or self.width_graphs[max(self.width_graphs)]) if self.width_graphs else None
        if self.drafts_short:  # draft graphs at the verify graphs' (depth, buckets)
            self.proposer.capture_graphs(log=log, depths={d: g.sizes for d, g in self.width_graphs.items()})
        else:
            self.proposer.capture_graphs(log=log)
        self.measure_costs(log=log)
        if AcceptGraphs.supported(self):  # captured whenever possible so the path can be switched at run time
            pool = torch.cuda.graph_pool_handle()  # every depth's accept graphs share one pool (`AcceptGraphs`)
            for graphs in reversed(self.width_graphs.values()):  # deepest first, as the verify graphs
                graphs.accept = AcceptGraphs(self, graphs, pool=pool)
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
        policy = getattr(self, "depth_policy", None)
        if isinstance(policy, RowBudget):  # what it measured while serving
            label = lambda b, s: f"{b}{':sampled' if s else ''}"  # noqa: E731
            out["budget"] = {"packed": policy.packed,
                             "acceptance": [round(a / t, 3) for a, t in zip(*policy.survival[False])],
                             "sampled_acceptance": [round(a / t, 3) for a, t in zip(*policy.survival[True])],
                             "shift": [round(v, 2) for v in policy.shift],
                             "tokens_per_ms": {label(b, s): round(v, 3) for (b, s), v in policy.rate.items()},
                             "overhead_ms": {label(b, s): round(v, 3) for (b, s), v in policy.overhead.items()},
                             "depths": {str(b): {str(d): n for (bb, d), n in sorted(policy.chosen.items()) if bb == b}
                                        for b in sorted({b for b, _ in policy.chosen})},
                             "rows": {str(b): {str(r): n for (bb, r), n in sorted(policy.chosen_rows.items()) if bb == b}
                                      for b in sorted({b for b, _ in policy.chosen_rows})}}
        if self.timing:  # phase split is only meaningful with the synchronizing timers on
            out.update({
                "draft_ms_per_round": 1000 * self.draft_seconds / r,
                "verify_ms_per_round": 1000 * self.verify_seconds / r,
                "accept_ms_per_round": 1000 * self.accept_seconds / r,
                "accept_split_ms": [round(1000 * x / r, 1) for x in (self.accept_decide_seconds, self.accept_commit_seconds, self.accept_feedback_seconds)],
            })
        return out

    COST_CONTEXT = 4096  # cached tokens per request while measuring verify costs

    @property
    def drafts_short(self) -> bool:
        """Whether a round drafts only the depth it verifies (Capabilities.short_drafts), unless packed rows draft a
        block drafter's whole block."""
        return getattr(self, "short_drafts", self.proposer.requirements.capabilities.short_drafts)

    def measure_costs(self, log=None) -> None:
        """The measured costs `RowBudget` decides from: each captured verify graph's time per batch bucket."""
        policy = self.depth_policy
        if not isinstance(policy, RowBudget):
            return
        context = min(self.COST_CONTEXT, self.ex.max_context)
        ragged = getattr(self, "ragged_graphs", None)
        if ragged is not None:
            context = min(self.COST_CONTEXT, self.ex.max_context - 1)  # the graphs hold max_context + k cached tokens
            for b in ragged.sizes:
                for rows in ragged.budgets[b]:
                    policy.costs[b, rows] = ragged.time_replay(b, rows, context)
            self._common_costs()
            if log:
                for b in ragged.sizes:
                    log(f"  verify cost, batch {b}: " + ", ".join(f"{r} rows {policy.costs[b, r]:.1f}" for r in ragged.budgets[b]) + " ms")
            return
        for depth, graphs in sorted(self.width_graphs.items()):
            for b in graphs.sizes:
                policy.costs[b, b * (depth + 1)] = graphs.time_replay(b, context)
        self._common_costs()
        if log:
            for b in policy.sizes:
                log(f"  verify cost, batch {b}: " + ", ".join(f"{d}: {policy.costs[b, b * (d + 1)]:.1f}" for d in policy.candidates) + " ms")

    def _common_costs(self) -> None:
        """Tensor-parallel ranks decide alike only from the same numbers: every rank keeps the slowest rank's costs."""
        costs = self.depth_policy.costs
        keys = sorted(costs)
        costs.update(zip(keys, all_reduce_max([costs[k] for k in keys])))

    def ingest_prefill(self, req, ids, start, result) -> None:
        self.proposer.ingest_prefill(req, ids, start, result)

    def start(self, req, token) -> None:
        self.proposer.start(req, token)

    def ready(self, req) -> bool:
        return self.proposer.ready(req)

    # ---- one round ----
    @torch.inference_mode()
    def round(self, reqs: list[Request]) -> None:
        debug_sync("before spec round", reqs)
        learn = getattr(self.depth_policy, "observe_round", None) if self.device.type == "cuda" else None
        if learn is not None:
            self._learn_round(learn)
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            emitted = self.emitted
        batch = self.prepare(reqs)
        debug_sync("after draft (prepare)", reqs)
        self.verify(batch)
        debug_sync("after verify and accept", reqs)
        if learn is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            depth = batch.proposal.tokens.shape[1]
            rows = batch.ragged[1] if batch.ragged is not None else graph_set.pick(self.depth_policy.sizes, len(reqs)) * (depth + 1)
            self._round_timings.append((len(reqs), rows, start, end, self.emitted - emitted, batch.sampled))

    TP_LEARN_EVERY = 8  # tensor-parallel rounds whose times the ranks exchange at once

    def _learn_round(self, learn) -> None:
        """Finished rounds' GPU times to the depth policy. One Spark: the previous round's, once the GPU finished it
        (never waits). Tensor-parallel ranks must decide alike, so they learn the same times at the same round: every
        TP_LEARN_EVERY rounds, the slowest rank's times of the rounds before the last (one small all-reduce)."""
        pending = self._round_timings
        if get_tp_info().size == 1:
            if pending and pending[-1][3].query():
                b, rows, start, end, tokens, sampled = pending[-1]
                learn(b, rows, start.elapsed_time(end), tokens, sampled=sampled)
            pending.clear()
            return
        if len(pending) <= self.TP_LEARN_EVERY:
            return
        done, self._round_timings = pending[:self.TP_LEARN_EVERY], pending[self.TP_LEARN_EVERY:]
        for t in done:
            t[3].synchronize()
        times = all_reduce_max([start.elapsed_time(end) for _, _, start, end, *_ in done])
        for (b, rows, _, _, tokens, sampled), ms in zip(done, times):
            learn(b, rows, ms, tokens, sampled=sampled)

    def prepare(self, reqs: list[Request]) -> Verification:
        """Reserve and draft one target verification batch."""
        ex = self.ex
        policy = getattr(self, 'depth_policy', None)
        k = policy.choose(reqs) if policy is not None else self.k
        packed = isinstance(policy, RowBudget) and policy.packed  # packed graphs hold every depth up to k
        if not (1 <= k <= self.k if packed else k in getattr(self, 'verify_depths', (self.k,))):
            raise ValueError('uncaptured verification depth')
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
        short = self.drafts_short
        proposed = self.proposer.propose(reqs, new_pages, inputs, graphs=ex.use_graphs, **({"depth": k} if short else {}))
        proposal = proposed if isinstance(proposed, Proposal) else Proposal(proposed)
        y = proposal.tokens
        s_len = k + 1
        sampled = any(not r.params.greedy for r in reqs)
        if (y.shape != (b_sz, k if short else self.k) or y.dtype != torch.int64 or y.device.type != self.device.type
                or (self.device.index is not None and y.device.index != self.device.index)):
            raise ValueError("proposer must return int64 [batch, k] on the target device")
        if getattr(self, "ragged_graphs", None) is not None and ex.use_graphs:
            return self._prepare_ragged(reqs, proposal, y, k, toks_t, poss_t, slots_t, new_pages, sampled, t0, timing)
        if isinstance(policy, RowBudget) and not sampled and not any(r.grammar is not None for r in reqs):
            k = policy.allot(reqs, proposal.acceptance, k)[0][0]  # padded graphs: one depth from the drafts' scores
            s_len = k + 1
        y = y[:, :k]
        proposal = Proposal(y, proposal.probabilities[:, :k] if proposal.probabilities is not None else None,
                            proposal.support[:, :k] if proposal.support is not None else None,
                            confidence=proposal.confidence[:, :k] if proposal.confidence is not None else None, columns=proposal.columns,
                            acceptance=proposal.acceptance[:, :k] if proposal.acceptance is not None else None)
        new_pages_flat = new_pages[:, :s_len].reshape(-1)
        drafted_host = None
        if any(r.grammar is not None for r in reqs) and y.device.type == "cuda":
            host = torch.empty(y.shape, dtype=torch.int64, pin_memory=True)
            host.copy_(y, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
            drafted_host = (host, ready)
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
                            slots_t, kv_all, t0, t1, drafted_host)

    def _prepare_ragged(self, reqs, proposal, y, k, toks_t, poss_t, slots_t, new_pages, sampled, t0, timing) -> Verification:
        """The round's packed verification: each request's depth (the policy's limits), its rows, then padding rows on
        its own reserved pages and the bucket's padded sequences, to the captured row total."""
        ex, d, policy, graphs = self.ex, self.device, self.depth_policy, self.ragged_graphs
        b = len(reqs)
        bucket = graph_set.pick(graphs.sizes, b)
        if sampled or any(r.grammar is not None for r in reqs):
            limits, rows = policy.allot(reqs, None, k)  # one depth for all, chosen before drafting
        else:
            limits, rows = policy.allot(reqs, proposal.acceptance, k)
        width, steps = max(limits) + 1, self.steps
        lengths = [n + 1 for n in limits]
        padded = [1] * (bucket - b)
        spare = rows - sum(lengths) - len(padded)
        for group, cap in ((padded, steps), (lengths, steps)):  # the padded sequences first, then the requests' own
            for i in range(len(group)):
                add = min(spare, cap - group[i])
                group[i] += add
                spare -= add
        assert spare == 0, (rows, limits)
        drafts = proposal.tokens
        if drafts.shape[1] < steps - 1:  # a short draft: the rows past it pad (never within a limit, which is at most k)
            drafts = torch.nn.functional.pad(drafts, (0, steps - 1 - drafts.shape[1]))
        blocks = torch.cat([toks_t.unsqueeze(1), drafts], dim=1).reshape(-1)  # [b * steps]: pending, then drafts
        index = [i * steps + j for i, n in enumerate(lengths) for j in range(n)]
        local = [j for n in lengths for j in range(n)]
        seq = [i for i, n in enumerate(lengths) for _ in range(n)]
        host = torch.tensor([index, local, seq], dtype=torch.int64, pin_memory=d.type == "cuda").to(d, non_blocking=True)
        tokens = blocks.index_select(0, host[0])
        positions = poss_t.index_select(0, host[2]) + host[1]
        pages = new_pages.reshape(-1).index_select(0, host[0]).to(torch.int32)
        kv_all = [ex.state.step_pages(r)[:r.committed_len + n] for r, n in zip(reqs, lengths)]
        if padded:
            pad_rows = sum(padded)
            tokens = torch.cat([tokens, torch.zeros(pad_rows, dtype=tokens.dtype, device=d)])
            positions = torch.cat([positions, torch.zeros(pad_rows, dtype=positions.dtype, device=d)])
            pages = torch.cat([pages, ex.kv_pool.pad_pages(pad_rows).to(torch.int32)])
            kv_all += [torch.zeros(n + 1, dtype=torch.int32, device=d) for n in padded]
        slots_all = torch.cat([slots_t, torch.zeros(len(padded), dtype=slots_t.dtype, device=d)])
        cut = width - 1
        proposal = Proposal(proposal.tokens[:, :cut], proposal.probabilities[:, :cut] if proposal.probabilities is not None else None,
                            proposal.support[:, :cut] if proposal.support is not None else None,
                            confidence=proposal.confidence[:, :cut] if proposal.confidence is not None else None,
                            columns=proposal.columns)
        drafted_host = None
        if any(r.grammar is not None for r in reqs) and d.type == "cuda":
            drafted_host = (torch.empty(proposal.tokens.shape, dtype=torch.int64, pin_memory=True), torch.cuda.Event())
            drafted_host[0].copy_(proposal.tokens, non_blocking=True)
            drafted_host[1].record()
        if timing:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return Verification(list(reqs), proposal, sampled, tokens, positions, pages, slots_t, kv_all[:b], t0, t1, drafted_host,
                            ragged=(bucket, rows, lengths + padded, width, limits), slots_all=slots_all, kv_all=kv_all)

    def _verify_ragged(self, batch: Verification) -> None:
        bucket, rows, lengths, width, limits = batch.ragged
        st = self.ragged_graphs.run(bucket, rows, lengths, width, batch.tokens, batch.positions, batch.pages,
                                    batch.slots_all, batch.kv_all)
        reqs = batch.requests
        b = len(reqs)
        n = b * width
        logits = st["logits"][:n]
        constrained = self._constrain(batch, logits)
        z_flat = logits.argmax(-1) if constrained else st["argmax"][:n]
        norm = st["norm"][:n].view(b, width, -1).clone() if self.proposer.requirements.features.final_norm else None
        layers = {lid: value[:n].view(b, width, -1).clone() for lid, value in st["features"].items()} if st["features"] else None
        z = logits.view(b, width, -1) if batch.sampled else z_flat.view(b, width).clone()
        y = batch.proposal.tokens
        relaxed = None if batch.sampled else self._relaxed_matches(reqs, logits, y, b, width)
        self._accept(reqs, y, z, VerifiedFeatures(norm, layers), batch.started, batch.prepared,
                     proposal=batch.proposal if batch.sampled else None, relaxed=relaxed, confidence=batch.proposal.confidence,
                     limits=None if batch.sampled else limits)

    def verify(self, batch: Verification) -> None:
        if batch.ragged is not None:
            return self._verify_ragged(batch)
        b_sz, k = batch.proposal.tokens.shape
        s_len = k + 1
        seqs = [(r.context_len - 1, s_len, pages) for r, pages in zip(batch.requests, batch.kv_pages)]
        variant = self.ex.variant(seqs)
        graphs = self.width_graphs.get(k, self.verify_graphs)
        if self.ex.use_graphs and graphs is not None and variant in graphs.variants:
            norm_rows, z_flat, ctx, layers, logits = graphs.run(batch.tokens, batch.positions, batch.pages,
                                                               batch.slots, batch.kv_pages, sampled=batch.sampled,
                                                               side=(batch.requests, batch.proposal.tokens, seqs, batch.proposal.columns), variant=variant)
            if self._constrain(batch, logits) and not batch.sampled:
                # The masked argmax, written into the graph's own buffer: the device acceptance graph reads it there
                # (and the masked logits for the relaxed rule). Sampled batches read the masked logits themselves.
                z_flat.copy_(logits.argmax(-1))
            accept = getattr(graphs, "accept", None)
            if accept is not None and self.device_accept and (accept.eligible_sampled(batch.requests, batch.proposal) if batch.sampled
                                                              else accept.eligible(batch.requests)):
                if self.timing:
                    torch.cuda.synchronize()
                if accept.round(batch, batch.proposal.tokens, batch.started, batch.prepared):
                    return  # else a sampled law wider than its candidates: the host path below
            norm = norm_rows.view(b_sz, s_len, -1).clone() if self.proposer.requirements.features.final_norm else None
            layers = {lid: value.view(b_sz, s_len, -1).clone() for lid, value in layers.items()} if layers else None
            z = z_flat.view(b_sz, s_len, -1) if batch.sampled else z_flat.view(b_sz, s_len).clone()
            relaxed = None if batch.sampled else self._relaxed_matches(batch.requests, logits, batch.proposal.tokens, b_sz, s_len)
            self._accept_batch(batch, z, VerifiedFeatures(norm, layers), relaxed=relaxed)
            return
        ctx = self.context(batch, self._eager_wrapper())
        self.finish(batch, self.ex.model.forward(batch.tokens, ctx))

    @staticmethod
    def _confidence_rows(reqs, confidence):
        """The draft's (top-1 probability, top-1 minus top-2) per position as host lists, when a request records rounds."""
        if confidence is None or not any(r.trace_rounds is not None for r in reqs):
            return None
        return confidence.float().tolist()

    def _eager_wrapper(self):
        """The eager verification's own paged-prefill wrapper (Executor._verify_prefill). plan()
        uploads its metadata from a pinned buffer asynchronously, so sharing the prefill wrapper
        let a cache-hit prefill and a verification in the same scheduler step overwrite each
        other's plan before the GPU read it (illegal memory access under load)."""
        return getattr(self.ex, "_verify_prefill", self.ex._paged_prefill)  # test doubles have only one

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

    def _constrain(self, batch: Verification, logits: torch.Tensor) -> bool:
        """Mask the verified rows of constrained requests by their grammar (engine/grammar.py), in place.

        logits [b * (k + 1), V]. Row j of a request is masked by the grammar state after its drafted y[:j], so
        acceptance and the correction run on what the grammar allows. Returns False (and downloads nothing) when
        no request in the batch is constrained."""
        if not any(r.grammar is not None for r in batch.requests):
            return False
        from sharar.engine.grammar import apply_bitmask, verify_masks
        early = getattr(batch, "drafted_host", None)
        if early is not None:
            early[1].synchronize()  # waits for the draft only; the verify graph keeps the GPU busy during the walk
            drafted = early[0].tolist()
        else:
            drafted = batch.proposal.tokens.tolist()
        masks = verify_masks(batch.requests, drafted, logits.shape[-1])
        s = batch.proposal.tokens.shape[1] + 1
        for i, r in enumerate(batch.requests):
            if r.grammar is not None:  # unconstrained requests' rows stay as they are
                apply_bitmask(logits[i * s:(i + 1) * s], masks[i * s:(i + 1) * s])
        self.constrained_rounds = getattr(self, "constrained_rounds", 0) + 1
        return True

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
        if getattr(self.ex, "extras", None) is not None:
            ctx.extras = self.ex.extras.for_verify(batch.requests, batch.proposal.tokens,
                                                   [(r.context_len - 1, s_len, pages) for r, pages in zip(batch.requests, batch.kv_pages)])
        indptr = [0]
        for pages in batch.kv_pages:
            indptr.append(indptr[-1] + pages.shape[0])
        self.ex.attention.paged_prefill(wrapper, cu_host, torch.tensor(indptr, dtype=torch.int32), torch.cat(batch.kv_pages),
                                        torch.ones(b_sz, dtype=torch.int32), causal=True, page_size=1)
        ctx.paged_prefill_wrapper = wrapper
        return ctx

    def finish(self, batch: Verification, out) -> None:
        """Accept owned target output; preserve the common stop/commit/feedback path."""
        b_sz, k = batch.proposal.tokens.shape
        s_len = k + 1
        norm = out["norm"].view(b_sz, s_len, -1) if self.proposer.requirements.features.final_norm else None
        layers = {lid: value.view(b_sz, s_len, -1) for lid, value in out["features"].items()} if "features" in out else None
        logits = self.ex.model.logits(out["norm"])
        self._constrain(batch, logits)
        z = logits.view(b_sz, s_len, -1) if batch.sampled else logits.argmax(-1).view(b_sz, s_len)
        relaxed = None if batch.sampled else self._relaxed_matches(batch.requests, logits, batch.proposal.tokens, b_sz, s_len)
        self._accept_batch(batch, z, VerifiedFeatures(norm, layers), relaxed=relaxed)

    def _accept_batch(self, batch, z, features, relaxed=None):
        self._accept(batch.requests, batch.proposal.tokens, z, features, batch.started, batch.prepared,
                     proposal=batch.proposal if batch.sampled else None, relaxed=relaxed, confidence=batch.proposal.confidence)

    def _accept(self, reqs, y, z, features: VerifiedFeatures, t0, t1, *, proposal=None, relaxed=None, confidence=None,
                limits=None) -> None:
        if self.timing:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        ex = self.ex
        k, s_len = y.shape[1], y.shape[1] + 1
        observe = getattr(getattr(self, "depth_policy", None), "observe", None)  # adaptive depth learns each request's rate
        # ---- accept ----
        if proposal is None:
            exact = (y == z[:, :k])
            match = exact if relaxed is None else (exact | relaxed)
            if limits is not None:  # a request's drafts past its own depth were not verified for it
                limit_t = torch.tensor(limits, device=y.device)
                match = match & (torch.arange(k, device=y.device)[None, :] < limit_t[:, None])
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
            # computed on the device and downloaded once with the drafts. Point-mass
            # drafts (MTP) and sparse draft laws (DFlash2) verify the whole batch on
            # each row's candidates (`rejection.verify_batch`); dense draft laws, and
            # laws wider than the candidates, verify row by row (`verify_rows`).
            acc_rows, corr_rows, target_rows, flag_rows = [], [], [], []
            batched = None
            if (proposal.probabilities is None) == (proposal.support is None):  # point masses, or a sparse law on its support
                batched = verify_batch(y, z, [r.params for r in reqs], [r.generator for r in reqs], proposal.probabilities, proposal.support,
                                       rng=[[r.rng_seed, o] for r, o in zip(reqs, sampling_rng.offsets(reqs, sampling_rng.VERIFY))])
            if batched is not None:
                acc_b, corr_b, flags_b, argmax_b = batched
                # sampled rows carry only their correction at index a (-1 elsewhere); greedy rows every argmax, for traces
                targets = torch.full((len(reqs), s_len), -1, dtype=y.dtype, device=y.device).scatter_(1, acc_b.view(-1, 1), corr_b.view(-1, 1))
                if argmax_b is not None:
                    greedy_rows = torch.tensor([r.params.greedy for r in reqs], device=y.device).view(-1, 1)
                    targets = torch.where(greedy_rows, argmax_b.to(y.dtype), targets)
                acc_rows, corr_rows, flag_rows, target_rows = list(acc_b), list(corr_b), list(flags_b.to(torch.int64)), list(targets)
            for i, req in enumerate(reqs if batched is None else ()):
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
        conf = self._confidence_rows(reqs, confidence)
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
                    # block offsets (0 = first drafted token) emitted under the relaxed rule rather than as the argmax
                    record["rx"] = [j for j, f in enumerate(flags[i][:count]) if f]
                if conf is not None:  # the draft's confidence per drafted position (SHARAR_TRACE_CONFIDENCE)
                    record["p1"] = [round(c[0], 5) for c in conf[i]]
                    record["m"] = [round(c[1], 5) for c in conf[i]]
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
            depth = k if limits is None else limits[i]
            self.proposed += depth
            self.accepted += a
            self.emitted += len(emitted)
            if observe is not None:
                observe(r, depth, a)
            relaxed_here = min(used[i], count)  # stop truncation can cut accepted tokens
            self.relaxed_accepts += relaxed_here
            r.spec_relaxed += relaxed_here
            r.spec_rounds += 1
            r.spec_proposed += depth
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
        self.rounds += 1
        if self.timing:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        self.draft_seconds += t1 - t0
        self.verify_seconds += t2 - t1
        self.accept_seconds += t3 - t2
