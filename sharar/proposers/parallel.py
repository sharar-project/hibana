"""The DFlash2 adapter with bounded context KV and feature restoration.

Target pages and recurrent state remain owned by StateManager. This adapter
owns a fixed-size draft ring per active request. Cached projected features
are bound to recurrent-slot lifetime when installed in a cached executor.
Radix entries contain identity-keyed handles only.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace

import torch

from sharar.target import DEFAULT_TARGET
from sharar.engine.contracts import Capabilities, FeatureRequest, ProposerRequirements
from sharar.engine.rejection import Proposal, sample_lattice

from sharar.model.qwen38_config import DRAFT_PINS as PINS  # the pinned parallel drafts


class FeatureCache:
    """Independent LRU reference for standalone adapters and benchmark controls."""
    def __init__(self, capacity):
        self.capacity, self.bytes, self.serial = capacity, 0, 0
        self.entries = OrderedDict()

    def put(self, start, end, features, *, slot=None):
        size = features.numel() * features.element_size()
        if size > self.capacity:
            return None
        while self.entries and self.bytes + size > self.capacity:
            _, (_, _, old) = self.entries.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        self.serial += 1
        self.entries[self.serial] = (start, end, features.clone())
        self.bytes += size
        return self.serial

    def get(self, handle):
        value = self.entries.get(handle)
        if value is not None:
            self.entries.move_to_end(handle)
        return value


class CheckpointFeatureCache(FeatureCache):
    """Feature handles share the lifetime of their target recurrent slot.

    No independent eviction: each usable slot can retain one window. Freeing
    or recycling the target slot invalidates its handle, including duplicate
    radix insertion, cancellation, snapshot eviction and flush.
    """
    def __init__(self, pool, window_bytes):
        super().__init__((pool.num_slots - 1) * window_bytes)
        self.pool, self.window_bytes, self.slots = pool, window_bytes, {}
        pool.on_free.append(self.drop_slot)

    def drop_slot(self, slot):
        handle = self.slots.pop(slot, None)
        value = self.entries.pop(handle, None)
        if value is not None:
            self.bytes -= value[2].numel() * value[2].element_size()

    def put(self, start, end, features, *, slot=None):
        if not self.pool.is_allocated(slot):
            raise ValueError('features require an owned recurrent slot')
        size = features.numel() * features.element_size()
        if size > self.window_bytes:
            raise ValueError('checkpoint features exceed the reserved window')
        value = features.clone()
        self.drop_slot(slot)
        self.serial += 1
        self.entries[self.serial] = (start, end, value)
        self.slots[slot] = self.serial
        self.bytes += size
        assert self.bytes <= self.capacity
        return self.serial


@dataclass
class DraftState:
    slot: int
    end: int = 0
    valid: bool = True
    pending: int | None = None


class ParallelProposer:
    def __init__(self, model, context, k, name):
        import flashinfer
        self.model, self.context, self.k, self.name = model, context, k, name
        self.feedback_graphs = None
        self._feedback_enabled = False
        self.noise_steps = k + 1  # the anchor (the last committed token) and k draft positions
        self.window = model.window
        self.pitch = self.window + self.noise_steps
        self.free_slots = list(range(context.max_batch, 0, -1))
        served = getattr(model.target, 'target', DEFAULT_TARGET)  # the draft's target model carries the serving target
        self.cache_identity = (served.model_revision, *PINS[name], model.features, 'projected-bf16-kv-v1', self.window,
                               getattr(model, 'source', None))
        device = context.device
        shape = ((context.max_batch + 1) * self.pitch, 1, model.kv_heads, model.head_dim)
        self.caches = [(torch.empty(shape, dtype=torch.bfloat16, device=device), torch.empty(shape, dtype=torch.bfloat16, device=device))
                       for _ in range(model.layers)]
        self.features = torch.empty((context.max_batch + 1, self.window, model.hidden), dtype=torch.bfloat16, device=device)
        self.feature_cache = FeatureCache(context.max_batch * self.window * model.hidden * 2)
        self.requirements = ProposerRequirements(name, FeatureRequest(layers=model.features),
                                                persistent_bytes=self.bytes, greedy_only=False, max_depth=15,
                                                capabilities=Capabilities(device_feedback=True, checkpoint_cache=True,
                                                                          cold_payload=True))
        self.workspace = torch.empty(128 * 2**20, dtype=torch.uint8, device=device)
        self.attention = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, 'NHD')
        from sharar.kernels.attention_plan import AttentionGeometry, AttentionPlanner
        self.planner = AttentionPlanner(AttentionGeometry(model.heads, model.kv_heads, model.head_dim,
                                                          window_left=self.window - 1))
        self.graphs = None
        self.cache_hits = self.cache_fallbacks = 0

    def bind_checkpoint_cache(self, pool):
        if self.feature_cache.entries:
            raise ValueError('bind checkpoint retention before serving requests')
        self.feature_cache = CheckpointFeatureCache(pool, self.window * self.model.hidden * 2)
        self.requirements = replace(self.requirements, persistent_bytes=self.bytes)

    @property
    def bytes(self):
        tensors = [self.features, *(t for pair in self.caches for t in pair)]
        feedback_bytes = self.feedback_graphs.bytes if self.feedback_graphs is not None else 0
        return (self.model.bytes + sum(t.numel() * t.element_size() for t in tensors)
                + self.feature_cache.capacity + 128 * 2**20 + feedback_bytes)

    def _state(self, req):
        if req.proposer_state is None:
            if not self.free_slots:
                raise RuntimeError('draft request capacity exceeded')
            req.proposer_state = DraftState(self.free_slots.pop())
        if not isinstance(req.proposer_state, DraftState):
            raise TypeError('foreign proposer state')
        return req.proposer_state

    def _write(self, state, start, hidden):
        positions = torch.arange(start, start + len(hidden), device=self.context.device)
        ring = positions % self.window
        self.features[state.slot, ring] = hidden
        self.model.context_kv(hidden, positions, self.caches, state.slot * self.pitch + ring)
        state.end = start + len(hidden)

    def _save(self, req, position, *, slot=None):
        state = self._state(req)
        if not state.valid or not req.cache_enabled:
            return
        req.cache_auxiliary = {p: value for p, value in req.cache_auxiliary.items() if p in req.checkpoints}
        start = max(0, position - self.window)
        if slot is None:
            slot = req.checkpoints.get(position, getattr(req, 'slot', None))
        # Prompt completion and release can save the same immutable prefix.
        # Reuse its live handle so a duplicate window does not evict another
        # checkpoint's features while the target snapshot is still cached.
        handle = req.cache_auxiliary.get(position, {}).get(self.cache_identity)
        cached = self.feature_cache.get(handle)
        same_slot = (not isinstance(self.feature_cache, CheckpointFeatureCache)
                     or self.feature_cache.slots.get(slot) == handle)
        if cached is not None and cached[:2] == (start, position) and same_slot:
            return
        ring = torch.arange(start, position, device=self.context.device) % self.window
        handle = self.feature_cache.put(start, position, self.features[state.slot, ring], slot=slot)
        if handle is not None:
            req.cache_auxiliary[position] = {self.cache_identity: handle}

    def ingest_prefill(self, req, ids, start, result):
        state = self._state(req)
        if start and state.end == 0:
            handle = req.match.node.auxiliary.get(self.cache_identity) if req.match is not None else None
            cached = self.feature_cache.get(handle)
            if cached is not None and cached[1] == start:
                self._write(state, cached[0], cached[2])
                self.cache_hits += 1
            else:
                state.valid = False
                self.cache_fallbacks += 1
        end = start + len(ids)
        # Split at snapshot boundaries before the ring overwrites their history.
        bounds = sorted({p for p in req.checkpoints if start < p < end} | {end})
        projected = self.model.project_features(result['features'])
        offset = start
        for limit in bounds:
            # A long first prefill can exceed a sliding window. Only its last
            # window is needed at this boundary; absolute RoPE positions stay.
            left = max(offset, limit - self.window)
            self._write(state, left, projected[left - start:limit - start])
            if limit in req.checkpoints:
                self._save(req, limit)
            offset = limit

    def start(self, req, token):
        state = self._state(req)
        state.pending = token
        self._save(req, state.end)  # prompt snapshot (also used by one-token primers)

    def ready(self, req):
        state = req.proposer_state
        return isinstance(state, DraftState) and state.valid and state.pending is not None

    def auxiliary_codec(self):
        """Export/import of this adapter's cache auxiliary (a feature-cache
        handle) for the cold tier: the projected feature window travels with
        the parked session so a restored request keeps speculating."""
        def export(handle):
            value = self.feature_cache.get(handle)  # (start, end, features on the device); the tier copies the tensor
            return None if value is None else (value[0], value[1], value[2])

        def import_(payload, *, slot=None):
            start, end, features = payload  # features is a view into the tier's arena; put() clones it
            return self.feature_cache.put(start, end, features, slot=slot)

        return self.cache_identity, export, import_

    def _plan(self, wrapper, histories, batch):
        s = self.noise_steps
        lengths = [len(p) for p in histories]
        indptr = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
        self.planner.paged_prefill(wrapper, torch.arange(0, (batch + 1) * s, s, dtype=torch.int32), indptr,
                                   torch.cat(histories).to(torch.int32), torch.ones(batch, dtype=torch.int32), causal=False, guard=False)

    def propose(self, requests, pages, inputs, *, graphs):
        self._feedback_enabled = graphs
        batch, s, device = len(requests), self.noise_steps, self.context.device
        tokens = torch.full((batch, s), 248070, device=device, dtype=torch.int64)
        tokens[:, 0] = inputs.tokens
        positions = (inputs.positions[:, None] + torch.arange(s, device=device)).reshape(-1)
        write, histories = [], []
        for req in requests:
            state = self._state(req)
            if state.end != req.committed_len or not state.valid:
                raise ValueError('draft context is incomplete')
            base = state.slot * self.pitch
            noise = torch.arange(base + self.window, base + self.pitch, device=device)
            history = torch.arange(max(0, state.end - self.window), state.end, device=device) % self.window + base
            write.append(noise)
            histories.append(torch.cat([history, noise]))
        if graphs and self.graphs is not None:
            hidden, candidates, scores = self.graphs.run(tokens, positions, torch.cat(write), histories)
        else:
            self._plan(self.attention, histories, batch)
            hidden = self.model.forward(tokens, positions, torch.cat(write), self.caches, self.attention)[:, 1:]
            candidates, scores = self.model.selector(hidden, inputs.tokens)
        sampled = any(not r.params.greedy for r in requests)
        if not sampled:
            from sharar.kernels.draft_selection import greedy_lattice
            return Proposal(greedy_lattice(candidates, scores))
        # One batched walk of the lattice for every row (rejection.sample_lattice): each step draws from the
        # request's law over the candidates that follow its previous choice, with the request's own generator.
        tokens_s, laws = sample_lattice(candidates, scores, [r.params for r in requests], [r.generator for r in requests])
        return Proposal(tokens_s, laws, candidates)

    def accepted_batch(self, requests, emitted, features):
        """Project committed rows together, then scatter into private rings.

        The target has already committed all rows. Temporary work is bounded
        by max_batch * (k + 1), independent of context/cache capacity. Single
        requests use exact-count graphs when available; interior checkpoints
        retain the sequential oracle.
        """
        if not 1 <= len(requests) <= self.context.max_batch or len(requests) != len(emitted):
            raise ValueError('invalid accepted batch')
        counts = [len(tokens) for tokens in emitted]
        starts = [req.committed_len - count for req, count in zip(requests, counts)]
        states = [req.proposer_state for req in requests]
        if (any(not 1 <= n <= self.k + 1 for n in counts)
                or any(not isinstance(s, DraftState) or not s.valid or s.end != start or start < 0
                       for s, start in zip(states, starts))
                or len({s.slot for s in states}) != len(states)):
            raise ValueError('accepted rows do not follow private draft prefixes')
        if features.layers is None or any(features.layers[lid].ndim != 3 or features.layers[lid].shape[0] != len(requests)
                                         or features.layers[lid].shape[1] < max(counts) for lid in self.model.features):
            raise ValueError('accepted features do not cover the committed rows')
        if len(requests) == 1 or any(count > self.window or any(start < p < req.committed_len for p in req.checkpoints)
                                     for req, start, count in zip(requests, starts, counts)):
            for row, (req, tokens) in enumerate(zip(requests, emitted)):
                self._accepted(req, tokens, features, row)
            return
        layers = {lid: torch.cat([features.layers[lid][row, :count] for row, count in enumerate(counts)])
                  for lid in self.model.features}
        hidden = self.model.project_features(layers)
        # One asynchronous upload; no GPU scalar extraction or per-row wait.
        metadata = [[pos, state.slot * self.window + pos % self.window,
                     state.slot * self.pitch + pos % self.window]
                    for state, start, count in zip(states, starts, counts) for pos in range(start, start + count)]
        metadata = torch.tensor(metadata, dtype=torch.int64, pin_memory=self.context.device.type == 'cuda')
        positions, rows, pages = metadata.to(self.context.device, non_blocking=True).unbind(1)
        self.features.view(-1, self.features.shape[-1])[rows] = hidden
        self.model.context_kv(hidden, positions, self.caches, pages)
        for req, state, tokens in zip(requests, states, emitted):
            state.end = req.committed_len
            if state.end in req.checkpoints:
                self._save(req, state.end)
            state.pending = None if req.finished else tokens[-1]

    # ---- device-resident acceptance (engine/accept_graphs.py) ----
    def accepted_prepare(self, requests):
        """Validate the batch before a captured feedback pass whose counts the device decides.

        Returns (starts, ring slots) per request: the committed prefix length
        each request's emitted rows continue from and its private draft ring.
        """
        if not 1 <= len(requests) <= self.context.max_batch:
            raise ValueError('invalid accepted batch')
        states = [req.proposer_state for req in requests]
        if (any(not isinstance(s, DraftState) or not s.valid or s.end != req.committed_len or s.end < 0
                for s, req in zip(states, requests))
                or len({s.slot for s in states}) != len(states)
                or any(any(req.committed_len < p < req.committed_len + self.k + 1 for p in req.checkpoints) for req in requests)):
            raise ValueError('accepted rows do not follow private draft prefixes')
        return [req.committed_len for req in requests], [s.slot for s in states]

    def feedback_device(self, features, starts, ring_slots, counts, steps, *, norm=None, tokens=None, pages=None):
        """Project every verify row and scatter the first counts[i] of each request into its ring
        (the verify block's norm rows, tokens and pages are for sequential drafts; unused here).

        Static shapes for graph capture: features {lid: [b*steps, H]} verify
        outputs, starts/ring_slots/counts [b] int64 on the device. Rows past a
        request's count (and rows of padded requests, ring slot 0) are written
        to the dummy ring slot 0, whose contents only padded draft rows read.
        """
        batch = starts.shape[0]
        hidden = self.model.project_features({lid: features[lid][:batch * steps] for lid in self.model.features})
        offsets = torch.arange(steps, device=starts.device)
        positions = starts[:, None] + offsets[None, :]
        valid = offsets[None, :] < counts[:, None]
        slot = torch.where(valid, ring_slots[:, None], torch.zeros_like(ring_slots)[:, None])
        ring = positions % self.window
        rows = (slot * self.window + ring).reshape(-1)
        pages = (slot * self.pitch + ring).reshape(-1)
        self.features.view(-1, self.features.shape[-1])[rows] = hidden
        self.model.context_kv(hidden, positions.reshape(-1), self.caches, pages)

    def accepted_publish(self, requests, emitted):
        """Advance the private draft prefixes after the device feedback pass landed."""
        for req, tokens in zip(requests, emitted):
            state = req.proposer_state
            if not isinstance(state, DraftState) or not 1 <= len(tokens) <= self.k + 1 or state.end + len(tokens) != req.committed_len:
                raise ValueError('accepted rows do not follow private draft prefixes')
            state.end = req.committed_len
            if state.end in req.checkpoints:
                self._save(req, state.end)
            state.pending = None if req.finished else tokens[-1]

    def _accepted(self, req, emitted, features, row):
        count = len(emitted)
        start = req.committed_len - count
        if (self.feedback_graphs is not None and self._feedback_enabled and count in self.feedback_graphs.graphs
                and not any(start < point < req.committed_len for point in req.checkpoints)):
            self.feedback_graphs.run(req, emitted, features, row)
            return
        self.ingest_prefill(req, emitted, req.committed_len - count,
                            {'features': {lid: rows[row, :count] for lid, rows in features.layers.items()}})
        self._state(req).pending = None if req.finished else emitted[-1]

    def prepare_release(self, req):
        if isinstance(req.proposer_state, DraftState) and req.cacheable and req.reservation is None:
            # StateManager transfers the committed slot at the endpoint and
            # frees any duplicate endpoint checkpoint before radix insertion.
            self._save(req, req.committed_len, slot=req.slot)

    def release(self, req):
        if isinstance(req.proposer_state, DraftState):
            self.free_slots.append(req.proposer_state.slot)
        req.proposer_state = None

    def capture_graphs(self, log=None):
        from sharar.proposers.parallel_graphs import ParallelGraphs
        from sharar.proposers.parallel_feedback import FeedbackGraphs
        self.graphs = ParallelGraphs(self)
        self.graphs.capture(log=log)
        self.feedback_graphs = FeedbackGraphs(self)
        self.feedback_graphs.capture(log=log)
