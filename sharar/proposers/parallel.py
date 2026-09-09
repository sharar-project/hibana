"""DSpark/DFlash2 adapters with bounded context KV and feature restoration.

Target pages and recurrent state remain owned by StateManager. This adapter
owns a separate, fixed-size draft ring per active request and a byte-bounded
LRU of projected features. Radix entries contain identity-keyed handles only.
"""
from collections import OrderedDict
from dataclasses import dataclass

import torch

from sharar import config
from sharar.engine.contracts import FeatureRequest, ProposerRequirements
from sharar.engine.rejection import Proposal, distribution

PINS = {
    'dspark': ('RadixArk/Qwen3.8-27B-DSpark', 'b9a5dbdf03bc999c6c73c426b19c2d9041cea393'),
    'dflash2': ('incoai/Qwen3.8-27B-DFlash2', 'dedf8df68adfb1afeaf7b7480c0a0243108177b4'),
}


class FeatureCache:
    def __init__(self, capacity):
        self.capacity, self.bytes, self.serial = capacity, 0, 0
        self.entries = OrderedDict()

    def put(self, start, end, features):
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
        self.noise_steps = k if name == 'dspark' else k + 1
        self.window = model.window
        self.pitch = self.window + self.noise_steps
        self.free_slots = list(range(context.max_batch, 0, -1))
        self.cache_identity = (config.MODEL_REVISION, *PINS[name], model.features, 'projected-bf16-kv-v1', self.window)
        device = context.device
        shape = ((context.max_batch + 1) * self.pitch, 1, 8, 128)
        self.caches = [(torch.empty(shape, dtype=torch.bfloat16, device=device), torch.empty(shape, dtype=torch.bfloat16, device=device)) for _ in range(5)]
        self.features = torch.empty((context.max_batch + 1, self.window, 5120), dtype=torch.bfloat16, device=device)
        self.feature_cache = FeatureCache(context.max_batch * self.window * 5120 * 2)
        self.requirements = ProposerRequirements(name, FeatureRequest(layers=model.features),
                                                persistent_bytes=self.bytes, greedy_only=False, max_depth=15 if name == 'dflash2' else 7)
        self.workspace = torch.empty(128 * 2**20, dtype=torch.uint8, device=device)
        self.attention = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, 'NHD')
        self.graphs = None
        self.cache_hits = self.cache_fallbacks = 0

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

    def _save(self, req, position):
        state = self._state(req)
        if not state.valid or not req.cache_enabled:
            return
        req.cache_auxiliary = {p: value for p, value in req.cache_auxiliary.items() if p in req.checkpoints}
        start = max(0, position - self.window)
        ring = torch.arange(start, position, device=self.context.device) % self.window
        handle = self.feature_cache.put(start, position, self.features[state.slot, ring])
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

        def import_(payload):
            start, end, features = payload  # features is a view into the tier's arena; put() clones it
            return self.feature_cache.put(start, end, features)

        return self.cache_identity, export, import_

    def _plan(self, wrapper, histories, batch):
        s = self.noise_steps
        lengths = [len(p) for p in histories]
        indptr = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
        wrapper.plan(torch.arange(0, (batch + 1) * s, s, dtype=torch.int32), indptr,
                     torch.cat(histories).to(torch.int32), torch.ones(batch, dtype=torch.int32),
                     32, 8, 128, 1, causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                     sm_scale=128**-.5, window_left=self.window - 1 if self.name == 'dflash2' else -1)

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
            hidden, candidates, scores, base_logits = self.graphs.run(tokens, positions, torch.cat(write), histories)
        else:
            self._plan(self.attention, histories, batch)
            hidden = self.model.forward(tokens, positions, torch.cat(write), self.caches, self.attention)
            if self.name == 'dflash2':
                hidden = hidden[:, 1:]
                candidates, scores = self.model.selector(hidden, inputs.tokens)
            else:
                base_logits = self.model.target.logits(hidden.reshape(-1, 5120)).view(batch, self.k, -1)
        sampled = any(not r.params.greedy for r in requests)
        if self.name == 'dflash2' and not sampled:
            from sharar.kernels.draft_selection import greedy_lattice
            return Proposal(greedy_lattice(candidates, scores))
        all_tokens, all_q = [], []
        for row, req in enumerate(requests):
            # The lattice walk is sequential, but every step stays on the
            # device: indices are 1-element tensors used with index_select, so
            # no step converts a CUDA scalar to a Python int (a host sync).
            previous = inputs.tokens[row:row + 1]
            local = torch.zeros(1, dtype=torch.int64, device=device)
            chosen, laws = [], []
            for j in range(self.k):
                if self.name == 'dflash2':
                    logits = scores[row, j].index_select(0, local)[0]
                else:
                    logits = self.model.markov(base_logits[row:row + 1, j], previous, hidden[row:row + 1, j])[0]
                law = distribution(logits, req.params)
                index = law.argmax().reshape(1) if req.params.greedy else torch.multinomial(law, 1, generator=req.generator)
                token = candidates[row, j].index_select(0, index)[0] if self.name == 'dflash2' else index[0]
                previous = token.reshape(1)
                local = index
                chosen.append(token)
                if sampled:
                    laws.append(law)
            all_tokens.append(torch.stack(chosen))
            if sampled:
                all_q.append(torch.stack(laws))
        return Proposal(torch.stack(all_tokens), torch.stack(all_q) if sampled else None,
                        candidates if sampled and self.name == 'dflash2' else None)

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
            self._save(req, req.committed_len)

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
