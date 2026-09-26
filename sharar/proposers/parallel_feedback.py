"""Exact-count graphs for single-request accepted-feature projection and KV."""
import torch

from sharar.engine import graph_set


class FeedbackGraphs:
    def __init__(self, proposer):
        self.proposer = proposer
        self.pool = torch.cuda.graph_pool_handle()
        self.features = {lid: torch.zeros((proposer.k + 1, 5120), dtype=torch.bfloat16,
                                          device=proposer.context.device)
                         for lid in proposer.model.features}
        self.metadata = torch.zeros(2, dtype=torch.int64, device=proposer.context.device)
        self.graphs, self.steps = {}, {}

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size()
                   for t in (*self.features.values(), self.metadata, *self.steps.values()))

    def capture(self, log=None):
        p = self.proposer
        for count in range(1, min(p.k + 1, p.window) + 1):
            # These are graph inputs, allocated outside the private graph pool.
            # Retain their storage for the lifetime of each captured graph.
            steps = self.steps[count] = torch.arange(count, dtype=torch.int64, device=p.context.device)

            def forward():
                hidden = p.model.project_features({lid: rows[:count] for lid, rows in self.features.items()})
                positions = steps + self.metadata[0]
                ring = positions % p.window
                rows = self.metadata[1] * p.window + ring
                pages = self.metadata[1] * p.pitch + ring
                p.features.view(-1, 5120)[rows] = hidden
                p.model.context_kv(hidden, positions, p.caches, pages)

            self.graphs[count] = graph_set.capture(self.pool, forward)
        if log:
            log(f'  captured exact-count draft feedback graphs: {len(self.graphs)}')

    def run(self, req, emitted, features, row):
        p, count = self.proposer, len(emitted)
        start, state = req.committed_len - count, req.proposer_state
        for lid, buffer in self.features.items():
            buffer[:count].copy_(features.layers[lid][row, :count])
        # A fresh pinned allocation keeps queued metadata alive independently
        # of later host writes. All copies and replay use the caller's stream.
        metadata = torch.tensor([start, state.slot], dtype=torch.int64, pin_memory=True)
        self.metadata.copy_(metadata, non_blocking=True)
        self.graphs[count].replay()
        state.end = req.committed_len
        if state.end in req.checkpoints:
            p._save(req, state.end)
        state.pending = None if req.finished else emitted[-1]
