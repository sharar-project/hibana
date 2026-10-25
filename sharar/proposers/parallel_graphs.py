"""Draft graphs per (batch bucket, depth) for parallel draft backbones and vocabulary projection: a round at depth d
drafts a block of d + 1 rows, at the pairs the verify graphs use."""
import torch

from sharar.engine import graph_set


class ParallelGraphs:
    def __init__(self, proposer, depths):
        self.proposer = proposer
        # sorted, so tensor-parallel ranks capture in one order (the 27B's head gathers inside the draft graph)
        self.sizes = {depth: tuple(buckets) for depth, buckets in sorted(depths.items())}
        self.pool = torch.cuda.graph_pool_handle()
        self.static, self.graphs = {}, {}

    def capture(self, log=None):
        import flashinfer
        from sharar.kernels.draft_selection import greedy_lattice
        p, device = self.proposer, self.proposer.context.device
        for depth, buckets in self.sizes.items():
            s = depth + 1
            for batch in buckets:
                st = dict(tokens=torch.zeros((batch, s), dtype=torch.int64, device=device),
                          positions=torch.zeros(batch * s, dtype=torch.int64, device=device),
                          pages=torch.arange(batch * s, dtype=torch.int64, device=device),
                          qo=torch.arange(0, (batch + 1) * s, s, dtype=torch.int32, device=device),
                          kv=torch.zeros(batch + 1, dtype=torch.int32, device=device),
                          indices=torch.zeros(batch * (p.window + s), dtype=torch.int32, device=device),
                          last=torch.ones(batch, dtype=torch.int32, device=device))
                # each pair its own wrapper: a graph keeps the plan (and kernel) it was captured with
                st['wrapper'] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    p.workspace, 'NHD', use_cuda_graph=True, qo_indptr_buf=st['qo'],
                    paged_kv_indptr_buf=st['kv'], paged_kv_indices_buf=st['indices'],
                    paged_kv_last_page_len_buf=st['last'])
                self.static[batch, depth] = st
                if getattr(p.model, 'uses_context_anchor', False):
                    st['anchor_features'] = torch.zeros((batch, p.model.hidden), dtype=torch.bfloat16, device=device)
                p._plan(st['wrapper'], [torch.arange(i * s, (i + 1) * s, device=device) for i in range(batch)], batch, s)
                def forward(st=st):
                    st['hidden'] = p.model.forward(st['tokens'], st['positions'], st['pages'], p.caches, st['wrapper'])[:, 1:]
                    st['candidates'], st['scores'] = p.select(st['hidden'], st['tokens'][:, 0], st.get('anchor_features'))
                self.graphs[batch, depth] = graph_set.capture(self.pool, forward)
                if log:
                    log(f'  captured {p.name} backbone/projection graph for batch {batch}, depth {depth}')
            greedy_lattice(st['candidates'], st['scores'])  # its steps are a compile-time constant: compiled here, not mid-serving

    def run(self, tokens, positions, pages, histories, *, anchor_features=None):
        p, (batch, s) = self.proposer, tokens.shape
        depth = s - 1
        if depth not in self.sizes:
            raise ValueError(f'uncaptured draft depth {depth}')
        size = graph_set.pick(self.sizes[depth], batch)  # the verify graphs' bucket for this batch
        st = self.static[size, depth]
        st['tokens'][:batch].copy_(tokens)
        st['positions'][:batch * s].copy_(positions)
        st['pages'][:batch * s].copy_(pages)
        if 'anchor_features' in st:
            st['anchor_features'].zero_()
            st['anchor_features'][:batch].copy_(anchor_features)
        padded = list(histories)
        if size > batch:
            st['tokens'][batch:].zero_()
            st['positions'][batch * s:].zero_()
            dummy = torch.arange((size - batch) * s, device=tokens.device)
            st['pages'][batch * s:].copy_(dummy)
            padded += [dummy[i * s:(i + 1) * s] for i in range(size - batch)]
        p._plan(st['wrapper'], padded, size, s)
        self.graphs[size, depth].replay()
        # Borrowed only until next replay; proposal sampling creates owned IDs/q.
        return st['hidden'][:batch], st['candidates'][:batch], st['scores'][:batch]
