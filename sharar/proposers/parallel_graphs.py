"""Fixed batch buckets for parallel draft backbones and vocabulary projection."""
import torch

from sharar.engine import graph_set
from sharar.engine.graphs import capture_sizes


class ParallelGraphs:
    def __init__(self, proposer):
        self.proposer = proposer
        self.sizes = capture_sizes(proposer.context.max_batch)
        self.pool = torch.cuda.graph_pool_handle()
        self.static, self.graphs = {}, {}

    def capture(self, log=None):
        import flashinfer
        p, device, s = self.proposer, self.proposer.context.device, self.proposer.noise_steps
        for batch in self.sizes:
            st = dict(tokens=torch.zeros((batch, s), dtype=torch.int64, device=device),
                      positions=torch.zeros(batch * s, dtype=torch.int64, device=device),
                      pages=torch.arange(batch * s, dtype=torch.int64, device=device),
                      qo=torch.arange(0, (batch + 1) * s, s, dtype=torch.int32, device=device),
                      kv=torch.zeros(batch + 1, dtype=torch.int32, device=device),
                      indices=torch.zeros(batch * (p.window + s), dtype=torch.int32, device=device),
                      last=torch.ones(batch, dtype=torch.int32, device=device))
            st['wrapper'] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                p.workspace, 'NHD', use_cuda_graph=True, qo_indptr_buf=st['qo'],
                paged_kv_indptr_buf=st['kv'], paged_kv_indices_buf=st['indices'],
                paged_kv_last_page_len_buf=st['last'])
            self.static[batch] = st
            p._plan(st['wrapper'], [torch.arange(i * s, (i + 1) * s, device=device) for i in range(batch)], batch)
            def forward():
                hidden = p.model.forward(st['tokens'], st['positions'], st['pages'], p.caches, st['wrapper'])
                st['hidden'] = hidden[:, 1:] if p.name == 'dflash2' else hidden
                if p.name == 'dflash2':
                    st['candidates'], st['scores'] = p.model.selector(st['hidden'], st['tokens'][:, 0])
                else:
                    st['logits'] = p.model.draft_head.logits(hidden.reshape(-1, 5120)).view(batch, p.k, -1)
            self.graphs[batch] = graph_set.capture(self.pool, forward)
            if log:
                log(f'  captured {p.name} backbone/projection graph for batch {batch}')

    def run(self, tokens, positions, pages, histories):
        p, batch, s = self.proposer, len(tokens), self.proposer.noise_steps
        size = next(n for n in self.sizes if n >= batch)
        st = self.static[size]
        st['tokens'][:batch].copy_(tokens)
        st['positions'][:batch * s].copy_(positions)
        st['pages'][:batch * s].copy_(pages)
        padded = list(histories)
        if size > batch:
            st['tokens'][batch:].zero_()
            st['positions'][batch * s:].zero_()
            dummy = torch.arange((size - batch) * s, device=tokens.device)
            st['pages'][batch * s:].copy_(dummy)
            padded += [dummy[i * s:(i + 1) * s] for i in range(size - batch)]
        p._plan(st['wrapper'], padded, size)
        self.graphs[size].replay()
        # Borrowed only until next replay; proposal sampling creates owned IDs/q.
        return (st['hidden'][:batch],
                st['candidates'][:batch] if p.name == 'dflash2' else None,
                st['scores'][:batch] if p.name == 'dflash2' else None,
                st['logits'][:batch] if p.name == 'dspark' else None)
