# SPDX-License-Identifier: Apache-2.0
"""Optimized TP1/TP2 eager-prefill adapter; plans belong to one forward."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .compression import load_compressor
from .maintenance_plan import make_compression_plan
from .query_prep import load_query_prep
from .topk_pages import load_topk_pages


@dataclass
class ScorePlan:
    table: torch.Tensor
    indptr: torch.Tensor
    visible: torch.Tensor
    max_query_len: int


class PrefillKernels:
    def __init__(self, cos_sin):
        # Constructed only for opt-in SM121. Native compilation is outside capture.
        self.query = load_query_prep()
        self.compress = load_compressor()
        self.topk = load_topk_pages()
        self.cos_sin = cos_sin

    @staticmethod
    def in_scope(x, ctx, indexer):
        """Only sparse eager target prefill is governed by backend selection."""
        seqs = getattr(ctx.extras, 'qsa_seqs', None)
        return (getattr(ctx, 'mode', None) == 'prefill' and bool(seqs)
                and any(first+n-1 >= indexer.width for first,n,_ in seqs)
                and not (x.is_cuda and torch.cuda.is_current_stream_capturing()))

    @staticmethod
    def eligible(x, ctx, indexer):
        seqs = getattr(ctx.extras, 'qsa_seqs', None)
        return (x.is_cuda and x.dtype == torch.bfloat16
                and PrefillKernels.in_scope(x, ctx, indexer)
                and (indexer.heads,indexer.dim,indexer.ratio,indexer.budget,indexer.rotary_dim) == (4,128,4,2048,64)
                and all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in (indexer.qk,indexer.q_norm,indexer.k_norm))
                and sum(n for _,n,_ in seqs) == x.shape[0]
                and all(first >= 0 and n >= 0 and first+n <= 32768 for first,n,_ in seqs))

    def run(self, indexer, x, ctx, token_pages):
        from .tiled_scaled import score
        side=ctx.extras;seqs=side.qsa_seqs
        # Host metadata and data_ptr only: never read GPU counts/positions to dispatch.
        key=tuple((first,n,pages.data_ptr(),len(pages)) for first,n,pages in seqs)+(ctx.positions.data_ptr(),)
        cached=getattr(side,'qsa_prefill_plan',None)
        if cached is None or cached[0] != key:
            compression=make_compression_plan(seqs,x.device)
            gcap=max((first+n)//4 for first,n,_ in seqs)
            table=torch.zeros(len(seqs),gcap,device=x.device,dtype=torch.int32)
            offsets=[0];bases=[];page_offset=0
            for i,(first,n,pages) in enumerate(seqs):
                groups=(first+n)//4
                table[i,:groups]=pages[3:groups*4:4]
                offsets.append(offsets[-1]+n)
                bases.extend([page_offset]*n)
                page_offset+=len(pages)
            plan=ScorePlan(table,torch.tensor(offsets,device=x.device,dtype=torch.int32),
                ((ctx.positions+1)//4).to(torch.int32),max(n for _,n,_ in seqs))
            flat=torch.cat([pages for _,_,pages in seqs]).to(torch.int32)
            base=torch.tensor(bases,device=x.device,dtype=torch.int64)
            cached=(key,compression,plan,flat,base)
            side.qsa_prefill_plan=cached
        _,compression,plan,flat,base=cached
        aux=ctx.kv_pool.page_aux[indexer.name]
        projected=F.linear(x,indexer.qk)
        query=torch.empty(x.shape[0],4,128,device=x.device,dtype=torch.bfloat16)
        self.query(projected,aux,token_pages.to(torch.int32).contiguous(),ctx.positions,
                   indexer.q_norm,self.cos_sin,query,indexer.eps)
        self.compress(aux,compression,indexer.k_norm,self.cos_sin,indexer.eps)
        pages=torch.empty(x.shape[0],2051,device=x.device,dtype=torch.int32)
        counts=torch.empty(x.shape[0],device=x.device,dtype=torch.int32)
        chunk=1024 if plan.table.shape[1] <= 2048 else 256
        for start in range(0,x.shape[0],chunk):
            end=min(start+chunk,x.shape[0])
            scores=score(query,aux,plan,start,end-start,config=(64,4,4))
            self.topk(scores,flat,base[start:end],ctx.positions[start:end],out=(pages[start:end],counts[start:end]))
        side.qsa_optimized_prefill=True
        return pages,counts
