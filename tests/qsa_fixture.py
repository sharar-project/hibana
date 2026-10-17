"""Shape-faithful synthetic full-QSA fixture; all serving kernels unchanged."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from sharar.cache.kv_pool import KvPool
from sharar.model.flashnext.attention import FlashNextAttention
from sharar.model.flashnext.indexer import FlashNextIndexer
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.norm import GemmaRMSNorm
from sharar.model.rope import PartialRotary
from sharar.target import FLASHNEXT


class QsaFixture:
    """Actual module/head geometry with small synthetic projection weights."""
    def __init__(self,batch=1,length=8192,prefix=0,*,seed=937,hidden=None,budget=None):
        target=FLASHNEXT
        hidden=target.hidden_size if hidden is None else hidden
        budget=target.indexer_budget if budget is None else budget
        if min(batch,length)<=0 or prefix<0 or prefix+length<=budget+target.indexer_compress_ratio-1:
            raise ValueError("baseline requires nonempty sequences entering the sparse path")
        if batch*length*target.num_kv_heads>65535:
            raise ValueError("existing sparse-attention grid Y exceeds 65535; no baseline launch")
        torch.manual_seed(seed)
        self.batch,self.length,self.prefix,self.total=batch,length,prefix,prefix+length
        self.rows=batch*length
        dev=torch.device("cuda")
        def rand(shape,scale=1.):
            return torch.randn(shape,device=dev,dtype=torch.bfloat16)*scale
        m=FlashNextAttention.__new__(FlashNextAttention)
        m.layer_id=target.attention_layer_ids[0]
        m.eps=target.rms_norm_eps
        m.num_heads,m.num_kv_heads,m.head_dim=target.num_attention_heads,target.num_kv_heads,target.head_dim
        m.q_size=m.num_heads*m.head_dim;m.kv_size=m.num_kv_heads*m.head_dim
        m.qkv=TrunkLinear.from_bf16(rand((2*m.q_size+2*m.kv_size,hidden),hidden**-.5),"fp8")
        m.o=TrunkLinear.from_bf16(rand((hidden,m.q_size),m.q_size**-.5),"fp8")
        m.q_norm=GemmaRMSNorm(rand((m.head_dim,),.1),m.eps)
        m.k_norm=GemmaRMSNorm(rand((m.head_dim,),.1),m.eps)
        m.rope=PartialRotary(m.head_dim,target.rotary_dim,target.rope_theta,self.total+4,dev)
        # The fixture always enters the sparse path; dense attention is not in scope.
        m.ragged=None
        ix=FlashNextIndexer.__new__(FlashNextIndexer)
        ix.name=f"qsa.{m.layer_id}"
        ix.heads,ix.dim=target.indexer_n_heads,target.indexer_head_dim
        ix.ratio,ix.budget=target.indexer_compress_ratio,budget
        ix.width=budget+ix.ratio-1
        ix.eps,ix.rotary_dim,ix.theta=m.eps,target.rotary_dim,target.rope_theta
        ix.qk=rand(((ix.heads+1)*ix.dim,hidden),hidden**-.5)
        ix.q_norm,ix.k_norm=rand((ix.dim,),.1),rand((ix.dim,),.1)
        m.indexer=ix
        self.module,self.indexer=m,ix
        self.pool=KvPool(batch*self.total+1,dev,layer_ids=[m.layer_id],target=target)
        ix.register(self.pool)
        perm=(torch.randperm(batch*self.total,device=dev)+1).to(torch.int32)
        self.seq_pages=list(perm.reshape(batch,self.total).unbind(0))
        self.token_pages=torch.cat([p[prefix:] for p in self.seq_pages])
        self.x=rand((self.rows,hidden))
        positions=torch.arange(prefix,self.total,device=dev,dtype=torch.int64).repeat(batch)
        self.seqs=[(prefix,length,p) for p in self.seq_pages]
        self.ctx=SimpleNamespace(mode="prefill",positions=positions,pages=self.token_pages,
                                 kv_pool=self.pool,extras=SimpleNamespace(qsa_seqs=self.seqs))
        self.aux=self.pool.page_aux[ix.name]
        if prefix:
            px=rand((batch*prefix,hidden))
            pp=torch.cat([p[:prefix] for p in self.seq_pages])
            pos=torch.arange(prefix,device=dev,dtype=torch.int64).repeat(batch)
            q,k,v,_=m._qkv(px,pos)
            self.pool.write(m.layer_id,pp,k,v)
            proj=F.linear(px,ix.qk)
            self.aux[pp.long(),0]=proj[:,ix.heads*ix.dim:]
            ix._compress(self.aux,[(0,prefix,p[:prefix]) for p in self.seq_pages],dev,px.dtype)
            del px,q,k,v,proj

    def native(self):
        return self.module.forward(self.x,self.ctx)
