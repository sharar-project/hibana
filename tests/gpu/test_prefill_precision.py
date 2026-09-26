"""Mixed-prefix prefill must retain new BF16 K/V and sequence boundaries."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 tests')


@pytest.mark.parametrize('prefixes', [(0,0), (17,0), (0,31), (17,31)])
@torch.inference_mode()
def test_mixed_prefix_attention_matches_dense_reference(prefixes):
    import flashinfer
    from sharar.engine.context import ForwardContext
    from sharar.kernels.ops import RaggedAttention
    from sharar.cache.kv_pool import KvPool
    from sharar.model.attention import Attention

    torch.manual_seed(317)
    device=torch.device('cuda')
    lengths=[33,65];total=sum(lengths);hq,hk,dim=24,4,256
    pool=KvPool(300,device,layer_ids=[3])
    q=torch.randn(total,hq,dim,device=device,dtype=torch.bfloat16)*.25
    k=torch.randn(total,hk,dim,device=device,dtype=torch.bfloat16)*.25
    v=torch.randn_like(k)
    prefix_k=[torch.randn(n,hk,dim,device=device,dtype=torch.bfloat16)*.25 for n in prefixes]
    prefix_v=[torch.randn_like(x) for x in prefix_k]
    prefix_pages=[pool.alloc(n) for n in prefixes]
    for pages,pk,pv in zip(prefix_pages,prefix_k,prefix_v):pool.write(3,pages,pk,pv)
    pages=torch.cat([pool.alloc(n) for n in lengths])
    ctx=ForwardContext(mode='prefill',kv_pool=pool,state_pool=None,
        positions=torch.cat([torch.arange(p,p+n,device=device) for p,n in zip(prefixes,lengths)]),
        pages=pages,seq_len=total,cu_seqlens=torch.tensor([0,33,total],device=device,dtype=torch.int32),
        seq_lens=lengths,seq_prefix_lens=list(prefixes))
    workspace=torch.empty(128*2**20,device=device,dtype=torch.uint8)
    if any(prefixes):
        wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace,'NHD')
        wrapper.plan(torch.tensor([0,33,total],dtype=torch.int32),
            torch.tensor([0,prefixes[0],sum(prefixes)],dtype=torch.int32),torch.cat(prefix_pages),
            torch.ones(2,dtype=torch.int32),hq,hk,dim,1,causal=False,
            q_data_type=torch.bfloat16,kv_data_type=pool.dtype,sm_scale=dim**-.5)
        ctx.paged_prefill_wrapper=wrapper
    layer=Attention.__new__(Attention)
    layer.layer_id=3;layer.q_size=hq*dim
    layer.ragged=RaggedAttention(device,hq,hk,dim)
    actual=layer._prefill_attention(q,k,v,ctx).view(total,hq,dim)
    start=0
    for n,p,pk,pv in zip(lengths,prefixes,prefix_k,prefix_v):
        # Only cached values have the FP8 boundary. New keys/values remain BF16.
        keys=torch.cat([pk.to(pool.dtype).float(),k[start:start+n].float()]).repeat_interleave(hq//hk,dim=1)
        vals=torch.cat([pv.to(pool.dtype).float(),v[start:start+n].float()]).repeat_interleave(hq//hk,dim=1)
        scores=torch.einsum('thd,shd->hts',q[start:start+n].float(),keys)*dim**-.5
        mask=torch.arange(p+n,device=device)[None,:] <= p+torch.arange(n,device=device)[:,None]
        scores.masked_fill_(~mask,-float('inf'))
        expected=torch.einsum('hts,shd->thd',scores.softmax(-1),vals)
        delta=actual[start:start+n].float()-expected
        relative_rms=delta.square().mean().sqrt()/expected.square().mean().sqrt()
        assert relative_rms < .0035, relative_rms.item()
        assert torch.isfinite(actual[start:start+n]).all()
        start+=n
    assert torch.equal(pool.k[3][pages,0].view(torch.uint8),k.to(pool.dtype).view(torch.uint8))
    assert torch.equal(pool.v[3][pages,0].view(torch.uint8),v.to(pool.dtype).view(torch.uint8))
    for pp,pk,pv in zip(prefix_pages,prefix_k,prefix_v):
        assert torch.equal(pool.k[3][pp,0].view(torch.uint8),pk.to(pool.dtype).view(torch.uint8))
        assert torch.equal(pool.v[3][pp,0].view(torch.uint8),pv.to(pool.dtype).view(torch.uint8))
