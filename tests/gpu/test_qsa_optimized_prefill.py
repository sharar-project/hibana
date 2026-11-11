"""Opt-in runtime QSA path, using synthetic actual-head geometry."""
import pytest
import torch
from hibana.engine.config import EngineConfig,using
from hibana.target import FLASHNEXT

@pytest.mark.parametrize('batch,length,prefix',[(1,4096,0),(2,4096,0),(1,2048,8192),(1,257,2049)])
@pytest.mark.parametrize('backend', ['auto', 'optimized'])
@torch.inference_mode()
def test_actual_module_opt_in(batch,length,prefix,backend):
    from tests.qsa_fixture import QsaFixture
    f=QsaFixture(batch,length,prefix,hidden=256)
    gold=f.native().clone();aux=f.aux.clone()
    with using(EngineConfig(qsa_prefill_backend=backend)):
        f.indexer.configure_prefill(FLASHNEXT,f.module.rope.cos_sin)
    assert f.indexer._prefill is not None
    actual=f.native()
    assert f.ctx.extras.qsa_optimized_prefill
    torch.testing.assert_close(actual,gold,atol=.02,rtol=.02)
    assert torch.equal(f.aux,aux)
    plan=f.ctx.extras.qsa_prefill_plan
    f.native();assert f.ctx.extras.qsa_prefill_plan is plan
    # A new forward/host sequence key must rebuild metadata, not freeze it.
    f.ctx.extras.qsa_prefill_plan=None
    f.native();assert f.ctx.extras.qsa_prefill_plan is not plan


@torch.inference_mode()
@pytest.mark.parametrize('backend', ['auto', 'optimized'])
def test_runtime_capture_guard(backend):
    from tests.qsa_fixture import QsaFixture
    f=QsaFixture(1,4096,0,hidden=256)
    with using(EngineConfig(qsa_prefill_backend=backend)):
        f.indexer.configure_prefill(FLASHNEXT,f.module.rope.cos_sin)
    # Guard inspection during capture must decline the eager-only adapter.
    result=[]
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        f.x.add_(0)
        result.append(f.indexer._prefill.eligible(f.x,f.ctx,f.indexer))
    assert result==[False]


@torch.inference_mode()
def test_forced_sparse_prefill_rejects_over_limit_before_cache_write():
    from tests.qsa_fixture import QsaFixture
    f = QsaFixture(1,257,2049,hidden=256)
    with using(EngineConfig(qsa_prefill_backend='optimized')):
        f.indexer.configure_prefill(FLASHNEXT, f.module.rope.cos_sin)
    aux = f.aux.clone()
    f.ctx.extras.qsa_seqs = [(32768,257,f.seq_pages[0])]
    with pytest.raises(ValueError, match='context <=32768'):
        f.indexer.run(f.x, f.ctx, f.token_pages)
    assert torch.equal(f.aux, aux)


@torch.inference_mode()
def test_forced_backend_preserves_decode():
    from tests.qsa_fixture import QsaFixture
    f = QsaFixture(1,1,4096,hidden=256)
    f.ctx.mode = 'decode'
    f.ctx.out_pages = f.token_pages
    gold = f.native().clone()
    with using(EngineConfig(qsa_prefill_backend='optimized')):
        f.indexer.configure_prefill(FLASHNEXT, f.module.rope.cos_sin)
    actual = f.native()
    assert not f.ctx.extras.qsa_optimized_prefill
    assert torch.equal(actual, gold)
