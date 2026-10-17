from types import SimpleNamespace
import pytest
import torch
from sharar.kernels.qsa.prefill import PrefillKernels


def fake_indexer():
    from sharar.model.flashnext.indexer import FlashNextIndexer
    ix = FlashNextIndexer.__new__(FlashNextIndexer)
    ix.heads, ix.dim, ix.ratio, ix.budget, ix.rotary_dim = 4, 128, 4, 2048, 64
    ix.width = 2051
    ix._prefill = None
    ix.qk = SimpleNamespace(is_cuda=True, device='cuda')
    return ix

@pytest.mark.parametrize('mode,first,length,expected',[
    ('prefill',0,4096,True),('prefill',0,128,False),
    ('prefill',30720,2048,True),('prefill',32768,1,False),
    ('decode',4096,1,False),('verify',4096,4,False)])
def test_prefill_scope(monkeypatch,mode,first,length,expected):
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    ix=SimpleNamespace(heads=4,dim=128,ratio=4,budget=2048,rotary_dim=64,width=2051,
        qk=torch.zeros(640,8,dtype=torch.bfloat16),q_norm=torch.zeros(128,dtype=torch.bfloat16),
        k_norm=torch.zeros(128,dtype=torch.bfloat16))
    ctx=SimpleNamespace(mode=mode,extras=SimpleNamespace(qsa_seqs=[(first,length,None)]))
    x=SimpleNamespace(is_cuda=True,dtype=torch.bfloat16,shape=(length,8))
    assert PrefillKernels.eligible(x,ctx,ix)==expected


def test_tp2_configuration_enables_sharded_attention(monkeypatch):
    from sharar.engine.config import EngineConfig,using
    import sharar.distributed as distributed
    monkeypatch.setattr(distributed,'get_tp_info',lambda:SimpleNamespace(size=2))
    monkeypatch.setattr('sharar.kernels.qsa.prefill.PrefillKernels',lambda table:table)
    ix=fake_indexer()
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(12,1))
    with using(EngineConfig(qsa_prefill_backend='auto')):
        ix.configure_prefill(SimpleNamespace(num_attention_heads=12,num_kv_heads=1,head_dim=256),'rope')
    assert ix._prefill == 'rope'


def test_stable_numerics_stays_original():
    from sharar.engine.config import EngineConfig,using
    ix=fake_indexer()
    with using(EngineConfig(qsa_prefill_backend='auto',stable_numerics=True)):
        ix.configure_prefill(SimpleNamespace(num_attention_heads=24,num_kv_heads=2,head_dim=256),None)
    assert ix._prefill is None


def test_replicated_tp2_draft_stays_original(monkeypatch):
    from sharar.engine.config import EngineConfig,using
    import sharar.distributed as distributed
    monkeypatch.setattr(distributed,'get_tp_info',lambda:SimpleNamespace(size=2))
    ix=fake_indexer()
    with using(EngineConfig(qsa_prefill_backend='auto')):
        ix.configure_prefill(SimpleNamespace(num_attention_heads=24,num_kv_heads=2,head_dim=256),None)
    assert ix._prefill is None


@pytest.mark.parametrize('backend', ['auto', 'optimized'])
def test_supported_backend_initializes_adapter(monkeypatch, backend):
    from sharar.engine.config import EngineConfig, using
    import sharar.distributed as distributed
    monkeypatch.setattr(distributed, 'get_tp_info', lambda: SimpleNamespace(size=1))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 1))
    adapter = object()
    monkeypatch.setattr('sharar.kernels.qsa.prefill.PrefillKernels', lambda table: adapter)
    ix = fake_indexer()
    with using(EngineConfig(qsa_prefill_backend=backend)):
        ix.configure_prefill(SimpleNamespace(num_attention_heads=24,num_kv_heads=2,head_dim=256), None)
    assert ix._prefill is adapter


@pytest.mark.parametrize('reason', ['hardware', 'stable', 'geometry'])
@pytest.mark.parametrize('backend', ['auto', 'optimized'])
def test_unsupported_configuration_fallback_or_error(monkeypatch, reason, backend):
    from sharar.engine.config import EngineConfig, using
    import sharar.distributed as distributed
    monkeypatch.setattr(distributed, 'get_tp_info', lambda: SimpleNamespace(size=1))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0) if reason == 'hardware' else (12, 1))
    ix = fake_indexer()
    if reason == 'geometry':
        ix.dim = 64
    target = SimpleNamespace(num_attention_heads=24, num_kv_heads=2, head_dim=256)
    with using(EngineConfig(qsa_prefill_backend=backend, stable_numerics=reason == 'stable')):
        if backend == 'optimized':
            with pytest.raises(ValueError, match='SHARAR_QSA_PREFILL_BACKEND=optimized'):
                ix.configure_prefill(target, None)
        else:
            ix.configure_prefill(target, None)
    assert ix._prefill is None


def test_original_clears_adapter_without_gpu_probe(monkeypatch):
    from sharar.engine.config import EngineConfig, using
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: pytest.fail('GPU probe'))
    ix = fake_indexer()
    ix._prefill = object()
    with using(EngineConfig(qsa_prefill_backend='original')):
        ix.configure_prefill(None, None)
    assert ix._prefill is None


def test_forced_backend_excludes_replicated_draft():
    from sharar.engine.config import EngineConfig, using
    ix = fake_indexer()
    with using(EngineConfig(qsa_prefill_backend='optimized')):
        ix.configure_prefill(None, None, sharded=False)
    assert ix._prefill is None and ix._prefill_backend == 'original'


@pytest.mark.parametrize('backend', ['auto', 'optimized'])
def test_compile_error_is_not_hidden(monkeypatch, backend):
    from sharar.engine.config import EngineConfig, using
    import sharar.distributed as distributed
    monkeypatch.setattr(distributed, 'get_tp_info', lambda: SimpleNamespace(size=1))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 1))
    def failed(table):
        raise RuntimeError('compile failed')
    monkeypatch.setattr('sharar.kernels.qsa.prefill.PrefillKernels', failed)
    ix = fake_indexer()
    with using(EngineConfig(qsa_prefill_backend=backend)):
        with pytest.raises(RuntimeError, match='compile failed'):
            ix.configure_prefill(SimpleNamespace(num_attention_heads=24,num_kv_heads=2,head_dim=256), None)


@pytest.mark.parametrize('mode,first,capturing,reject', [
    ('prefill',32768,False,True), ('prefill',0,False,False),
    ('decode',32768,False,False), ('verify',32768,False,False),
    ('prefill',32768,True,False)])
@pytest.mark.parametrize('backend', ['auto', 'optimized'])
def test_force_applies_only_to_sparse_eager_scope(monkeypatch, mode, first, capturing, reject, backend):
    ix = fake_indexer()
    ix.name = 'qsa.test'
    ix.qk = torch.zeros(640, 8, dtype=torch.bfloat16)
    ix.q_norm = ix.k_norm = torch.zeros(128, dtype=torch.bfloat16)
    ix._prefill = PrefillKernels.__new__(PrefillKernels)
    ix._prefill_backend = backend
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: capturing)
    called = []
    def original(*args):
        called.append(True)
        raise RuntimeError('original path reached')
    monkeypatch.setattr('sharar.model.flashnext.indexer.F.linear', original)
    ctx = SimpleNamespace(mode=mode, extras=SimpleNamespace(qsa_seqs=[(first,1,None)]),
                          kv_pool=SimpleNamespace(page_aux={ix.name: object()}))
    x = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, shape=(1,8))
    if backend == 'optimized' and reject:
        with pytest.raises(ValueError, match='context <=32768'):
            ix.run(x, ctx, None)
        assert not called
    else:
        with pytest.raises(RuntimeError, match='original path reached'):
            ix.run(x, ctx, None)
        assert called


@pytest.mark.parametrize('backend', ['auto', 'optimized'])
def test_launch_error_is_not_hidden(backend):
    ix = fake_indexer()
    ix._prefill_backend = backend
    def failed(*args):
        raise RuntimeError('launch failed')
    ix._prefill = SimpleNamespace(eligible=lambda *args: True, run=failed)
    with pytest.raises(RuntimeError, match='launch failed'):
        ix.run(None, SimpleNamespace(extras=None), None)
