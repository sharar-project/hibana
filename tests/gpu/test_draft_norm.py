import pytest
import torch
import os

pytestmark = pytest.mark.skipif(os.environ.get("SHARAR_RUN_GPU") != "1", reason="opt-in GB10 tests")

@pytest.mark.parametrize('tokens',[1,3,16,31,37,47,64,2048])
@pytest.mark.parametrize('heads,dim',[(1,5120),(8,128),(32,128)])
def test_exact(tokens,heads,dim):
    from sharar.kernels.draft_norm import rms_bf16 as rms
    from sharar.model.parallel_draft import rms_reference as reference
    torch.manual_seed(42)
    x=torch.randn(tokens,heads,dim,device='cuda',dtype=torch.bfloat16)
    x=torch.stack([x,x],dim=1)[:,1]
    w=torch.randn(dim,device='cuda',dtype=torch.bfloat16)
    for amplitude in (.01,1.,100.):
        value=x*amplitude
        y=rms(value,w)
        expected=reference(value,w)
        mismatch=int((y!=expected).sum())
        assert torch.equal(y,expected), (mismatch,float((y.float()-expected.float()).abs().max()))
    # Directly test the strided K/V half without materializing a scale product.
    assert torch.equal(rms(x,w),reference(x,w))


@torch.inference_mode()
def test_every_bs4_feedback_count_and_graph_input_mutation():
    from sharar.kernels.draft_norm import rms_bf16
    from sharar.model.parallel_draft import rms_reference
    torch.manual_seed(314)
    weight = torch.randn(128, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(64, 2048, device='cuda', dtype=torch.bfloat16)
    # The K view has a gap between tokens, as in the real fused KV projection.
    for count in range(1, 65):
        x = kv[:count, :1024].view(count, 8, 128)
        assert torch.equal(rms_bf16(x, weight), rms_reference(x, weight))
    x = kv[:37, :1024].view(37, 8, 128)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = rms_bf16(x, weight)
    for scale in (.01, 1., 100.):
        kv.copy_((torch.randn_like(kv.float()) * scale).bfloat16())
        graph.replay()
        assert torch.equal(actual, rms_reference(x, weight))
