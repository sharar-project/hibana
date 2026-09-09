import pytest
import torch
import os

pytestmark = pytest.mark.skipif(os.environ.get("SHARAR_RUN_GPU") != "1", reason="opt-in GB10 tests")

@pytest.mark.parametrize('tokens',[1,3,16,64,2048])
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
