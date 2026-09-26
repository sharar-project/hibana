"""Bitwise verification convolution, state ownership and capacity-stride tests."""

import pytest
import torch



@pytest.mark.parametrize('batch,steps', [(1, 2), (1, 4), (1, 8), (1, 16), (3, 4), (4, 16)])
@pytest.mark.parametrize('amplitude', [.001, 1., 100.])
@pytest.mark.parametrize('activation', [False, True])
def test_convolution_and_candidate_windows_are_bitwise_identical(batch, steps, amplitude, activation):
    from sharar.kernels.verify_conv import verify_convolve
    from sharar.kernels.conv1d import causal_conv1d_update
    torch.manual_seed(314)
    dim = 10240
    x = (torch.randn(batch * steps, dim, device='cuda') * amplitude).bfloat16()
    state = torch.randn(8, dim, 3, device='cuda', dtype=torch.bfloat16)
    state_before = state.clone()
    slots = torch.tensor([5, 2, 6, 1][:batch], dtype=torch.int32, device='cuda')
    weight = torch.randn(dim, 8, device='cuda', dtype=torch.bfloat16)[:, ::2]
    windows = torch.full((4, 16, dim, 3), -17., device='cuda', dtype=torch.bfloat16)
    expected_windows = windows.clone()
    history = state[slots].transpose(1, 2)
    xs = torch.cat((history, x.view(batch, steps, dim)), 1)
    reference_windows = windows.clone()
    expected = causal_conv1d_update(
        x.view(batch,steps,dim).transpose(1,2), state.clone(), weight,
        activation='silu' if activation else None, conv_state_indices=slots,
        intermediate_conv_window=reference_windows,
        intermediate_state_indices=torch.arange(batch,device=x.device,dtype=torch.int32),
    ).transpose(1,2)
    expected_windows[:batch, :steps] = torch.stack(
        [xs[:, s+1:s+4].transpose(1, 2) for s in range(steps)], 1)
    assert torch.equal(reference_windows, expected_windows)
    actual = verify_convolve(x, state, weight, slots, windows, steps, activation=activation)
    assert torch.equal(actual.view(torch.int16), expected.reshape_as(actual).view(torch.int16))
    assert torch.equal(windows, expected_windows)
    assert torch.equal(state, state_before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = verify_convolve(x, state, weight, slots, windows, steps, activation=activation)
    windows.fill_(-17.)
    graph.replay()
    assert torch.equal(actual.view(torch.int16), expected.reshape_as(actual).view(torch.int16))
    assert torch.equal(windows, expected_windows)
    assert torch.equal(state, state_before)
