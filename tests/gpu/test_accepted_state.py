"""Bitwise reconstruction against the actual verify recurrence, all commits."""
from types import SimpleNamespace
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 kernel tests')


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
@pytest.mark.parametrize('steps', [1, 2, 4, 5, 8, 16])
@pytest.mark.parametrize('graphs', [False, True])
def test_reconstruct_every_accepted_prefix(batch, steps, graphs):
    from sharar.kernels.fla.accepted_state import AcceptedState, reconstruct
    from sharar.kernels.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
    torch.manual_seed(1907 + batch + steps)
    layers, h, hv, k, v = 2, 16, 48, 128, 128
    state = torch.randn((layers, 7, hv, v, k), device='cuda') * .1
    initial = state.clone()
    pool = SimpleNamespace(ssm_all=state, layer_ids=list(range(layers)))
    replay = AcceptedState(pool, 4, steps, h)
    retained = (replay.raw_k, replay.raw_v, replay.gates, replay.betas)
    assert replay.bytes == sum(t.numel() * t.element_size() for t in retained)
    assert not hasattr(replay, 'final')
    final = torch.empty((layers, 4, hv, v, k), device='cuda')  # test-only old output layout
    full = torch.empty((layers, batch, steps, hv, v, k), device='cuda')
    slots = torch.tensor([6, 2, 5, 1][:batch], device='cuda', dtype=torch.int32)
    scratch = torch.arange(batch, device='cuda', dtype=torch.int32)
    cu = torch.arange(0, (batch + 1) * steps, steps, device='cuda', dtype=torch.int32)
    inputs = []
    for _ in range(layers):
        rand = lambda shape: torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        inputs.append(dict(A_log=torch.randn(hv, device='cuda'), dt_bias=rand((hv,)),
                           softplus_beta=1., softplus_threshold=20.,
                           q=rand((1, batch * steps, h, k)), k=rand((1, batch * steps, h, k)),
                           v=rand((1, batch * steps, hv, v)),
                           a=rand((batch * steps, hv)), b=rand((batch * steps, hv)),
                           initial_state_indices=slots, cu_seqlens=cu,
                           intermediate_state_indices=scratch, disable_state_update=True,
                           use_qk_l2norm_in_kernel=True))
    expected, actual = [], []
    def verify():
        actual.clear()
        for layer in range(layers):
            actual.append(fused_sigmoid_gating_delta_rule_update(
                **inputs[layer], initial_state_source=state[layer], ring_uses_scratch=True,
                **replay.layer_inputs[layer]))
    for layer in range(layers):
        expected.append(fused_sigmoid_gating_delta_rule_update(
            **inputs[layer], initial_state_source=state[layer], intermediate_states_buffer=full[layer]))
    verify()
    if graphs:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verify()
        graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(state, initial)
    for count in range(1, steps + 1):
        counts = torch.tensor([1 + (count - 1 + row) % steps for row in range(batch)], dtype=torch.int32, device='cuda')
        state.copy_(initial)
        reconstruct(state, replay.raw_k, replay.raw_v, replay.gates, replay.betas, slots, counts, final)
        target = torch.stack([full[:, row, int(counts[row]) - 1] for row in range(batch)], dim=1)
        assert torch.equal(final[:, :batch], target), (batch, steps, graphs, count, (final[:, :batch] - target).abs().max().item())
        assert torch.equal(state, initial)
        replay.run(pool, slots, counts)
        if graphs:
            state.copy_(initial)
            commit_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(commit_graph):
                replay.run(pool, slots, counts)
            state.copy_(initial)
            commit_graph.replay()
        expected_state = initial.clone()
        expected_state[:, slots.long()] = target
        # Includes the dummy slot and every unselected neighbor; tails/padding
        # must never become destinations, even when batch < scratch capacity.
        assert torch.equal(state, expected_state), (batch, steps, graphs, count)
