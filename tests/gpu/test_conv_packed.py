"""The causal convolution's packed q/k/v output holds the same values as its (dim, tokens) output."""

import pytest
import torch


@pytest.mark.parametrize("lens", [[2048], [5, 300, 1], [1024, 3000]])
@torch.inference_mode()
def test_packed_output_equals_the_split_output(lens):
    from sharar.kernels.conv1d import causal_conv1d_fn
    torch.manual_seed(len(lens))
    key, value, width = 2048, 6144, 4
    dim, t = 2 * key + value, sum(lens)
    x = torch.randn(t, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16) * 0.3
    cache = torch.arange(len(lens), dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    initial = torch.tensor([i % 2 == 1 for i in range(len(lens))], device="cuda")

    def run(**kw):
        states = torch.randn(len(lens), dim, width - 1, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(3))
        out = causal_conv1d_fn(x.transpose(0, 1), weight, None, conv_states=states, has_initial_state=initial,
                               cache_indices=cache, query_start_loc=starts, seq_lens_cpu=lens, activation="silu", **kw)
        return out, states

    plain, plain_states = run()
    packed, packed_states = run(key_dim=key)
    rows = plain.transpose(0, 1)
    flat = packed.transpose(0, 1).view(-1)
    assert torch.equal(flat[:t * key].view(t, key), rows[:, :key])
    assert torch.equal(flat[t * key:2 * t * key].view(t, key), rows[:, key:2 * key])
    assert torch.equal(flat[2 * t * key:].view(t, value), rows[:, 2 * key:])
    assert torch.equal(packed_states, plain_states)
