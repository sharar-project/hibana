"""Bitwise RoPE/cache equivalence, including private rings and graph replay."""

import pytest
import torch



@pytest.mark.parametrize('tokens', [1, 3, 16, 31, 37, 47, 64, 2048])
@pytest.mark.parametrize('heads', [8, 32])
def test_rotation_matches_bf16_reference(tokens, heads):
    from hibana.model.parallel_draft import rotate, rotate_reference
    torch.manual_seed(726)
    x = torch.randn(tokens, heads, 128, device='cuda', dtype=torch.bfloat16)
    # Strided inputs and long-context positions exercise both address maps.
    x = torch.stack((x, x), dim=1)[:, 1]
    angles = torch.randn(32768, 64, device='cuda') * 100
    cos, sin = angles.cos() * 1.2, angles.sin() * 1.2
    positions = torch.randint(0, 32768, (tokens,), device='cuda')
    assert torch.equal(rotate(x, positions, cos, sin), rotate_reference(x, positions, cos, sin))


@pytest.mark.parametrize('tokens', [1, 3, 16, 31, 37, 47, 64, 2048])
def test_fused_scatter_preserves_other_slots_and_replays(tokens):
    from hibana.model.parallel_draft import rotate_and_cache, rotate_reference
    torch.manual_seed(127)
    kv = torch.randn(tokens, 2048, device='cuda', dtype=torch.bfloat16)
    k, v = [x.view(tokens, 8, 128) for x in kv.chunk(2, -1)]
    angles = torch.randn(32768, 64, device='cuda') * 100
    cos, sin = angles.cos() * 1.2, angles.sin() * 1.2
    positions = torch.arange(10000, 10000 + tokens, device='cuda')
    pages = positions % 2048 + 2064
    # accepted_batch uploads a [T, 3] metadata table and unbinds its columns.
    positions, _, pages = torch.stack((positions, pages, pages), dim=1).unbind(1)
    caches = [torch.full((6200, 1, 8, 128), -17., device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    reference = [c.clone() for c in caches]
    reference[0][pages, 0] = rotate_reference(k, positions, cos, sin)
    reference[1][pages, 0] = v
    rotate_and_cache(k, v, positions, cos, sin, caches, pages)
    assert all(torch.equal(a, b) for a, b in zip(caches, reference))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rotate_and_cache(k, v, positions, cos, sin, caches, pages)
    for c in caches:
        c.fill_(-17.)
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(caches, reference))
    # Captured runtime extents still obey masks with new values and ring pages.
    kv.normal_()
    positions.add_(73)
    pages.copy_(positions % 2048 + 2064)
    for c in (*caches, *reference):
        c.fill_(-17.)
    reference[0][pages, 0] = rotate_reference(k, positions, cos, sin)
    reference[1][pages, 0] = v
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(caches, reference))
