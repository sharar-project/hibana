"""Exact conditional-rank traversal, including ties and graph input lifetimes."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 kernel tests')


def reference(candidates, scores):
    # Host reference deliberately traverses the selected predecessor. An
    # independent argmax for each position is incorrect for this lattice.
    candidates, scores = candidates.cpu(), scores.float().cpu()
    output = torch.empty(candidates.shape[:2], dtype=torch.int64)
    for row in range(len(candidates)):
        previous = 0
        for step in range(candidates.shape[1]):
            previous = int(scores[row, step, previous].argmax())
            output[row, step] = candidates[row, step, previous]
    return output


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
@pytest.mark.parametrize('steps', [1, 3, 7])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('graphs', [False, True])
def test_greedy_lattice_matches_conditional_reference(batch, steps, dtype, graphs):
    from sharar.kernels.draft_selection import greedy_lattice
    torch.manual_seed(1907 + batch + steps)
    width = 16
    # Non-contiguous views exercise padded graph rows and every input stride.
    candidates = torch.randint(0, 248320, (batch * 2, steps, width * 2), device='cuda')[::2, :, ::2]
    scores = torch.randn((batch * 2, steps, width, width * 2), dtype=dtype, device='cuda')[::2, :, :, ::2]
    output = greedy_lattice(candidates, scores)
    graph = None
    if graphs:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = greedy_lattice(candidates, scores)
    for pattern in ('random', 'ties', 'negative_infinity', 'nan'):
        if pattern == 'ties':
            scores.zero_()
            scores[..., 3] = 2
            scores[..., 11] = 2
        elif pattern == 'negative_infinity':
            scores.fill_(float('-inf'))
        elif pattern == 'nan':
            scores[..., 5] = float('nan')
            scores[..., 9] = float('nan')
        if graph is not None:
            graph.replay()
        else:
            output = greedy_lattice(candidates, scores)
        assert torch.equal(output.cpu(), reference(candidates, scores)), (batch, steps, dtype, graphs, pattern)
    # The adapter returns owned IDs, safe across reuse of the borrowed lattice.
    saved = output.clone()
    candidates.zero_()
    scores.zero_()
    assert torch.equal(output, saved)
