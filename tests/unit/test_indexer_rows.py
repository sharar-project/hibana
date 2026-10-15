# SPDX-License-Identifier: Apache-2.0
"""FlashNextIndexer.static_rows: the batched index build equals the per-sequence rule it replaced."""
import random

import pytest
import torch

from sharar.model.flashnext.indexer import FlashNextIndexer


def per_sequence(seqs, ratio, device):
    members, writes = [], []
    back = torch.arange(ratio - 1, -1, -1, device=device)
    for first, n, pages in seqs:
        pos = torch.arange(first, first + n, device=device)
        members.append(pages[(pos[:, None] - back).clamp_min(0)])
        writes.append(torch.where(pos % ratio == ratio - 1, pages[pos], torch.zeros_like(pages[pos])))
    return torch.cat(members).to(torch.int32), torch.cat(writes).to(torch.int32)


def random_seqs(rng, device, batch):
    seqs = []
    for _ in range(batch):
        n, first = rng.randint(1, 9), rng.randint(0, 300)
        pages = torch.randint(1, 10 ** 6, (first + n + rng.randint(0, 5),), dtype=torch.int32, device=device)
        seqs.append((first, n, pages))
    return seqs


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_static_rows_match_the_per_sequence_rule(device):
    rng = random.Random(0)
    for _ in range(100):
        ratio = rng.choice([1, 2, 4, 8])
        seqs = random_seqs(rng, device, rng.randint(1, 130))
        got, want = FlashNextIndexer.static_rows(seqs, ratio, device), per_sequence(seqs, ratio, device)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]) and got[0].dtype == torch.int32
