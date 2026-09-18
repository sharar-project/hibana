"""accept_block reproduces the host acceptance rule for every row shape."""
import random

import torch

from sharar.engine.accept import MAX_STOPS, accept_block, host_emitted


def host_reference(y, z, logits, relax, budget, stops):
    b, k = y.shape
    acc, counts, flags = [], [], []
    for i in range(b):
        top_k, margin = relax[0, i].item(), relax[1, i].item()
        row_flags = []
        a = 0
        for j in range(k):
            exact = y[i, j] == z[i, j]
            relaxed = False
            if logits is not None and top_k > 0:
                ly = logits[i, j, y[i, j]].float()
                lmax = logits[i, j].max().float()
                rank = (logits[i, j] > ly).sum().item()
                relaxed = rank < top_k and (lmax - ly).item() <= margin
            if not (exact or relaxed):
                break
            row_flags.append(not exact)
            a += 1
        stop_ids = [t for t in stops[i].tolist() if t >= 0]
        emitted = host_emitted(y[i].tolist(), z[i].tolist(), a, int(budget[i]), stop_ids, ignore_eos=not stop_ids)
        acc.append(a)
        counts.append(len(emitted))
        flags.append(row_flags + [False] * (k - len(row_flags)))
    return torch.tensor(acc), torch.tensor(counts), torch.tensor(flags)


def case(seed, b=4, k=15, vocab=32):
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    y = torch.randint(0, vocab, (b, k), generator=g)
    z = y.new_empty((b, k + 1))
    z[:, :k] = torch.where(torch.rand((b, k), generator=g) < 0.6, y, torch.randint(0, vocab, (b, k), generator=g))
    z[:, k] = torch.randint(0, vocab, (b,), generator=g)
    logits = torch.randn((b, k, vocab), generator=g)
    logits.scatter_(-1, z[:, :k].unsqueeze(-1), 5.0)  # the argmax row is the target's choice
    # Some drafted tokens sit within a logit of the argmax: the relaxed rule's territory.
    near = torch.rand((b, k), generator=g) < 0.3
    logits[near.unsqueeze(-1).expand_as(logits) & (torch.arange(vocab).view(1, 1, -1) == y.unsqueeze(-1))] = 4.5
    relax = torch.tensor([[rng.choice([0, 0, 1, 2, 3]) for _ in range(b)], [rng.choice([0.0, 1.0, 2.0]) for _ in range(b)]], dtype=torch.float32)
    budget = torch.tensor([rng.choice([0, 1, 2, 5, 16, 100]) for _ in range(b)])
    stops = torch.full((b, MAX_STOPS), -1, dtype=torch.int64)
    for i in range(b):
        for m in range(rng.choice([0, 0, 1, 2, MAX_STOPS])):
            stops[i, m] = rng.randrange(vocab)
    return y, z, logits, relax, budget, stops


def test_matches_host_rule_over_random_blocks():
    for seed in range(300):
        y, z, logits, relax, budget, stops = case(seed)
        acc, counts, flags = accept_block(y, z, logits, relax, budget, stops)
        ref_acc, ref_counts, ref_flags = host_reference(y, z, logits, relax, budget, stops)
        assert torch.equal(acc, ref_acc), seed
        assert torch.equal(counts, ref_counts), seed
        assert torch.equal(flags, ref_flags), seed


def test_exact_only_without_logits_and_padded_rows():
    y, z, _, relax, budget, stops = case(7)
    budget[-1] = 0  # a padded verify row emits nothing
    acc, counts, flags = accept_block(y, z, None, relax, budget, stops)
    ref_acc, ref_counts, ref_flags = host_reference(y, z, None, torch.zeros_like(relax), budget, stops)
    assert torch.equal(acc, ref_acc) and torch.equal(counts, ref_counts) and not flags.any()
    assert counts[-1] == 0


def test_stop_and_budget_bounds():
    y = torch.tensor([[3, 4, 5]])
    z = torch.tensor([[3, 4, 5, 9]])  # all accepted, correction 9
    stops = torch.full((1, MAX_STOPS), -1)
    relax = torch.zeros(2, 1)
    _, counts, _ = accept_block(y, z, None, relax, torch.tensor([100]), stops)
    assert counts.tolist() == [4]
    stops[0, 0] = 4  # stop at the second emitted token
    _, counts, _ = accept_block(y, z, None, relax, torch.tensor([100]), stops)
    assert counts.tolist() == [2]
    stops[0, 0] = 9  # the correction itself is the stop token
    _, counts, _ = accept_block(y, z, None, relax, torch.tensor([100]), stops)
    assert counts.tolist() == [4]
    _, counts, _ = accept_block(y, z, None, relax, torch.tensor([1]), stops)
    assert counts.tolist() == [1]
