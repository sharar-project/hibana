import itertools
import math

import pytest

from bench.archive.gb10_tree_screen import probability_tree


def test_best_first_tree_matches_exhaustive_conditional_mass():
    candidates = [[10, 11], [20, 21], [30, 31]]
    probs = [[[.6, .4], [.6, .4]], [[.8, .2], [.3, .7]], [[.55, .45], [.1, .9]]]
    scores = [[[math.log(x) for x in row] for row in rows] for rows in probs]
    paths = [p for depth in range(1, 4) for p in itertools.product(range(2), repeat=depth)]
    def mass(path):
        result, prev = 1., 0
        for depth, rank in enumerate(path):
            result *= probs[depth][prev][rank]
            prev = rank
        return result
    for count in range(1, 6):
        expected = max(sum(mass(p) for p in subset)
                       for subset in itertools.combinations(paths, count)
                       if all(len(p) == 1 or p[:-1] in subset for p in subset))
        tree = probability_tree(candidates, scores, count + 1)
        assert sum(math.exp(n['log_mass']) for n in tree['nodes']) == pytest.approx(expected)
        assert all(0 <= n['parent'] < n['id'] for n in tree['nodes'])


def test_probability_tree_can_spend_rows_on_siblings_instead_of_greedy_tail():
    candidates = [[10, 11], [20, 21]]
    scores = [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]
    tree = probability_tree(candidates, scores, 3)
    assert [n['token'] for n in tree['nodes']] == [10, 11]
    assert tree['greedy'] == [10, 20]


def test_anchor_only_and_invalid_budget():
    assert probability_tree([[1]], [[[0.]]], 1)['nodes'] == []
    with pytest.raises(ValueError, match='budget'):
        probability_tree([[1]], [[[0.]]], 3)
    with pytest.raises(ValueError, match='finite'):
        probability_tree([[1]], [[[float('nan')]]], 2)
