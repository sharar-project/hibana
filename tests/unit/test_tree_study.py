import itertools
import math

import pytest

from bench.tree_study import select_tree, summarize_coverage, tree_coverage


def prefixes(tree):
    paths = {0: ()}
    for node in tree['nodes']:
        assert node['parent'] in paths
        paths[node['id']] = paths[node['parent']] + (node['rank'],)
        assert node['depth'] == len(paths[node['id']])
    return set(paths.values()) - {()}


def test_frontier_selection_matches_exhaustive_prefix_closed_mass_optimum():
    probabilities = [[[.6, .4], [.6, .4]], [[.7, .3], [.2, .8]],
                     [[.55, .45], [.1, .9]]]
    scores = [[[math.log(p) for p in row] for row in rows] for rows in probabilities]
    tree = select_tree([[10, 11], [20, 21], [30, 31]], scores, extra_nodes=3)
    chosen = prefixes(tree)
    backbone = {(0,), (0, 0), (0, 0, 0)}
    assert backbone <= chosen and tree['target_rows'] == 7
    all_paths = {p for d in range(1, 4) for p in itertools.product(range(2), repeat=d)}
    def mass(p):
        return math.prod(probabilities[d][p[d - 1] if d else 0][rank] for d, rank in enumerate(p))
    feasible = [backbone | set(extra) for extra in itertools.combinations(all_paths - backbone, 3)
                if all(p[:-1] in backbone | set(extra) | {()} for p in extra)]
    optimum = max(sum(map(mass, nodes)) for nodes in feasible)
    assert sum(map(mass, chosen)) == pytest.approx(optimum)
    assert sum(math.exp(n['log_mass']) for n in tree['nodes']) == pytest.approx(optimum)


def test_ties_are_full_rank_lexical_and_greedy_chain_is_retained():
    tree = select_tree([[10, 11], [20, 21]], [[[0, 0]] * 2] * 2, extra_nodes=3)
    assert tree['greedy'] == [10, 20]
    assert prefixes(tree) == {(0,), (0, 0), (1,), (0, 1), (1, 0)}
    assert [n['rank'] for n in tree['nodes']] == [0, 0, 1, 1, 0]


def test_converging_draft_ranks_do_not_merge_distinct_target_histories():
    tree = select_tree([[10, 11], [20, 21]], [[[2, 1]] * 2] * 2, extra_nodes=4)
    assert tree['target_rows'] == 7
    same_token = [n for n in tree['nodes'] if n['token'] == 20]
    assert len(same_token) == 2 and len({n['parent'] for n in same_token}) == 2
    assert tree_coverage(tree, [11, 20, 999]) == 3
    assert tree_coverage(tree, [11, 999, 999]) == 2
    assert tree_coverage(tree, [999, 20, 999]) == 1
    with pytest.raises(ValueError, match='lookahead'):
        tree_coverage(tree, [11, 20])


def test_sustained_coverage_excludes_only_insufficient_terminal_lookahead():
    tree = select_tree([[10, 11], [20, 21]], [[[2, 1]] * 2] * 2, extra_nodes=4)
    captured = {'output_ids': [0] * 512 + [11, 21, 77, 88], 'rows': [
        {'offset': 512, 'emitted': [11], 'tree': tree},
        {'offset': 513, 'emitted': [21], 'tree': tree},
        {'offset': 514, 'emitted': [77], 'tree': tree}]}
    result = summarize_coverage(captured)
    assert result['included_rounds'] == 2 and result['excluded_terminal'] == 1
    assert result['native_tokens'] == 2 and result['candidate_tokens'] == 4
    assert result['after_512'] == {'native': 2, 'candidate': 4}
    captured['rows'][0]['emitted'].append(21)
    with pytest.raises(AssertionError, match='native lattice'):
        summarize_coverage(captured)


@pytest.mark.parametrize('candidates,scores,extra', [
    ([], [], 0),
    ([[1, 1]], [[[0, 1], [0, 1]]], 0),
    ([[1, 2]], [[[0, 1]]], 0),
    ([[1, 2]], [[[0, float('nan')], [0, 1]]], 0),
    ([[1, 2]], [[[0, float('inf')], [0, 1]]], 0),
    ([[1]], [[[0]]], 1),
    ([[1]], [[[0]]], -1),
])
def test_invalid_lattices_and_impossible_budgets_fail(candidates, scores, extra):
    with pytest.raises(ValueError):
        select_tree(candidates, scores, extra)
