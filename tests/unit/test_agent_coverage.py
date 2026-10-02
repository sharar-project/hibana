from bench.agent_coverage import select_branch, score_branch


def test_branch_screen_counts_recovered_correction_and_lost_full_chain():
    branch = {'depth': 1, 'token': 42}
    assert score_branch(branch, [7, 42])['potential_token_delta'] == 1
    assert score_branch(branch, [7, 43])['potential_token_delta'] == 0
    assert score_branch(branch, list(range(8)))['potential_token_delta'] == -1


def test_branch_selection_uses_draft_path_mass_and_stable_ties():
    candidates = [[10 + i * 2, 11 + i * 2] for i in range(7)]
    scores = [[[0., 0.], [0., 0.]] for _ in range(7)]
    # Equal first-step choices: choose rank zero for the main chain and rank
    # one for the alternative. Later alternatives have smaller path mass.
    assert select_branch(candidates, scores) == {'depth': 0, 'token': 11, 'draft_path_mass': .5}
