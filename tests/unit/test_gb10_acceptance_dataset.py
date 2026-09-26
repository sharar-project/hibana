import pytest

from bench.gb10_acceptance_dataset import summarize_capture, walk


def test_oracle_support_and_terminal_censoring_are_separate_from_acceptance():
    rows = [
        {'output_offset': 1, 'proposal_ids': [2, 9], 'candidates': [[2, 7], [9, 3]],
         'emitted_ids': [2, 3]},
        {'output_offset': 3, 'proposal_ids': [4, 9], 'candidates': [[4, 7], [9, 5]],
         'emitted_ids': [4, 5]},
    ]
    summary = summarize_capture(rows, [1, 2, 3, 4, 5], depth=2)
    assert summary['all']['rounds'] == 1
    assert summary['all']['mean_native_emitted'] == 2
    assert summary['all']['mean_oracle_emitted'] == 3
    assert summary['censored_rounds'] == 1
    assert rows[1]['native_matches_recorded_emit']


def test_capture_rejects_wrong_emitted_boundary():
    with pytest.raises(ValueError, match='reproduce'):
        summarize_capture([{'output_offset': 1, 'proposal_ids': [2, 9],
                            'candidates': [[2, 7], [9, 3]], 'emitted_ids': [2]}],
                          [1, 2, 3, 4], depth=2)


def test_lattice_walk_conditions_on_previous_candidate_rank_and_first_tie():
    assert walk([[10, 20], [30, 40]], [[[0, 1], [9, 0]], [[9, 0], [1, 1]]]) == [20, 30]


def test_full_acceptance_includes_bonus_and_missing_support_is_counted():
    row = {'output_offset': 1, 'proposal_ids': [2, 3],
           'candidates': [[2, 8], [3, 9]], 'emitted_ids': [2, 3, 4]}
    assert summarize_capture([row], [1, 2, 3, 4], 2)['all']['mean_native_emitted'] == 3
    row = {'output_offset': 1, 'proposal_ids': [8, 9],
           'candidates': [[8, 7], [3, 9]], 'emitted_ids': [2]}
    assert summarize_capture([row], [1, 2, 3, 4], 2)['all']['first_rejection_missing_support'] == 1


def test_kernel_attribution_uses_launch_correlation_not_gpu_timestamp():
    from bench.gb10_profile import kernel_breakdown

    result = kernel_breakdown({'traceEvents': [
        {'ph': 'X', 'name': 'sharar.target_verify', 'ts': 0, 'dur': 10, 'tid': 1},
        {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaGraphLaunch', 'ts': 2,
         'dur': 1, 'tid': 1, 'args': {'correlation': 7}},
        {'ph': 'X', 'cat': 'kernel', 'name': 'GemmBlockScaled', 'ts': 100,
         'dur': 500, 'args': {'correlation': 7}},
    ]})
    assert result['groups'] == [{'owner': 'sharar.target_verify', 'family': 'nvfp4_gemm',
                                 'kernel_ms': .5, 'calls': 1}]


def test_driver_launch_is_attributed_to_innermost_scope():
    from bench.gb10_profile import kernel_breakdown

    result = kernel_breakdown({'traceEvents': [
        {'ph': 'X', 'name': 'sharar.accept', 'ts': 0, 'dur': 10, 'tid': 1},
        {'ph': 'X', 'name': 'sharar.reconstruction', 'ts': 1, 'dur': 5, 'tid': 1},
        {'ph': 'X', 'cat': 'cuda_driver', 'name': 'cuLaunchKernel', 'ts': 2,
         'dur': 1, 'tid': 1, 'args': {'correlation': 9}},
        {'ph': 'X', 'cat': 'kernel', 'name': 'reconstruct', 'ts': 100,
         'dur': 100, 'args': {'correlation': 9}},
    ]})
    assert result['groups'][0]['owner'] == 'sharar.reconstruction'
