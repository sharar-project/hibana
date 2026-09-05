from bench.mixed_forward import summarize


def samples():
    return [dict(case='selected', mode=mode, warmup=False, ablation=False,
                 automatic_passed=True, rubric_review_pending=True, mixed_steps=int(mode != 'off'),
                 arrival_mixed=mode != 'off',
                 anchor_completion_s=seconds, arrival_completion_s=2., arrival_ttft_s=.2,
                 anchor_max_gap_s=.3, arrival_max_gap_s=.2)
            for mode, seconds in [('off', 1.), ('full', .8)]]


def test_missing_overlap_and_failed_quality_never_produce_speed_ratios():
    for field, value in [('mixed_steps', 0), ('arrival_mixed', False), ('automatic_passed', False)]:
        rows = samples()
        rows[1][field] = value
        assert 'anchor_reduction' not in summarize(rows)['selected']


def test_warmups_and_ablations_are_excluded_and_manual_review_stays_pending():
    rows = samples()
    rows.extend([{**rows[0], 'anchor_completion_s': 100., 'warmup': True},
                 {**rows[0], 'anchor_completion_s': 100., 'ablation': True}])
    result = summarize(rows)['selected']
    assert abs(result['anchor_reduction'] - .2) < 1e-10
    assert result['modes']['off']['samples'] == 1
    assert result['rubric_review_pending']


def test_failed_stream_without_first_token_is_reported_without_summary_crash():
    rows = samples()
    rows[1].update(automatic_passed=False, arrival_ttft_s=None)
    result = summarize(rows)['selected']
    assert result['modes']['full']['arrival_ttft_s'] is None
    assert 'anchor_reduction' not in result
