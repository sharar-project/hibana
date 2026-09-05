from types import SimpleNamespace

import torch

from bench.mixed_profile import eligible_requests, numerical_check, shape_grid


def test_overlap_excludes_new_prefill_and_requests_that_cannot_verify():
    def req(**changes):
        return SimpleNamespace(**(dict(finished=False, cancel_requested=False, released=False,
                                       prefill_done=True, ready=True) | changes))
    existing = req()
    new = req()  # Its next verify depends on completing this very prefill.
    others = [req(finished=True), req(cancel_requested=True), req(released=True),
              req(prefill_done=False), req(ready=False)]
    assert eligible_requests([existing, new, *others], [(new, 128)], lambda r: r.ready) == [existing]
    for shape in shape_grid():
        assert shape['verify_rows'] // 8 + 1 <= 4
        if shape['layout'] == 'fill_budget':
            assert shape['prefill_rows'] + shape['verify_rows'] == shape['budget']


def test_numerical_check_keeps_prefill_and_verify_rows_distinct():
    items = [{'name': 'projection', 'fn': lambda x: x * 2}]
    result = numerical_check(items, {'projection': torch.ones(5, 2)},
                             {'projection': torch.full((3, 2), -7.)}, 5, 3)
    assert result['passed'] and result['unequal_fraction'] == 0
    assert [r['elements'] for r in result['operations']] == [10, 6]


def test_numerical_check_rejects_small_partition_error_and_nonfinite():
    # A long prefill must not hide corrupted verification rows in a global RMS.
    def corrupted(x):
        out = x.clone()
        if len(x) == 101:
            out[-1] *= 1.1
        return out
    args = ({'x': torch.ones(100, 2)}, {'x': torch.ones(1, 2)}, 100, 1)
    assert not numerical_check([{'name': 'x', 'fn': corrupted}], *args)['passed']
    def nonfinite(x):
        return x * float('inf')
    assert not numerical_check([{'name': 'x', 'fn': nonfinite}], *args)['passed']


def test_padded_baseline_executes_dummy_rows_but_checks_only_active_outputs():
    shapes = []
    def project(x):
        shapes.append(len(x))
        return x * 2
    verify = torch.cat((torch.ones(24, 2), torch.full((8, 2), float('nan'))))
    result = numerical_check([{'name': 'x', 'fn': project}],
                             {'x': torch.ones(5, 2)}, {'x': verify}, 5, 24, 32)
    assert shapes == [5, 32, 29]
    assert result['passed']
