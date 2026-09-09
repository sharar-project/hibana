import pytest

from sharar.cli import main


@pytest.mark.parametrize('option,value', [('--mixed-forward', 'full'), ('--interactive-target-ms', '250'), ('--verify-policy', 'policy.json')])
def test_failed_serving_experiments_cannot_be_enabled(option, value):
    with pytest.raises(SystemExit) as exc:
        main(['serve', '--spec', '7', option, value])
    assert exc.value.code == 2


@pytest.mark.parametrize('args', [['--spec', '0'], ['--proposer', 'mtp'], ['--proposer', 'ngram']])
def test_fp8_draft_requires_parallel_speculation_before_model_import(args):
    with pytest.raises(SystemExit) as exc:
        main(['serve', '--draft-precision', 'fp8', *args])
    assert exc.value.code == 2
