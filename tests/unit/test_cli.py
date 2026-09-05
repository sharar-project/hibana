import pytest

from sharar.cli import main


def test_failed_mixed_serving_experiment_cannot_be_enabled():
    with pytest.raises(SystemExit) as exc:
        main(['serve', '--spec', '7', '--mixed-forward', 'full'])
    assert exc.value.code == 2
