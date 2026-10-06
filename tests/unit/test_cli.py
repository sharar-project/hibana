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


@pytest.mark.parametrize('env, expected', [
    ({'SHARAR_ACCEPT_DEFAULT_SAMPLED': '1', 'SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD': '2:1'}, ((2, 1.0), '2:1.0')),
    ({'SHARAR_ACCEPT_DEFAULT_SAMPLED': '1'}, (True, None)),
    ({'SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD': '2:1.0'}, (False, None)),  # the threshold alone turns nothing on
    ({'SHARAR_ACCEPT_DEFAULT_SAMPLED': '1', 'SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD': 'off'}, (False, None))])
def test_sampled_acceptance_threshold_reaches_serve(monkeypatch, env, expected):
    import sharar.api.server
    from sharar.engine import config as engine_config
    for k in ('SHARAR_ACCEPT_DEFAULT_SAMPLED', 'SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD'):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    seen = {}
    monkeypatch.setattr(sharar.api.server, 'serve', lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(engine_config, 'install', lambda cfg: seen.update(cfg=cfg))
    assert main(['serve', '--spec', '15', '--accept-default', '3:2.0']) == 0
    assert (seen['accept_default_sampled'], seen['cfg'].accept_default_sampled_threshold) == expected
    assert seen['cfg'].accept_default_sampled == bool(expected[0])
