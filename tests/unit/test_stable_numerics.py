"""Shape-stable numerics: window padding logic and the flag plumbing (no GPU)."""
from hibana.kernels import stable


def test_padded_rows_steps_over_the_measured_window_only():
    assert [stable.padded_rows(m, (5120, 6144)) for m in (8, 26, 27, 30, 32, 33, 64)] == [8, 26, 33, 33, 33, 33, 64]
    assert stable.padded_rows(27, (14336, 5120)) == 27  # other shapes have one cuBLASLt class
    assert stable.padded_rows(16, (5120, 6144)) == 16  # verify rows are untouched


def test_flag_reads_the_installed_config(monkeypatch):
    from hibana.engine import config
    monkeypatch.setenv("SHARAR_STABLE_NUMERICS", "1")
    config.reset()
    assert stable.enabled()
    with config.using(config.current().replace(stable_numerics=False)):
        assert not stable.enabled()
    monkeypatch.delenv("SHARAR_STABLE_NUMERICS")
    config.reset()
    assert not stable.enabled()
