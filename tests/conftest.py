import os
from pathlib import Path

import pytest


def _load_env() -> None:
    env_file = Path(__file__).resolve().parents[1] / "config" / "gb10.env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_env()


@pytest.fixture(autouse=True)
def _engine_config():
    """Each test starts from the environment (#44): a value one test installs never leaks into the next."""
    from sharar.engine import config
    config.reset()
    yield
    config.reset()


@pytest.fixture(scope="session")
def snapshot():
    from sharar import config

    try:
        return config.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
