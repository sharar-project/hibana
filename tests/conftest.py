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


@pytest.fixture(scope="session")
def snapshot():
    from sharar import config

    try:
        return config.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return Path(os.environ.get("SHARAR_RESULTS", "/tmp/sharar-results"))


@pytest.fixture(scope="session")
def reference_url() -> str:
    return os.environ.get("SHARAR_REFERENCE_URL", "http://127.0.0.1:30000")
