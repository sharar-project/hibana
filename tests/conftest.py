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


def _gpu_environment() -> None:
    """With SHARAR_RUN_GPU=1 the suite runs against the real checkpoints: the paths and 3-bit copies `sharar serve` would
    use (sharar/launch/settings.py), unless the environment sets them: `SHARAR_RUN_GPU=1 uv run pytest tests/gpu`."""
    if os.environ.get("SHARAR_RUN_GPU") != "1":
        return
    from sharar.launch.settings import resolve

    s27, fn = resolve("27b", "fast"), resolve("flashnext", "fast")
    os.environ.setdefault("SHARAR_HF_CACHE", str(s27.hf_cache))
    os.environ.setdefault("SHARAR_FLASHNEXT_CACHE", str(fn.model_cache))
    os.environ.setdefault("SHARAR_RESULTS", str(s27.results))
    current = s27.w3_root / "qwen38-current"  # the served 27B profile (tests that need the exact path set their own)
    if "SHARAR_W3_CHECKPOINT" not in os.environ and (current / "manifest.json").is_file():
        os.environ.update(SHARAR_W3_CHECKPOINT=str(current.resolve()), SHARAR_W3_DECODE="1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"


_gpu_environment()


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
