"""The GPU suite runs only when asked (SHARAR_RUN_GPU=1; tests/conftest.py points it at the real checkpoints) on a CUDA device.
Otherwise its modules are not even imported: several need Triton or FlashInfer at import time."""
import os

import pytest


def _gpu_suite() -> bool:
    if os.environ.get("SHARAR_RUN_GPU") != "1":
        return False
    import torch
    return torch.cuda.is_available()


def pytest_ignore_collect(collection_path, config):
    return collection_path.name.startswith("test_") and collection_path.suffix == ".py" and not _gpu_suite()


def pytest_collection_modifyitems(config, items):
    """The modules on the shared 27B (`qwen38`) run last, so it is never resident with another model (the Flash-Next
    lifecycle module loads one). The sort is stable: every other item, and each module's items, keep their order."""
    users = {item.module for item in items if "qwen38" in getattr(item, "fixturenames", ())}
    items.sort(key=lambda item: item.module in users)


@pytest.fixture(scope="session")
def qwen38():
    """The pinned 27B target, loaded once (about 12 s) for every module that builds an executor on it. A fixture may
    build executors on it but must leave the model itself as it found it."""
    import gc

    import torch

    from hibana import config
    from hibana.model.qwen38 import Qwen38
    config.check()
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        model = Qwen38(config.snapshot_dir(), log=print)
    torch.cuda.empty_cache()
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()
