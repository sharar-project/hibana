"""The GPU suite runs only when asked (SHARAR_RUN_GPU=1, set by tools/gpu-tests.sh) on a CUDA device.
Otherwise its modules are not even imported: several need Triton or FlashInfer at import time."""
import os


def _gpu_suite() -> bool:
    if os.environ.get("SHARAR_RUN_GPU") != "1":
        return False
    import torch
    return torch.cuda.is_available()


def pytest_ignore_collect(collection_path, config):
    return collection_path.name.startswith("test_") and collection_path.suffix == ".py" and not _gpu_suite()
