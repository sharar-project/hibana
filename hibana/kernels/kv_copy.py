"""The SSD tier's KV row copies (see csrc/kv_copy.cu), compiled on first use.

The build is cached by torch under $HOME/.cache/torch_extensions, so it persists across restarts.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "csrc" / "kv_copy.cu"


@lru_cache(maxsize=1)
def module():
    from torch.utils.cpp_extension import load

    from hibana.engine.config import current
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    revision = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]  # one build per source revision (as kernels/w3.py)
    return load(name=f"sharar_kv_copy_{revision}", sources=[str(_SRC)], extra_cuda_cflags=["-O3"], verbose=False)
