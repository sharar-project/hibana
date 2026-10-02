"""Modules of the sharar-eval checkout, for engine-side tools that drive their own server with its HTTP clients."""
import importlib
import os
import sys
from pathlib import Path


def evals_module(name: str):
    """Import `name` (e.g. "evals.speed.local") from $SHARAR_EVALS, else ../sharar-eval beside this checkout."""
    root = Path(os.environ.get("SHARAR_EVALS") or Path(__file__).resolve().parents[3] / "sharar-eval")
    if not (root / "evals").is_dir():
        raise SystemExit(f"{root}: no sharar-eval checkout; clone abdelfattah-lab/sharar-eval there or set SHARAR_EVALS")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(name)
