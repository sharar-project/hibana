"""Shared source identity, result writing and cache cleanup for benchmarks."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n')


def clear(ex) -> None:
    ex.flush_cache()
    assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1, ex.kv_pool.free_count
    assert ex.state_pool.free_count == ex.state_pool.num_slots - 1, ex.state_pool.free_count
    proposer = ex.spec.proposer
    assert len(proposer.free_slots) == ex.max_batch, proposer.free_slots


def identity(args) -> dict:
    import torch
    root = Path(__file__).resolve().parents[2]
    sources = {str(p.relative_to(root)): sha_file(p) for p in sorted((root / 'sharar').rglob('*.py'))}
    sources.update({str(p.relative_to(root)): sha_file(p) for p in sorted((root / 'bench').rglob('*.py'))})
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        head = None
    return {'git_head': head, 'source_sha256': sources, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'python': platform.python_version(),
            'argv': sys.argv[1:], 'hostname': platform.node(), 'env': {k: v for k, v in os.environ.items() if k.startswith('SHARAR_')}}
