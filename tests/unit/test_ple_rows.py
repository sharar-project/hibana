"""Flash-Next PLE rows on the host (#36): the page-cache-first gather against the memory-map reference."""

import json

import numpy as np
import pytest

from sharar.model.flashnext.ple import ROW_BYTES, PleTable

NAME = "ple.ngram_embedding"


def _write_fp8(path, tensors: dict) -> None:
    """A safetensors file of uint8 rows labelled F8_E4M3 (the checkpoint's PLE shards)."""
    header, offset = {}, 0
    for name, value in tensors.items():
        header[name] = {"dtype": "F8_E4M3", "shape": list(value.shape), "data_offsets": [offset, offset + value.nbytes]}
        offset += value.nbytes
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(len(text).to_bytes(8, "little") + text + b"".join(v.tobytes() for v in tensors.values()))


@pytest.fixture
def table(tmp_path):
    """Two files, three shards of different sizes."""
    rng = np.random.default_rng(0)
    counts, weight_map, files = [97, 64, 131], {}, {0: {}, 1: {}}
    for i, count in enumerate(counts):
        weight_map[f"{NAME}.shard_{i}.weight"] = f"part-{i % 2}.safetensors"
        files[i % 2][f"{NAME}.shard_{i}.weight"] = rng.integers(0, 256, size=(count, ROW_BYTES), dtype=np.uint8)
    for part, tensors in files.items():
        _write_fp8(tmp_path / f"part-{part}.safetensors", tensors)
    return PleTable(tmp_path, weight_map, NAME, shards=len(counts))


def test_gather_matches_the_memory_maps(table):
    rows = np.random.default_rng(1).integers(0, table.rows, size=(9, 16))
    reference = table.gather_mapped(rows)
    assert (table.gather(rows) == reference).all()  # page cache first (the files were just written)
    assert table.hits > 0
    out = np.zeros((9, 16, ROW_BYTES), dtype=np.uint8)
    assert table.gather(rows, out=out) is not None and (out == reference).all()
    table._nowait = False  # every row through the threads
    assert (table.gather(rows) == reference).all()
    assert (table.gather(rows[:1, :1]) == reference[:1, :1]).all()
