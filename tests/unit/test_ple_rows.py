"""Flash-Next PLE rows on the host (#36): the page-cache-first gather, the per-column hash and the
verification rows read column by column as the draft steps end, each against the reference path."""

import json

import numpy as np
import pytest
import torch

from sharar.engine.rejection import DraftColumns
from sharar.model.flashnext.extras import FlashNextExtras
from sharar.model.flashnext.ple import ROW_BYTES, PleLayer, PleTable

EOS = 7
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


class _Hash:
    ngram, heads_per_ngram, heads = 3, 2, 4
    multipliers = torch.tensor([1000003, 999983, 1000033], dtype=torch.int64)
    vocab_sizes = torch.tensor([53, 59, 61, 67], dtype=torch.int64)
    offsets = torch.tensor([0, 53, 112, 173], dtype=torch.int64)
    ngram_ids, token_ids, _hash, rows_for = PleLayer.ngram_ids, PleLayer.token_ids, PleLayer._hash, PleLayer.rows_for


def test_column_hash_matches_the_sequence_hash():
    h = _Hash()
    for history in (None, torch.tensor([3, 4]), torch.tensor([EOS, 4]), torch.tensor([3, EOS])):
        ids = torch.tensor([5, EOS, 9, 11, EOS, EOS, 2, 6])
        full = h.ngram_ids(ids, history, EOS)
        ext = [EOS, EOS] + ids.tolist() if history is None else history.tolist() + ids.tolist()
        hist = torch.tensor([ext[j:j + 2] for j in range(len(ids))])
        assert torch.equal(h.token_ids(ids, hist, EOS), full)


class _Event:
    def synchronize(self):
        pass


def test_streamed_verify_rows_match_the_block(table):
    ple = _Hash()
    ple.table, ple.device = table, torch.device("cpu")
    extras = object.__new__(FlashNextExtras)
    extras.ple, extras.device, extras.eos = ple, torch.device("cpu"), EOS
    extras._staging, extras._staged = None, None

    class Req:
        def __init__(self, prompt, output):
            self.prompt_ids, self.output_ids = prompt, output
            self.context_len = len(prompt) + len(output)

    reqs = [Req([1, 2, 3], [4]), Req([9], [EOS]), Req([5, EOS, 6, 8], [2, 3])]
    drafts = torch.tensor([[11, 12, 13, 14], [EOS, 15, 16, 17], [18, EOS, EOS, 19]])
    reference = extras._verify_rows(reqs, drafts.tolist())
    k = drafts.shape[1]
    columns = DraftColumns(drafts.t().contiguous(), [_Event()] * k, len(reqs))
    for _ in range(2):  # the second round reuses the staging rows
        out = torch.zeros_like(reference)
        extras._stream_verify_rows(out, reqs, columns, k)
        assert torch.equal(out, reference)
