"""Flash-Next PLE rows on the host: the native gather, the per-column hash and the
verification rows read column by column as the draft steps end, each against the reference path."""

import json

import numpy as np
import pytest
import torch

from hibana.engine.rejection import DraftColumns
from hibana.model.flashnext.extras import FlashNextExtras
from hibana.model.flashnext.ple import ROW_BYTES, PleLayer, PleTable, ResidentRows

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
    assert (table.gather(rows) == reference).all()
    assert table.hits + table.misses == rows.size  # each row read once, from the page cache or blocking
    out = np.zeros((9, 16, ROW_BYTES), dtype=np.uint8)
    assert table.gather(rows, out=out) is not None and (out == reference).all()
    assert (table.gather(rows[:1, :1]) == reference[:1, :1]).all()


def test_gather_reads_every_row_once(table):
    """Each row read once, in the caller's order, any count of rows including across shard boundaries and repeats."""
    rng = np.random.default_rng(4)
    for shape in [(1,), (17,), (3, 16), (1000,)]:
        rows = rng.integers(0, table.rows, size=shape)
        rows.reshape(-1)[:2] = [table.rows - 1, 0][:rows.size]  # the last row of the last shard, the first of the first
        hits, misses = table.hits, table.misses
        assert (table.gather(rows) == table.gather_mapped(rows)).all()
        assert table.hits + table.misses - hits - misses == rows.size


def test_gather_reads_rows_out_of_the_page_cache(table):
    """Rows the page cache does not hold take the blocking pass (one pool thread each) and come back the same."""
    import os
    rows = np.random.default_rng(5).integers(0, table.rows, size=(5, 16))
    for fd in table.fds:  # before the maps touch them: mapped pages stay cached
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    misses = table.misses
    got = table.gather(rows)
    if table.misses == misses:
        pytest.skip("this filesystem kept the rows cached")
    assert (got == table.gather_mapped(rows)).all()


class _Hash:
    ngram, heads_per_ngram, heads = 3, 2, 4
    multipliers = torch.tensor([1000003, 999983, 1000033], dtype=torch.int64)
    vocab_sizes = torch.tensor([53, 59, 61, 67], dtype=torch.int64)
    offsets = torch.tensor([0, 53, 112, 173], dtype=torch.int64)
    ngram_ids, token_ids, _hash, rows_for, column_ids = (PleLayer.ngram_ids, PleLayer.token_ids, PleLayer._hash, PleLayer.rows_for,
                                                        PleLayer.column_ids)
    device_rows, resident = PleLayer.device_rows, None


def test_column_hash_matches_the_sequence_hash():
    h = _Hash()
    for history in (None, torch.tensor([3, 4]), torch.tensor([EOS, 4]), torch.tensor([3, EOS])):
        ids = torch.tensor([5, EOS, 9, 11, EOS, EOS, 2, 6])
        full = h.ngram_ids(ids, history, EOS)
        ext = [EOS, EOS] + ids.tolist() if history is None else history.tolist() + ids.tolist()
        hist = torch.tensor([ext[j:j + 2] for j in range(len(ids))])
        assert torch.equal(h.token_ids(ids, hist, EOS), full)
        assert (h.column_ids(ids.numpy(), hist[:, 1].numpy(), hist[:, 0].numpy(), EOS) == full.numpy()).all()


def test_column_hash_in_numpy_wraps_as_torch():
    """Token ids whose products overflow int64 (the served multipliers are large): numpy's wrapping product, xor and
    floor remainder give torch's ids bit for bit."""
    h = _Hash()
    h.multipliers = torch.tensor([6364136223846793005, -7046029254386353131, 3935559000370003845], dtype=torch.int64)
    rng = np.random.default_rng(3)
    for _ in range(5):
        cur, b1, b2 = (torch.from_numpy(rng.integers(0, 1 << 20, 64)) for _ in range(3))
        b1[::7], b2[::5] = EOS, EOS
        want = h.token_ids(cur, torch.stack([b2, b1], 1), EOS)
        assert (h.column_ids(cur.numpy(), b1.numpy(), b2.numpy(), EOS) == want.numpy()).all()


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

    class Resident:  # two Sparks: the ids column by column, then one device gather
        def gather(self, ids):
            return torch.from_numpy(table.gather(ids.numpy()))
    ple.resident = Resident()
    out = torch.zeros_like(reference)
    extras._stream_verify_rows(out, reqs, columns, k)
    assert torch.equal(out, reference)


def test_resident_shares_give_the_host_rows(table, monkeypatch):
    """Two Sparks: each rank copies its heads' rows (here across shard boundaries) and gathers them; the
    all-gather of both ranks' parts, in rank order, is the host gather's bytes."""
    from hibana import distributed
    h = _Hash()
    shares = [ResidentRows(table, h.offsets, h.vocab_sizes, r, 2, torch.device("cpu"), log=lambda *_: None) for r in (0, 1)]
    assert [(s.h0, s.base, s.rows.shape[0]) for s in shares] == [(0, 0, 112), (2, 112, 128)]
    rng = np.random.default_rng(4)
    ids = torch.stack([h.offsets[j] + torch.from_numpy(rng.integers(0, int(h.vocab_sizes[j]), 37)) for j in range(4)], 1)
    reference = torch.from_numpy(table.gather(ids.numpy()))
    parts = [s.rows.index_select(0, (ids[:, s.h0:s.h0 + s.per] - s.base).reshape(-1)).view(37, -1) for s in shares]

    def all_gather(x):
        assert any(torch.equal(x, p) for p in parts)  # each rank sends its own heads' rows
        return torch.cat(parts, -1)
    monkeypatch.setattr(distributed, "all_gather", all_gather)
    for s in shares:
        assert torch.equal(s.gather(ids), reference)
    ple = _Hash()
    ple.resident = shares[0]
    assert torch.equal(ple.device_rows(ids), reference)


def test_prefetch_asks_for_every_page_the_rows_touch(table, monkeypatch):
    """WILLNEED on each distinct page a row lies on (both pages of a row across a page boundary)."""
    import os
    from hibana.model.flashnext import ple as ple_module
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, off, n, advice: calls.append((fd, off, n, advice)))
    rows = np.array([[0, 1, 25], [25, 150, 291]])  # row 25 of shard 0 crosses its 4 KiB page; 150 and 291 lie in other shards
    table.prefetch(rows)
    table._prefetcher.shutdown(wait=True)
    want = set()
    for r in np.unique(rows):
        s = int(np.searchsorted(table.starts, r, side="right") - 1)
        first = table.offsets[s] + (int(r) - int(table.starts[s])) * ROW_BYTES
        want |= {(table.fds[s], first & -ple_module._PAGE), (table.fds[s], (first + ROW_BYTES - 1) & -ple_module._PAGE)}
    assert {(fd, off) for fd, off, _, _ in calls} == want and len(calls) == len(want)
    assert all(n == ple_module._PAGE and advice == os.POSIX_FADV_WILLNEED for _, _, n, advice in calls)


def test_a_prompt_prefetches_its_later_chunks_once(table):
    """The first chunk of a prompt asks for the rows of the rest of it, once; nothing with resident heads."""
    ple = _Hash()
    ple.table = table
    asked = []
    table.prefetch = lambda rows: asked.append(rows.copy())
    extras = object.__new__(FlashNextExtras)
    extras.ple, extras.eos, extras._prefetched = ple, EOS, __import__("weakref").WeakSet()

    class Req:
        def __init__(self, prompt):
            self.prompt_ids, self.output_ids = prompt, []
    prompt = [5, 9, EOS, 11, 3, 8, 2, 6, 4]
    req = Req(prompt)
    extras._prefetch_prompts([(req, 0, 4, None)])
    extras._prefetch_prompts([(req, 4, 3, None)])  # a later chunk: already asked
    assert len(asked) == 1
    full = ple.ngram_ids(torch.tensor(prompt), None, EOS).numpy()
    assert np.array_equal(asked[0], full[4:])  # the rows of tokens 4.. exactly as their own chunks will hash them
    extras._prefetch_prompts([(Req([1, 2, 3]), 0, 3, None)])  # the whole prompt in one chunk: nothing to prefetch
    assert len(asked) == 1
    ple.resident = object()
    extras._prefetch_prompts([(Req(prompt), 0, 4, None)])
    assert len(asked) == 1
