"""Flash-Next PLE rows on the host (#36): the page-cache-first gather, the per-column hash and the
verification rows read column by column as the draft steps end, each against the reference path."""

import json

import numpy as np
import pytest
import torch

from sharar.engine.rejection import DraftColumns
from sharar.model.flashnext.extras import FlashNextExtras
from sharar.model.flashnext.ple import ROW_BYTES, PleLayer, PleTable, ResidentRows

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


@pytest.mark.parametrize("native", [True, False])
def test_gather_matches_the_memory_maps(table, native):
    table.native = native
    rows = np.random.default_rng(1).integers(0, table.rows, size=(9, 16))
    reference = table.gather_mapped(rows)
    assert (table.gather(rows) == reference).all()
    # Each row is read once, from the page cache or through the threads. How many hit the cache depends on the
    # filesystem (one without RWF_NOWAIT, like some container mounts, sends every row through the threads).
    assert table.hits + table.misses == rows.size
    out = np.zeros((9, 16, ROW_BYTES), dtype=np.uint8)
    assert table.gather(rows, out=out) is not None and (out == reference).all()
    table._nowait = False  # the page cache under the memory maps answers
    assert (table.gather(rows) == reference).all()
    assert (table.gather(rows[:1, :1]) == reference[:1, :1]).all()


def test_without_rwf_nowait_the_maps_answer_and_keep_the_rows(table, monkeypatch):
    """ZFS (#166): preadv refuses RWF_NOWAIT. The gather switches to the memory maps mid-call; rows it reads through
    them stay in the page cache, so the next gather copies them all without the threads."""
    import os
    table.native = False  # the Python gather's own fallbacks
    rows = np.random.default_rng(2).integers(0, table.rows, size=(5, 16))
    reference = table.gather_mapped(rows)

    def refuse(*args):
        raise OSError(95, "Operation not supported")
    monkeypatch.setattr(os, "preadv", refuse)
    monkeypatch.setattr(os, "RWF_NOWAIT", 8, raising=False)  # uv's standalone CPython has no preadv2
    table._nowait = True
    assert (table.gather(rows) == reference).all()
    assert not table._nowait
    hits, misses = table.hits, table.misses
    assert (table.gather(rows) == reference).all()
    assert (table.hits, table.misses) == (hits + rows.size, misses)


def test_where_mincore_cannot_see_the_page_cache_every_row_goes_to_the_threads(table, monkeypatch):
    """Files this process neither owns nor may write: mincore would call every page resident, so the maps answer
    through the threads only, as before #166."""
    import os
    from pathlib import Path
    from sharar.model.flashnext import ple
    path = Path(table.maps[0].filename)
    assert ple._residency_visible(path)  # the test's own file
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(path).st_uid + 1)
    monkeypatch.setattr(os, "access", lambda *args, **kwargs: False)
    assert not ple._residency_visible(path)
    rows = np.random.default_rng(3).integers(0, table.rows, size=(4, 16))
    reference = table.gather_mapped(rows)
    table.native = False
    table._nowait, table._mincore = False, False
    hits, misses = table.hits, table.misses
    assert (table.gather(rows) == reference).all()
    assert (table.hits, table.misses) == (hits, misses + rows.size)


def test_native_gather_reads_every_row_once(table):
    """the native pool reads each row once (a page-cache read or a blocking one), in the caller's order, any
    count of rows including across shard boundaries and repeats."""
    from sharar.model.flashnext import ple
    if ple._native() is None:
        pytest.skip("no C++ toolchain for the native gather")
    rng = np.random.default_rng(4)
    for shape in [(1,), (17,), (3, 16), (1000,)]:
        rows = rng.integers(0, table.rows, size=shape)
        rows.reshape(-1)[:2] = [table.rows - 1, 0][:rows.size]  # the last row of the last shard, the first of the first
        hits, misses = table.hits, table.misses
        assert (table.gather(rows) == table.gather_mapped(rows)).all()
        assert table.hits + table.misses - hits - misses == rows.size


class _Hash:
    ngram, heads_per_ngram, heads = 3, 2, 4
    multipliers = torch.tensor([1000003, 999983, 1000033], dtype=torch.int64)
    vocab_sizes = torch.tensor([53, 59, 61, 67], dtype=torch.int64)
    offsets = torch.tensor([0, 53, 112, 173], dtype=torch.int64)
    ngram_ids, token_ids, _hash, rows_for = PleLayer.ngram_ids, PleLayer.token_ids, PleLayer._hash, PleLayer.rows_for
    device_rows, resident = PleLayer.device_rows, None


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

    class Resident:  # two Sparks (#178): the ids column by column, then one device gather
        def gather(self, ids):
            return torch.from_numpy(table.gather(ids.numpy()))
    ple.resident = Resident()
    out = torch.zeros_like(reference)
    extras._stream_verify_rows(out, reqs, columns, k)
    assert torch.equal(out, reference)


def test_resident_shares_give_the_host_rows(table, monkeypatch):
    """Two Sparks (#178): each rank copies its heads' rows (here across shard boundaries) and gathers them; the
    all-gather of both ranks' parts, in rank order, is the host gather's bytes."""
    from sharar import distributed
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
    """#178: WILLNEED on each distinct page a row lies on (both pages of a row across a page boundary)."""
    import os
    from sharar.model.flashnext import ple as ple_module
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
    """#178: the first chunk of a prompt asks for the rows of the rest of it, once; nothing with resident heads."""
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
