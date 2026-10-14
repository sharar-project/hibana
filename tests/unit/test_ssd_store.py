"""The SSD store: objects round-trip byte-exact across a restart, keys chain
over every token from the root and fold in the serving identity, a corrupt or
misnamed file is a miss and the store deletes it, one key's writes and
deletes keep their order, completions come back in submission order whichever
writer finishes first, and a flush starts a new epoch that nothing in flight
can write into."""
import errno
import mmap
import os
import threading
import time
import weakref

import pytest
import torch

from sharar.cache.ssd_store import BLOCK, Header, SsdStore, block_hash, root_hash, tail_hash

IDENTITY = {"target": "qwen38", "revision": "r1", "w3": "cal:1"}


def segments(seed=0, n=BLOCK, layers=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for j in range(layers):
        for t in "kv":
            out.append((f"kv.{j}.{t}", torch.randint(0, 255, (n, 1, 2, 8), generator=g, dtype=torch.uint8).view(torch.float8_e4m3fn)))
    out.append(("conv", torch.randn(4, 6, 3, generator=g).to(torch.bfloat16)))
    out.append(("ssm", torch.randn(4, 2, 3, 3, generator=g)))
    return out


def header(kind, key, parent, start, tokens, segs, scalars=None):
    spec = [[name, str(t.dtype).removeprefix("torch."), list(t.shape), t.numel() * t.element_size()] for name, t in segs]
    return Header(kind, key.hex(), parent.hex(), start, list(tokens), spec, scalars or {})


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def read(store, kind, key):
    """The object's header and segments, read the engine's way (read_into a page-aligned buffer)."""
    size = store.size(kind, key)
    if size is None:
        return None
    buf = mmap.mmap(-1, (size + 4095) // 4096 * 4096)
    got = store.read_into(kind, key, memoryview(buf))
    if got is None:
        return None
    header, offset = got
    return header, {k: v.clone() for k, v in store.segments(header, torch.frombuffer(buf, dtype=torch.uint8), offset).items()}


def put(store, h, segs):
    store.write(h, [t for _, t in segs])
    store.wait()
    return store.completions()


def test_keys_chain_every_token_and_fold_in_the_identity():
    r = root_hash(IDENTITY)
    a, b = block_hash(r, range(BLOCK)), block_hash(r, [1] + list(range(1, BLOCK)))
    assert a != b and block_hash(block_hash(r, range(BLOCK)), range(BLOCK)) != a
    assert root_hash({**IDENTITY, "w3": "nearest"}) != r
    assert root_hash({**IDENTITY, "block": 1024}) != root_hash({**IDENTITY, "block": 256})
    assert tail_hash(a, [5, 6]) != tail_hash(a, [5]) != tail_hash(a, [])


def test_round_trip_across_a_restart(tmp_path):
    store = SsdStore(tmp_path, IDENTITY)
    key = block_hash(store.root, range(BLOCK))
    segs = segments()
    h = header("kv", key, store.root, 0, range(BLOCK), segs, {"draft": [11, 42]})
    ((kind, name, ok, nbytes),) = put(store, h, segs)
    assert (kind, name, ok) == ("kv", key.hex(), True) and nbytes % 4096 == 0
    restarted = SsdStore(tmp_path, IDENTITY)
    (listed, size), = list(restarted.scan())
    assert listed.key == key.hex() and listed.tokens == list(range(BLOCK)) and listed.end == BLOCK and size == nbytes
    got, tensors = read(restarted, "kv", key.hex())
    assert got.scalars == {"draft": [11, 42]}
    for name, t in segs:
        assert tensors[name].dtype == t.dtype and tensors[name].shape == t.shape and same(tensors[name], t)


def test_o_direct_writes_and_buffered_writes_make_the_same_file(tmp_path, monkeypatch):
    """Through an O_DIRECT buffer smaller than the object, and through the page cache where the filesystem refuses
    O_DIRECT: byte for byte the same file, and it reads back."""
    monkeypatch.setattr("sharar.cache.ssd_store.WRITE_BUFFER", 8192)  # several writes per object
    segs, files = segments(seed=3), []
    for name in ("direct", "buffered"):
        store = SsdStore(tmp_path / name, IDENTITY)
        if name == "buffered":
            real = os.open

            def refuse(path, flags, *args):
                if flags & os.O_DIRECT and flags & os.O_WRONLY:
                    raise OSError(errno.EINVAL, "Invalid argument")
                return real(path, flags, *args)
            monkeypatch.setattr("os.open", refuse)
        key = block_hash(store.root, range(BLOCK))
        ((_, _, ok, _),) = put(store, header("kv", key, store.root, 0, range(BLOCK), segs), segs)
        assert ok and store.stats()["direct_writes"] == (name == "direct")
        _, tensors = read(store, "kv", key.hex())
        assert all(same(tensors[n], t) for n, t in segs)
        files.append(store.path("kv", key.hex()).read_bytes())
    assert len(files[0]) > 8192 and files[0] == files[1]


def test_corrupt_or_foreign_files_are_misses_and_are_deleted(tmp_path):
    store = SsdStore(tmp_path, IDENTITY)
    segs = segments(n=5)
    key = tail_hash(store.root, range(5))
    put(store, header("tail", key, store.root, 0, range(5), segs), segs)
    path = store.path("tail", key.hex())
    with open(path, "rb") as f:
        _, offset = store._read_header(f)
    data = bytearray(path.read_bytes())
    data[offset + 7] ^= 0xFF  # one flipped payload byte
    path.write_bytes(bytes(data))
    assert read(store, "tail", key.hex()) is None and store.read_errors == 1
    store.wait()
    assert not path.exists()
    assert read(store, "tail", key.hex()) is None and store.read_errors == 1  # deleted: a plain miss, not another error
    (path.parent / "junk.kv").write_bytes(b"\0" * 100)  # a zero-filled file after a power loss
    (path.parent / "x.kv.tmp").write_bytes(b"partial")
    assert list(store.scan()) == [] and not any(path.parent.iterdir())


def test_a_delete_queued_after_a_write_wins(tmp_path):
    store = SsdStore(tmp_path, IDENTITY)
    segs = segments(n=3)
    key = tail_hash(store.root, range(3))
    store.write(header("tail", key, store.root, 0, range(3), segs), [t for _, t in segs])
    store.delete("tail", key.hex())
    store.wait()
    assert not store.path("tail", key.hex()).exists() and read(store, "tail", key.hex()) is None


def tiny(store, i):
    """A one-token tail object: (header, tensors)."""
    t = torch.full((256,), i % 256, dtype=torch.uint8)
    return header("tail", tail_hash(store.root, [i]), store.root, 0, [i], [("data", t)]), [t]


def test_a_write_queued_after_a_delete_of_its_key_lands(tmp_path):
    store = SsdStore(tmp_path, IDENTITY, writers=4)
    h, tensors = tiny(store, 1)
    store.write(h, tensors)
    store.delete("tail", h.key)  # the budget drops a GPU-resident node's copy; eviction then backs it up again
    store.write(h, tensors)
    store.wait()
    assert store.path("tail", h.key).exists() and (store.writes, store.deletes) == (2, 1)


class Gate:
    """A `ready` event that holds its write (a device copy still running) until opened."""

    def __init__(self):
        self.open = threading.Event()

    def synchronize(self):
        assert self.open.wait(10)


def test_completions_come_back_in_submission_order_whichever_writer_finishes_first(tmp_path):
    store = SsdStore(tmp_path, IDENTITY, writers=4)
    first, tensors = tiny(store, 0)
    gate = Gate()
    store.write(first, tensors, gate)
    later = [x for x in (tiny(store, i) for i in range(1, 64)) if store._queue(x[0].key) is not store._queue(first.key)][:6]
    for h, t in later:
        store.write(h, t)
    for q in {store._queue(h.key) for h, _ in later}:  # the other writers finish theirs while the first is held
        q.join()
    assert all(store.path("tail", h.key).exists() for h, _ in later) and store.completions() == []
    gate.open.set()
    store.wait()
    assert [(key, ok) for _, key, ok, _ in store.completions()] == [(h.key, True) for h in [first, *(h for h, _ in later)]]


def test_the_idle_writer_releases_a_completed_staging_copy(tmp_path):
    store = SsdStore(tmp_path, IDENTITY)
    key = tail_hash(store.root, [1])
    tensor = torch.zeros(256, dtype=torch.uint8)
    copied = weakref.ref(tensor)
    store.write(header("tail", key, store.root, 0, [1], [("data", tensor)]), [tensor])
    del tensor
    store.wait()
    assert copied() is None  # the writer must not hold its last job while blocked on the next queue.get()


def test_flush_starts_a_new_epoch(tmp_path):
    store = SsdStore(tmp_path, IDENTITY)
    segs = segments(n=4)
    key = tail_hash(store.root, range(4))
    put(store, header("tail", key, store.root, 0, range(4), segs), segs)
    old = store.dir
    store.flush()
    assert store.dir != old and read(store, "tail", key.hex()) is None and list(store.scan()) == []
    assert SsdStore(tmp_path, IDENTITY).dir == store.dir  # the epoch survives a restart


def test_the_block_size_is_part_of_the_identity(tmp_path):
    assert SsdStore(tmp_path / "a", IDENTITY, block=1024).root != SsdStore(tmp_path / "b", IDENTITY).root
    with pytest.raises(ValueError):
        SsdStore(tmp_path / "c", IDENTITY, block=300)


def test_a_flipped_header_byte_is_a_miss(tmp_path):
    store = SsdStore(tmp_path, IDENTITY)
    segs = segments(n=5)
    key = tail_hash(store.root, range(5))
    put(store, header("tail", key, store.root, 0, range(5), segs), segs)
    path = store.path("tail", key.hex())
    data = path.read_bytes()
    path.write_bytes(data.replace(b'"tokens": [0, 1, 2', b'"tokens": [0, 1, 3', 1))  # still valid JSON, wrong tokens
    assert read(store, "tail", key.hex()) is None and store.read_errors == 1
    store.wait()
    assert list(store.scan()) == []


@pytest.mark.parametrize("entry", ["read", "scan"])
@pytest.mark.parametrize("damage", ["unknown_dtype", "not_a_dtype", "duplicate_name", "negative_shape", "bad_start", "bad_scalars"])
def test_malformed_metadata_is_a_miss_on_read_and_restart(tmp_path, entry, damage):
    store = SsdStore(tmp_path, IDENTITY)
    segs = segments(n=5)
    key = tail_hash(store.root, range(5))
    h = header("tail", key, store.root, 0, range(5), segs)
    if damage == "unknown_dtype":
        h.segments[0][1] = "floau32"
    elif damage == "not_a_dtype":
        h.segments[0][1] = "__doc__"  # getattr(torch, name) exists, but cannot describe a tensor
    elif damage == "duplicate_name":
        h.segments[1][0] = h.segments[0][0]
    elif damage == "negative_shape":
        h.segments[0][2] = [-5, -1, 2, 8]  # the product still matches the payload size
    elif damage == "bad_start":
        h.start = "0"
    else:
        h.scalars = []
    put(store, h, segs)  # intact payload and CRC, damaged metadata
    if entry == "read":
        assert read(store, "tail", key.hex()) is None and store.read_errors == 1
        store.wait()
    else:
        assert list(store.scan()) == []
    assert not store.path("tail", key.hex()).exists()


def test_a_failed_write_leaves_no_partial_file(tmp_path, monkeypatch):
    store = SsdStore(tmp_path, IDENTITY)
    segs = segments(n=3)
    key = tail_hash(store.root, range(3))

    def full(fd):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr("os.fdatasync", full)
    assert put(store, header("tail", key, store.root, 0, range(3), segs), segs) == [("tail", key.hex(), False, 0)]
    assert store.write_errors == 1 and not any(p.is_file() for p in store.dir.rglob("*"))


def test_a_restart_deletes_epochs_a_crash_left_behind(tmp_path):
    store = SsdStore(tmp_path, IDENTITY)
    store.flush()
    (store.base / "e0").mkdir()  # as if the background delete never ran
    (store.base / "e0" / "junk").write_bytes(b"x")
    restarted = SsdStore(tmp_path, IDENTITY)
    for _ in range(200):
        if not (store.base / "e0").exists():
            break
        time.sleep(0.01)
    assert not (store.base / "e0").exists() and restarted.dir.exists()
