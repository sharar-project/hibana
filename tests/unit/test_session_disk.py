"""The NVMe session store: parked entries written through, listed after a
restart only under the same identity, read back byte-exact into scattered pool
pages whatever the arena chunking, bounded by a budget, cleared by a flush."""
from types import SimpleNamespace

import torch

from sharar.cache.session_disk import SessionDisk

IDENTITY = {"target": "qwen38", "revision": "r1", "w3": "cal:1"}
CPU = torch.device("cpu")


def entry(key, chunk_sizes, seed=0, layers=3):
    g = torch.Generator().manual_seed(seed)
    chunks, start = [], 0
    for n in chunk_sizes:
        chunks.append((start, torch.randint(0, 255, (layers, 2, n, 4, 8), generator=g, dtype=torch.uint8).view(torch.float8_e4m3fn), {}))
        start += n
    return SimpleNamespace(
        key=list(key), own=start, chunks=chunks, ready=None, on_disk=False,
        conv=torch.randn(5, 7, generator=g).to(torch.bfloat16), ssm=torch.randn(5, 2, 3, generator=g),
        auxiliary={"__pool__": ((), [("ple_conv", torch.randn(9, 4, generator=g))]),
                   ("draft", "rev"): ((11, 42), [torch.randn(6, 5, generator=g).to(torch.bfloat16)])})


def pools(e, pages):
    """CPU stand-ins for the KV pool (one tensor per layer and K/V) and a state slot."""
    layers, rest = e.chunks[0][1].shape[0], e.chunks[0][1].shape[3:]  # chunks: (first, kv, page aux)
    kv = [torch.zeros((pages, *rest), dtype=torch.uint8).view(torch.float8_e4m3fn) for _ in range(2 * layers)]
    state = {"conv": torch.zeros_like(e.conv), "ssm": torch.zeros_like(e.ssm), "pool.ple_conv": torch.zeros(9, 4)}
    return kv, state


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def test_round_trip_across_a_restart_and_a_different_chunking(tmp_path):
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)
    e = entry(range(100, 150), [30, 20])
    store.submit(e)
    store.wait(e)
    assert e.on_disk and store.stats()["writes"] == 1 and store.stats()["errors"] == 0
    restarted = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)  # a new process lists the record
    (record,) = restarted.find(list(range(100, 160)))
    assert record.key == e.key and record.own == 50 and not restarted.find(list(range(100, 140)))
    assert record.aux_specs({("draft", "rev"): None}) == [(("draft", "rev"), [11, 42], [(torch.bfloat16, (6, 5))])]
    kv, state = pools(e, 128)
    pages = torch.randperm(128)[:50 - 7]  # restore tokens [7:] into scattered pages
    payloads = restarted.restore_into(record, 7, pages, kv, state, record.aux_specs({("draft", "rev"): None}))
    whole = torch.cat([c[1] for c in e.chunks], dim=2)  # [layers, 2, 50, ...]
    for i, pool in enumerate(kv):
        assert same(pool[pages], whole[i // 2, i % 2, 7:])
    assert same(e.conv, state["conv"]) and same(e.ssm, state["ssm"])
    assert same(e.auxiliary["__pool__"][1][0][1], state["pool.ple_conv"])
    (scalars, (feat,)) = payloads[("draft", "rev")]
    assert scalars == (11, 42) and same(e.auxiliary[("draft", "rev")][1][0], feat)


def test_other_identities_are_not_listed_and_covered_keys_are_not_rewritten(tmp_path):
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)
    long = entry(range(0, 40), [40])
    store.submit(long)
    store.wait(long)
    short = entry(range(0, 25), [25], seed=1)
    store.submit(short)  # a stored record already extends it
    assert short.on_disk and store.stats()["skipped"] == 1 and store.stats()["records"] == 1
    assert not SessionDisk(tmp_path, 2**30, {**IDENTITY, "w3": "nearest"}, CPU).records


def test_budget_drops_least_recently_used_and_flush_clears(tmp_path):
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)
    for i in range(4):
        e = entry(range(1000 * (i + 1), 1000 * (i + 1) + 64), [64], seed=i)
        store.submit(e)
        store.wait(e)
        if i == 0:
            store.budget = int(store.used_bytes * 2.5)  # room for two records of this size
    keys = sorted(r.key[0] for r in store.records.values())
    assert keys == [3000, 4000] and store.stats()["deleted"] == 2
    store.clear()
    assert not store.records and not list(tmp_path.glob("*.bin"))


def test_a_mismatched_segment_fails_the_read(tmp_path):
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)
    e = entry(range(10), [10])
    store.submit(e)
    store.wait(e)
    (record,) = store.find(list(range(12)))
    kv, state = pools(e, 16)
    state["ssm"] = torch.zeros(5, 2, 4)  # a different geometry
    assert store.restore_into(record, 0, torch.arange(10), kv, state, []) is None and store.stats()["errors"] == 1


def test_segments_larger_than_the_staging_buffer_round_trip(tmp_path, monkeypatch):
    import sharar.cache.session_disk as sd
    monkeypatch.setattr(sd, "STAGING", 64)  # every segment spans several staging chunks
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)
    e = entry(range(300, 340), [40])
    store.submit(e)
    store.wait(e)
    (record,) = store.find(list(range(300, 345)))
    kv, state = pools(e, 64)
    pages = torch.arange(40)
    assert store.restore_into(record, 0, pages, kv, state, []) is not None
    assert same(e.ssm, state["ssm"]) and same(e.conv, state["conv"])
    whole = torch.cat([c[1] for c in e.chunks], dim=2)
    assert all(same(pool[pages], whole[i // 2, i % 2]) for i, pool in enumerate(kv))



def test_page_aux_rows_travel_with_the_session(tmp_path):
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)
    e = entry(range(500, 530), [18, 12])
    g = torch.Generator().manual_seed(9)
    rows = torch.randn(30, 2, 16, generator=g).to(torch.bfloat16)  # e.g. raw and compressed index keys
    e.chunks = [(first, kv, {"qsa.3": rows[first:first + kv.shape[2]]}) for first, kv, _ in e.chunks]
    store.submit(e)
    store.wait(e)
    (record,) = store.find(list(range(500, 535)))
    kv, state = pools(e, 64)
    aux_pool = {"qsa.3": torch.zeros(64, 2, 16, dtype=torch.bfloat16)}
    pages = torch.randperm(64)[:30 - 4]
    assert store.restore_into(record, 4, pages, kv, state, [], page_aux=aux_pool) is not None
    assert same(aux_pool["qsa.3"][pages], rows[4:])


def test_short_sessions_are_not_written(tmp_path):
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU, min_tokens=32)
    short, long = entry(range(20), [20]), entry(range(100, 140), [40], seed=1)
    store.submit(short)
    store.submit(long)
    store.wait(long)
    assert not short.on_disk and long.on_disk
    assert store.stats()["too_short"] == 1 and store.stats()["records"] == 1


def _round_trip(store, tmp_path):
    e = entry(range(500, 537), [37])  # a 148-byte key: no segment starts on an O_DIRECT boundary
    store.submit(e)
    store.wait(e)
    (record,) = store.find(list(range(500, 540)))
    kv, state = pools(e, 64)
    pages = torch.randperm(64)[:37]
    assert store.restore_into(record, 0, pages, kv, state, []) is not None
    whole = e.chunks[0][1]
    assert all(same(pool[pages], whole[i // 2, i % 2]) for i, pool in enumerate(kv))
    assert same(e.ssm, state["ssm"]) and same(e.conv, state["conv"])


def test_reads_fall_back_to_plain_reads_without_o_direct(tmp_path):
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)
    store._direct = False  # as on a filesystem that refuses O_DIRECT opens
    _round_trip(store, tmp_path)
    assert store.stats()["errors"] == 0


def test_reads_fall_back_when_direct_reads_are_refused(tmp_path, monkeypatch):
    """A filesystem that opens with O_DIRECT but refuses the reads (EINVAL): the read is redone plainly, and later
    reads skip O_DIRECT."""
    import errno
    import fcntl
    import os
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU)
    real, calls = os.preadv, {"direct": 0}

    def preadv(fd, buffers, offset):
        if fcntl.fcntl(fd, fcntl.F_GETFL) & getattr(os, "O_DIRECT", 0):
            calls["direct"] += 1
            raise OSError(errno.EINVAL, "refused")
        return real(fd, buffers, offset)

    monkeypatch.setattr(os, "preadv", preadv)
    _round_trip(store, tmp_path)
    assert store._direct is False and store.stats()["errors"] == 0
    assert calls["direct"] == (1 if hasattr(os, "O_DIRECT") else 0)
