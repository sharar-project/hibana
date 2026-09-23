"""The NVMe session store: parked entries written through, listed after a
restart only under the same identity, read back byte-exact into scattered pool
pages whatever the arena chunking, bounded by a budget, cleared by a flush."""
from types import SimpleNamespace

import torch

from sharar.memory.session_disk import SessionDisk

IDENTITY = {"target": "qwen38", "revision": "r1", "w3": "cal:1"}
CPU = torch.device("cpu")


def entry(key, chunk_sizes, seed=0, layers=3):
    g = torch.Generator().manual_seed(seed)
    chunks, start = [], 0
    for n in chunk_sizes:
        chunks.append((start, torch.randint(0, 255, (layers, 2, n, 4, 8), generator=g, dtype=torch.uint8).view(torch.float8_e4m3fn)))
        start += n
    return SimpleNamespace(
        key=list(key), own=start, chunks=chunks, ready=None, on_disk=False,
        conv=torch.randn(5, 7, generator=g).to(torch.bfloat16), ssm=torch.randn(5, 2, 3, generator=g),
        auxiliary={"__pool__": ((), [("ple_conv", torch.randn(9, 4, generator=g))]),
                   ("draft", "rev"): ((11, 42), [torch.randn(6, 5, generator=g).to(torch.bfloat16)])})


def pools(e, pages):
    """CPU stand-ins for the KV pool (one tensor per layer and K/V) and a state slot."""
    layers, rest = e.chunks[0][1].shape[0], e.chunks[0][1].shape[3:]
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
    whole = torch.cat([c for _, c in e.chunks], dim=2)  # [layers, 2, 50, ...]
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
    import sharar.memory.session_disk as sd
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
    whole = torch.cat([c for _, c in e.chunks], dim=2)
    assert all(same(pool[pages], whole[i // 2, i % 2]) for i, pool in enumerate(kv))


def test_short_sessions_are_not_written(tmp_path):
    store = SessionDisk(tmp_path, 2**30, IDENTITY, CPU, min_tokens=32)
    short, long = entry(range(20), [20]), entry(range(100, 140), [40], seed=1)
    store.submit(short)
    store.submit(long)
    store.wait(long)
    assert not short.on_disk and long.on_disk
    assert store.stats()["too_short"] == 1 and store.stats()["records"] == 1
