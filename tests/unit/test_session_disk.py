"""The NVMe session store: parked entries written through, listed after a
restart only under the same identity, read back byte-exact whatever the arena
chunking, bounded by a budget, cleared by a flush."""
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


def blank_like(e, chunk_sizes):
    layers, rest = e.chunks[0][1].shape[0], e.chunks[0][1].shape[3:]
    chunks, start = [], 0
    for n in chunk_sizes:
        chunks.append((start, torch.zeros((layers, 2, n, *rest), dtype=torch.uint8).view(torch.float8_e4m3fn)))
        start += n
    aux = {"__pool__": ((), [("ple_conv", torch.zeros(9, 4))]), ("draft", "rev"): ((11, 42), [torch.zeros(6, 5, dtype=torch.bfloat16)])}
    return SimpleNamespace(key=e.key, own=e.own, chunks=chunks, conv=torch.zeros_like(e.conv), ssm=torch.zeros_like(e.ssm), auxiliary=aux)


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
    dst = blank_like(e, [50])  # one chunk instead of two
    assert restarted.read(record, dst)
    whole = torch.cat([kv for _, kv in e.chunks], dim=2)
    assert same(whole, dst.chunks[0][1])
    assert same(e.conv, dst.conv) and same(e.ssm, dst.ssm)
    assert same(e.auxiliary["__pool__"][1][0][1], dst.auxiliary["__pool__"][1][0][1])
    assert same(e.auxiliary[("draft", "rev")][1][0], dst.auxiliary[("draft", "rev")][1][0])


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
    wrong = blank_like(e, [10])
    wrong.ssm = torch.zeros(5, 2, 4)  # a different geometry
    assert not store.read(record, wrong) and store.stats()["errors"] == 1
