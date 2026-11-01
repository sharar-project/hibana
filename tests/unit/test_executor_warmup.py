"""Exercise the real warmup schedule around CPU-only executor operations."""
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.executor import Executor


class WarmupExecutor:
    def __init__(self, max_batch, speculative):
        self.max_batch = max_batch
        self.target = SimpleNamespace(first_text_token=0, vocab_size=1000)
        self.admitted, self.released, self.decoded = [], [], []
        self.live = set()
        self.prefilled = set()
        self.reset_count = self.flush_count = 0
        self.spec = SimpleNamespace(reset_stats=self.reset_stats, steps=16) if speculative else None  # the 27B's 15 drafts

    def admit(self, ids, params):
        req = SimpleNamespace(ids=ids, params=params)
        self.admitted.append(req)
        self.live.add(id(req))
        return req

    def prefill(self, req, max_tokens=None):
        assert id(req) in self.live
        self.prefilled.add(id(req))

    def prefill_batch(self, chunks):
        for req, _ in chunks:
            self.prefill(req)

    def decode_step(self, requests):
        assert 0 < len(requests) <= self.max_batch
        assert all(id(req) in self.live & self.prefilled for req in requests)
        self.decoded.append(list(requests))

    def release(self, req):
        self.live.remove(id(req))  # duplicate release is an error too
        self.released.append(req)

    def reset_stats(self):
        self.reset_count += 1

    def flush_cache(self):
        assert not self.live
        self.flush_count += 1


@pytest.mark.parametrize("max_batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("speculative", [False, True])
def test_warmup_respects_decode_capacity_and_releases_all_requests(monkeypatch, max_batch, speculative):
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronized.append(True))
    ex = WarmupExecutor(max_batch, speculative)

    Executor.warmup(ex)

    assert not ex.live
    assert {id(req) for req in ex.admitted} == {id(req) for req in ex.released}
    assert len(ex.admitted) == len(ex.released)
    assert synchronized == [True]
    assert ex.reset_count == int(speculative)
    if speculative:
        assert max(map(len, ex.decoded)) <= max_batch
        # Existing normal-capacity warmups still cover the single, two, and
        # supported wider paths, plus a real sampled request.
        sizes = [len(batch) for batch in ex.decoded]
        assert sizes[:2] == [1, min(2, max_batch)]
        assert {size for size in (4, 8, 16) if size <= max_batch} <= set(sizes)
        assert ex.decoded[-1][0].params.temperature == 1.
    else:
        assert ex.decoded == []
