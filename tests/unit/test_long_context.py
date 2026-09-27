"""The serving profile admits the target's native context, and pool planning
guarantees one full-context request without reserving it for every slot."""
import pytest

from sharar.engine.profile import MAX_CONTEXT, ServingProfile
from sharar.cache import ledger as ledger_mod
from sharar.cache.ledger import GIB, SHARED_CONTEXT, Ledger
from sharar.target import DEFAULT_TARGET


def test_profile_accepts_the_native_context_and_nothing_beyond():
    assert MAX_CONTEXT == DEFAULT_TARGET.max_position_embeddings == 262144
    assert ServingProfile(max_context=MAX_CONTEXT).max_context == MAX_CONTEXT
    with pytest.raises(ValueError):
        ServingProfile(max_context=MAX_CONTEXT + 1)
    assert ServingProfile().max_context == 32768  # the default profile is unchanged


def plan(monkeypatch, context_len, available_gib=90):
    monkeypatch.setattr(ledger_mod, "mem_available_bytes", lambda: available_gib * GIB)
    return Ledger(start_available=available_gib * GIB).plan(max_running=4, context_len=context_len)


def test_default_profile_plan_is_unchanged(monkeypatch):
    # The 32K profile still guarantees four full requests, as before.
    kv = DEFAULT_TARGET.kv_bytes_per_token
    got = plan(monkeypatch, 32768)
    budget = 90 * GIB - 16 * GIB - 8 * GIB
    must = 4 * 32768 * kv
    spare = int((budget - must - 5 * DEFAULT_TARGET.state_bytes_per_slot) * 0.5)
    assert got.kv_tokens == min(1 << 20, 4 * 32768 + int(spare * 0.4) // kv)


def test_long_context_guarantees_one_request_not_four(monkeypatch):
    got = plan(monkeypatch, MAX_CONTEXT)
    assert got.kv_tokens >= MAX_CONTEXT
    # Four full 262K requests (32 GiB of KV) would not be required up front.
    assert 4 * MAX_CONTEXT * DEFAULT_TARGET.kv_bytes_per_token > 30 * GIB
    tight = plan(monkeypatch, MAX_CONTEXT, available_gib=16 + 8 + 16)
    assert tight.kv_tokens >= MAX_CONTEXT


def test_intermediate_contexts_guarantee_the_larger_of_the_two(monkeypatch):
    kv = DEFAULT_TARGET.kv_bytes_per_token
    assert plan(monkeypatch, 65536).kv_tokens >= max(65536, 4 * SHARED_CONTEXT)
    with pytest.raises(MemoryError):
        plan(monkeypatch, MAX_CONTEXT, available_gib=16 + 8 + (MAX_CONTEXT * kv) // GIB - 1)


def test_wide_batches_split_the_shared_guarantee(monkeypatch):
    # Sixteen running requests (the throughput profile) are guaranteed the same
    # total as four: 8K each at a 32K context, the full context for one request.
    monkeypatch.setattr(ledger_mod, "mem_available_bytes", lambda: 90 * GIB)
    four = Ledger(start_available=90 * GIB).plan(max_running=4, context_len=32768)
    wide = Ledger(start_available=90 * GIB).plan(max_running=16, context_len=32768)
    assert 4 * SHARED_CONTEXT <= wide.kv_tokens <= four.kv_tokens and wide.state_slots >= 17
    assert Ledger(start_available=90 * GIB).plan(max_running=16, context_len=MAX_CONTEXT).kv_tokens >= MAX_CONTEXT
