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


def test_default_profile_plan(monkeypatch):
    # The 32K profile guarantees four full requests; all spare memory but the
    # 4 GiB margin goes to the pools, slots first (up to their cap), then pages.
    kv, state = DEFAULT_TARGET.kv_bytes_per_token, DEFAULT_TARGET.state_bytes_per_slot
    got = plan(monkeypatch, 32768)
    budget = 90 * GIB - 16 * GIB - 8 * GIB
    must = 4 * 32768 * kv
    spare = budget - must - 5 * state - 4 * GIB
    assert got.state_slots == min(96, 5 + int(spare * 0.6) // state)
    assert got.kv_tokens == min(1 << 20, 4 * 32768 + (spare - (got.state_slots - 5) * state) // kv)


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



def test_a_tight_budget_keeps_at_least_half_of_what_is_left_for_the_caches(monkeypatch):
    """Little left after the guarantees (Flash-Next `exact`: NVFP4 experts and the SSD staging): the 4 GiB margin
    alone would leave the pools only the guaranteed slots; they get half of what is left, as before the margin."""
    kv, state = DEFAULT_TARGET.kv_bytes_per_token, DEFAULT_TARGET.state_bytes_per_slot
    must = 4 * 32768 * kv + 9 * state
    available = 16 * GIB + 8 * GIB + must + 3 * GIB  # 3 GiB left after the guarantees, less than the margin
    got = plan(monkeypatch, 32768, available_gib=available / GIB)
    assert got.state_slots == min(96, 9 + int((3 * GIB // 2) * 0.6) // state) > 9


def test_the_guaranteed_slots_cover_each_request_and_its_prompt_end_snapshot(monkeypatch):
    """Nothing left after the guarantees: still two slots per running request and the dummy (warmup runs four
    requests with a snapshot each), so a start that fits works, and one that does not fails at planning."""
    kv, state = DEFAULT_TARGET.kv_bytes_per_token, DEFAULT_TARGET.state_bytes_per_slot
    must = 4 * 32768 * kv + 9 * state
    assert plan(monkeypatch, 32768, available_gib=(16 * GIB + 8 * GIB + must) / GIB).state_slots == 9
    with pytest.raises(MemoryError):
        plan(monkeypatch, 32768, available_gib=(16 * GIB + 8 * GIB + must - 2 * state) / GIB)
