"""The serving profile admits the target's native context, and pool planning
guarantees one full-context request without reserving it for every slot."""
import pytest

from hibana.engine.profile import MAX_CONTEXT, ServingProfile
from hibana.cache import ledger as ledger_mod
from hibana.cache.ledger import CACHED_SNAPSHOTS, GIB, SHARED_CONTEXT, Ledger
from hibana.target import DEFAULT_TARGET

FLOOR = ledger_mod.host_floor_bytes()  # the host floor the plans leave (SHARAR_HOST_FLOOR_GIB)
LEFT = FLOOR + ledger_mod.PLAN_SLACK_BYTES  # what a plan leaves unplanned: the floor and the slack for later reads


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
    # The 32K profile guarantees four full requests (and two slots each, plus the dummy); everything but the floor
    # (and the slack) goes to the pools (nothing is allocated after them here), split by what a cached 32K prompt costs (its pages and
    # two snapshots), with 96 slots at least; pages take the rest, uncapped.
    kv, state = DEFAULT_TARGET.kv_bytes_per_token, DEFAULT_TARGET.state_bytes_per_slot
    got = plan(monkeypatch, 32768)
    budget = int((90 * GIB - LEFT) / ledger_mod.POOL_COST)  # each pool byte costs POOL_COST of MemAvailable
    must = 4 * 32768 * kv
    spare = budget - must - 9 * state
    share = CACHED_SNAPSHOTS * state / (CACHED_SNAPSHOTS * state + SHARED_CONTEXT * kv)
    assert got.state_slots == min(9 + int(spare * 0.6) // state, max(96, 9 + int(spare * share) // state))
    assert got.kv_tokens == 4 * 32768 + (spare - (got.state_slots - 9) * state) // kv
    assert got.kv_tokens > 1 << 20 and got.after_pools_bytes == 0  # the old 2^20 cap no longer applies


def test_what_is_allocated_after_the_pools_comes_off_the_plan(monkeypatch):
    monkeypatch.setattr(ledger_mod, "mem_available_bytes", lambda: 90 * GIB)
    bare = Ledger(start_available=90 * GIB).plan(max_running=4, context_len=32768)
    after = Ledger(start_available=90 * GIB).plan(max_running=4, context_len=32768, future_bytes=6 * GIB)
    pages_and_slots = lambda p: p.kv_tokens * DEFAULT_TARGET.kv_bytes_per_token + p.state_slots * DEFAULT_TARGET.state_bytes_per_slot  # noqa: E731
    assert after.after_pools_bytes == 6 * GIB
    assert 5.9 * GIB / ledger_mod.POOL_COST < pages_and_slots(bare) - pages_and_slots(after) <= 6 * GIB / ledger_mod.POOL_COST
    with pytest.raises(MemoryError, match="SHARAR_HOST_FLOOR_GIB"):
        Ledger(start_available=90 * GIB).plan(max_running=4, context_len=32768, future_bytes=90 * GIB)


def test_after_plan_parts():
    parts64, parts128 = ledger_mod.after_plan_bytes(64), ledger_mod.after_plan_bytes(128)
    assert set(parts64) == {"workspaces", "graphs", "first_use", "activations"}
    assert parts64["workspaces"] == 512 * 2**20 and parts128["workspaces"] == 896 * 2**20  # FlashInfer's, as allocated
    assert parts128["graphs"] > parts64["graphs"] >= ledger_mod.graph_bytes(4) >= GIB // 2
    assert parts64["first_use"] == parts128["first_use"] and parts64["activations"] == parts128["activations"]


def test_admission_holds_at_half_the_floor(monkeypatch):
    led = Ledger(start_available=90 * GIB)
    monkeypatch.setattr(ledger_mod, "mem_available_bytes", lambda: FLOOR // 2 + 1)
    assert led.above_floor()  # under the floor but above half: other programs are using what the plan left them
    monkeypatch.setattr(ledger_mod, "mem_available_bytes", lambda: FLOOR // 2 - 1)
    assert not led.above_floor()


def test_an_explicit_kv_target_still_caps_the_pool(monkeypatch):
    monkeypatch.setattr(ledger_mod, "mem_available_bytes", lambda: 90 * GIB)
    capped = Ledger(start_available=90 * GIB).plan(max_running=4, context_len=32768, kv_token_target=1 << 20)
    assert capped.kv_tokens == 1 << 20


def test_a_wide_batch_with_compact_state_gets_slots_for_its_cached_prompts(monkeypatch):
    # The throughput profile: 128 requests, INT8 state (about 3.4x smaller than FP32). The guaranteed 257 slots are
    # kept and the spare memory still buys slots for cached prompts' resume points, not pages alone.
    monkeypatch.setattr(ledger_mod, "mem_available_bytes", lambda: 60 * GIB)
    int8 = DEFAULT_TARGET.state_bytes_per_slot * 10 // 34
    got = Ledger(start_available=60 * GIB).plan(max_running=128, context_len=262144, state_bytes_per_slot=int8)
    assert got.state_slots > 257 and got.kv_tokens > 262144


def test_long_context_guarantees_one_request_not_four(monkeypatch):
    got = plan(monkeypatch, MAX_CONTEXT)
    assert got.kv_tokens >= MAX_CONTEXT
    # Four full 262K requests (32 GiB of KV) would not be required up front.
    assert 4 * MAX_CONTEXT * DEFAULT_TARGET.kv_bytes_per_token > 30 * GIB
    tight = plan(monkeypatch, MAX_CONTEXT, available_gib=LEFT / GIB + 16)
    assert tight.kv_tokens >= MAX_CONTEXT


def test_intermediate_contexts_guarantee_the_larger_of_the_two(monkeypatch):
    kv = DEFAULT_TARGET.kv_bytes_per_token
    assert plan(monkeypatch, 65536).kv_tokens >= max(65536, 4 * SHARED_CONTEXT)
    with pytest.raises(MemoryError):
        plan(monkeypatch, MAX_CONTEXT, available_gib=LEFT / GIB + (MAX_CONTEXT * kv) // GIB - 1)


def test_wide_batches_split_the_shared_guarantee(monkeypatch):
    # Sixteen running requests (the throughput profile) are guaranteed the same
    # total as four: 8K each at a 32K context, the full context for one request.
    monkeypatch.setattr(ledger_mod, "mem_available_bytes", lambda: 90 * GIB)
    four = Ledger(start_available=90 * GIB).plan(max_running=4, context_len=32768)
    wide = Ledger(start_available=90 * GIB).plan(max_running=16, context_len=32768)
    assert 4 * SHARED_CONTEXT <= wide.kv_tokens <= four.kv_tokens and wide.state_slots >= 17
    assert Ledger(start_available=90 * GIB).plan(max_running=16, context_len=MAX_CONTEXT).kv_tokens >= MAX_CONTEXT



def test_a_tight_budget_gives_the_caches_all_that_is_left(monkeypatch):
    """Little left after the guarantees (Flash-Next `exact`: NVFP4 experts and the SSD staging): with what is
    allocated after the pools counted, there is no margin to take, and the caches get all of it."""
    kv, state = DEFAULT_TARGET.kv_bytes_per_token, DEFAULT_TARGET.state_bytes_per_slot
    must = 4 * 32768 * kv + 9 * state
    available = LEFT + (must + 3 * GIB) * ledger_mod.POOL_COST  # 3 GiB of pools left after the guarantees
    got = plan(monkeypatch, 32768, available_gib=available / GIB)
    assert got.state_slots == min(96, 9 + int(3 * GIB * 0.6) // state) > 9


def test_the_guaranteed_slots_cover_each_request_and_its_prompt_end_snapshot(monkeypatch):
    """Nothing left after the guarantees: still two slots per running request and the dummy (warmup runs four
    requests with a snapshot each), so a start that fits works, and one that does not fails at planning."""
    kv, state = DEFAULT_TARGET.kv_bytes_per_token, DEFAULT_TARGET.state_bytes_per_slot
    must = 4 * 32768 * kv + 9 * state
    cost = ledger_mod.POOL_COST  # MemAvailable a pool byte takes
    assert plan(monkeypatch, 32768, available_gib=(LEFT + must * cost + 64) / GIB).state_slots == 9
    with pytest.raises(MemoryError):
        plan(monkeypatch, 32768, available_gib=(LEFT + (must - 2 * state) * cost) / GIB)
