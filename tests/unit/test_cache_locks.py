"""Locks through the real Executor, StateManager, RadixCache, Scheduler and MtpProposer (the stub models of
`test_resume_under_slot_pressure`).

A request has one lock at a time (`req.lock`, a `radix.TreeLock`): taken at its matched node at admission, moved by
`StateManager.checkpoint` after each prefill chunk the drafter has ingested to the deepest snapshot that chunk placed (a
chunk end, then the prompt's last chunk end), and ended at release, retraction or a failed admission. Each move locks the
new node before it unlocks the old one. A running request owns no private snapshot between steps, and the tree holds at
most one of its spare chunk ends: the next checkpoint frees the one before.
"""
import random

import pytest

from hibana.engine.sampler import SamplingParams
from tests.unit.test_resume_under_slot_pressure import depth, draw, harness, locked, run_turn

PARAMS = SamplingParams(max_new_tokens=24, ignore_eos=True)


def last_chunk_end(n, start):
    """Where the forward [start, n) that ends a prompt snapshots: its last 64-token grid point before n (0: none)."""
    s = start + (n - 1 - start) // 64 * 64
    return s if s > start else 0


def first_turn(h, seed, n=650):
    return run_turn(h, draw(random.Random(seed), n)).request


def spare_snapshots(cache):
    return sorted(n.depth for n in cache._walk(l1_only=True) if n.snapshot is not None and n.snapshot.spare)


def test_one_lock_per_request_through_admission_every_chunk_decode_retraction_and_release():
    h = harness(slots=24, max_batch=4)
    cache = h.ex.cache
    first = first_turn(h, "one-lock")
    assert not cache.locks() and spare_snapshots(cache) == [721]  # its reply end
    prompt = first.prompt_ids + draw(random.Random("one-lock-2"), 5000)  # resumes at the first prompt's last chunk end
    req = h.ex.admit(prompt, PARAMS)
    assert req.prefix_len == 640 and req.lock.node is req.match.node and locked(cache, [req]) == 1
    n, depths = req.prefill_len, []
    while not req.prefill_done:
        h.ex.prefill(req, max_tokens=2048)
        depths.append(depth(req))
        assert locked(cache, [req]) == 1 and req.lock.node.state_lock_ref == 1 and not req.private
        own = [depths[-1]] if req.lock.node.snapshot.spare else []
        assert spare_snapshots(cache) == sorted(own + [721]), "one spare chunk end of its own: its lock's"
        h.ex.state.check()
    # every chunk moved the lock: its chunk end (2688, 4736), then the last chunk end in the last forward
    assert depths == [2688, 4736, last_chunk_end(n, 4736)] and not req.private
    assert cache.node_at(prompt[:2688]).snapshot is None, "the earlier spare chunk end went at the next checkpoint"
    for _ in range(3):
        h.ex.decode_step([req])
        assert locked(cache, [req]) == 1 and depth(req) == depths[-1]
    h.ex.retract(req)  # the scheduler's retraction under memory pressure
    assert req.lock is None and not cache.locks()
    h.ex.admit(req.prompt_ids, req.params, req)  # the replay resumes at its own last chunk end
    assert req.prefix_len == depths[-1] and depth(req) == depths[-1] and locked(cache, [req]) == 1
    while not req.prefill_done:
        h.ex.prefill(req)
        assert locked(cache, [req]) == 1
    while not req.finished:
        h.ex.decode_step([req])
    h.ex.release(req)
    assert not cache.locks() and all(n.lock_ref == 0 for n in cache._walk())
    h.ex.state.check()


def test_batched_prefill_keeps_one_lock_per_request():
    h = harness(slots=24, max_batch=4)
    cache = h.ex.cache
    first = first_turn(h, "batched")
    a = h.ex.admit(first.prompt_ids + draw(random.Random("a"), 3000), PARAMS)  # resumes: locks the matched node
    b = h.ex.admit(draw(random.Random("b"), 3000), PARAMS)  # resumes nowhere: no lock until its first checkpoint
    assert (a.lock.node is a.match.node, b.lock) == (True, None) and locked(cache, [a, b]) == 1
    while not (a.prefill_done and b.prefill_done):
        h.ex.prefill_batch([(r, min(1024, r.prefill_len - r.prefill_pos)) for r in (a, b) if not r.prefill_done])
        assert locked(cache, [a, b]) == sum(r.lock is not None for r in (a, b)) and not a.private and not b.private
        h.ex.state.check()
    # 1,024-token chunks: a's last starts at 2688 and b's at 2048, and each places its prompt's last chunk end
    assert locked(cache, [a, b]) == 2 and (depth(a), depth(b)) == (last_chunk_end(3650, 2688), last_chunk_end(3000, 2048))
    for r in (a, b):
        h.ex.release(r)
    assert not cache.locks()


@pytest.mark.parametrize("others", [0, 2], ids=["solo", "batched"])
def test_the_lock_never_moves_before_the_first_ingest(others):
    """With the drafter on, the request's lock leaves its matched node only after the drafter read that node at the
    first prefill chunk, alone or in a batched prefill (unrelated prompts prefilled alongside)."""
    h = harness(slots=16, max_batch=4)
    events = []  # (request, event, lock depth)
    ingest, checkpoint = h.proposer.ingest_prefill, h.ex.state.checkpoint

    def recording_ingest(req, new_ids, start, result):
        if not req.output_ids:  # prefill (its first token is sampled after the last chunk's ingest), not decode
            events.append((req, "ingest", depth(req) or 0))
        return ingest(req, new_ids, start, result)

    def recording_checkpoint(req, up_to):
        before = depth(req)
        checkpoint(req, up_to)
        if depth(req) != before:
            events.append((req, "move", depth(req)))
    h.proposer.ingest_prefill = recording_ingest
    h.ex.state.checkpoint = recording_checkpoint
    prompts = [draw(random.Random("order"), 650)]
    for t in range(1, 5):  # every other turn adds more than a chunk (8,192 tokens alone, 4,096 batched)
        prompts.append(prompts[-1] + draw(random.Random(t), 70 + 9000 * (t % 2)))
    turns = [run_turn(h, p, others=others) for p in prompts]
    requests = list(dict.fromkeys(r for r, _, _ in events))
    assert len(requests) == len(turns) * (1 + others)  # each turn and its unrelated prompts
    for req in requests:
        mine = [e for _, e, _ in ((r, e, at) for r, e, at in events if r is req)]
        first = next(at for r, e, at in events if r is req)
        # the first ingest reads the matched node while the request's lock is still there; every move comes after it
        # (each prompt adds more than a grid past its match, so its first forward places a snapshot)
        assert mine[0] == "ingest" and first == req.prefix_len and "move" in mine, mine
    for turn in turns[1:]:
        assert turn.drafter == "hit" and turn.locked_at_ingest
    assert any(len([1 for r, e, _ in events if r is t.request and e == "move"]) > 1 for t in turns)  # chunked


def test_a_chunked_prompt_moves_its_lock_per_chunk_and_keeps_one_spare_chunk_end():
    h = harness(slots=32, max_batch=4)
    cache = h.ex.cache
    prompt = draw(random.Random("chunks"), 9000)
    req = h.ex.admit(prompt, PARAMS)
    seen = []
    while not req.prefill_done:
        h.ex.prefill(req, max_tokens=2048)
        seen.append(depth(req))
        assert len(spare_snapshots(cache)) <= 1
        if len(seen) == 2:  # retracted mid-prefill: the retry resumes at its spare chunk end, no longer spare then
            h.ex.retract(req)
            h.ex.admit(req.prompt_ids, req.params, req)
            assert req.prefix_len == 4096 and not req.lock.node.snapshot.spare
    assert seen == [2048, 4096, 6144, 8192, last_chunk_end(9000, 8192)]
    assert [p for p in seen[:-1] if cache.node_at(prompt[:p]).snapshot is not None] == [4096], \
        "only the chunk end it resumed from (not spare after the resume) outlives the next checkpoint"
    assert not req.lock.node.snapshot.spare and not req.private and locked(cache, [req]) == 1
    h.ex.release(req)
    assert not cache.locks()


def agent_conversation(seed, turns=8, header=650, user=50, rerender=30, reply=70, tools=(300, 150, 500, 200)):
    """An agent's prompts as its chat template renders them again each turn, with the chat API's message boundary
    (the start of the prompt's last message): turn t keeps turn t-1's prompt but its last `rerender` tokens (the
    generation prompt, rendered again with the recorded reply), then the reply and a tool result, its last message. The
    first turn's last message is its user message. The server's own output (all TOKEN) is never part of the next
    prompt. Returns (prompt, message boundary) per turn."""
    rng = random.Random(seed)
    out = [(draw(rng, header), header - user)]
    for t in range(1, turns):
        prev = out[-1][0][:-rerender] + draw(rng, rerender + reply)
        out.append((prev + draw(rng, tools[(t - 1) % len(tools)]), len(prev)))
    return out


def test_each_agent_turn_resumes_at_the_previous_turns_message_boundary_or_last_chunk_end_under_slot_pressure():
    """Turn k+1 shares turn k's prompt only up to 30 tokens before its end, so turn k's reply end cannot serve it, nor
    its last chunk end when that falls in the last 30 tokens: it resumes at the deepest of turn k's two not-spare
    snapshots it keeps, its message boundary (the start of its last message, from the chat API, on its forward's grid)
    or its last chunk end. Slot pressure keeps both (it takes the spare reply ends first), and the turn drafts from
    the one it resumes at."""
    h = harness(slots=8, max_batch=4)
    cache = h.ex.cache
    turns, placed = [], []
    for prompt, boundary in agent_conversation("agent"):
        turns.append(run_turn(h, prompt, boundary=boundary))
        start, n = turns[-1].resume, len(prompt)  # one forward from its match: the solo chunk holds the new tokens
        placed.append((start + (boundary - start) // 64 * 64, start + (n - 1 - start) // 64 * 64, n - 30))
    assert cache.taken > 0, "precondition: slot pressure took snapshots"
    expected = [max(p for p in (at_boundary, last) if p <= kept) for at_boundary, last, kept in placed]
    assert [t.resume for t in turns[1:]] == expected[:-1]
    assert any(last > kept for _, last, kept in placed[:-1]), "precondition: some turn needed its message boundary"
    assert all(t.drafter == "hit" and t.kept and t.drafted and t.state_exact for t in turns[1:])
    assert (h.proposer.cache_hits, h.proposer.cache_fallbacks) == (len(turns) - 1, 0)
    assert not cache.locks()
