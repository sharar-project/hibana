"""MTP drafting on prompts resumed from the prefix cache, under state-slot pressure.

Slot pressure evicts every snapshot no lock sits on, including those on a running request's locked path (a lock keeps
its whole path's KV, but only its own node's snapshot). A request resuming from the cache copies its matched node's snapshot at admission, but the drafter reads that
node's metadata (MTP: the saved final-norm row of the resume position) only at the request's first prefill chunk, and
the prefill checkpoints planned in between (the request's own, or another prompt's) evict snapshots. The request's one
lock (`req.lock`, a `radix.TreeLock`), taken at the matched node at admission, therefore stays there until that chunk has
been ingested; only then does `StateManager.checkpoint` move it to the request's deepest new snapshot. Without that the
node is stripped first: the drafter misses, and (before the stand-in row) the request never drafted and cached its own
snapshots without rows, so later turns missed too.

Sharar's own Scheduler, Executor, StateManager, RadixCache, StatePool, KvPool allocator, SpecDecoder and MtpProposer
run; only the models are stubs:
- the target is an exact recurrence per state slot, s = (31 s + token) mod 97, written to the slot and to every
  prefill checkpoint the executor plans; a token's final-norm row is (s after it, its position); the logits pick 7;
- the MTP head's output is its input and its logits pick 7 (every draft is accepted). Its KV is one row per page:
  each row the proposer writes from the target is checked against the target's row of the position before, and every
  draft step checks the committed rows it attends to.
"""
import dataclasses
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hibana
from hibana.cache.kv_pool import KvPool
from hibana.cache.radix import RadixCache
from hibana.cache.request_state import StateManager
from hibana.cache.state_pool import StatePool
from hibana.engine.contracts import ProposerContext
from hibana.engine.executor import Executor
from hibana.engine.request import Request
from hibana.engine.sampler import Sampler, SamplingParams
from hibana.engine.scheduler import Scheduler, SchedulerConfig
from hibana.proposers.draft_head import TargetHead
from hibana.proposers.mtp import MtpProposer
from hibana.target import FLASHNEXT

CPU = torch.device("cpu")
TOKEN = 7  # what both stubs predict; prompts never contain it
STATE = dataclasses.replace(FLASHNEXT, num_layers=1, layer_types=("linear_attention",), linear_num_key_heads=1,
                            linear_num_value_heads=1, linear_key_head_dim=1, linear_value_head_dim=1,
                            linear_conv_kernel=2)
UNWRITTEN, EXACT, INEXACT, SPECULATIVE = range(4)  # a draft-KV row's state
BUILD = Path(hibana.__file__).resolve().parent


def recurrence(ids, s=0):
    for t in ids:
        s = (s * 31 + t) % 97
    return s


class Target:
    """The recurrence per state slot, written to the slot and to every planned prefill checkpoint (verification writes
    per-step states instead, as the GDN kernels do). Final-norm row: (s after the token, its position)."""

    def __init__(self, pool):
        self.pool = pool

    def state(self, slot):
        return int(self.pool.ssm_all[0, slot].reshape(-1)[0].item())

    def set_state(self, slot, s):
        self.pool.ssm_all[0, slot] = s
        self.pool.conv_all[0, slot] = s

    def forward(self, ids, ctx):
        ids, positions = ids.tolist(), ctx.positions.tolist()
        if ctx.mode == "decode":
            groups = [(i, i + 1, slot) for i, slot in enumerate(ctx.slots.tolist())]
        elif ctx.cu_seqlens is not None:
            cu = ctx.cu_seqlens.tolist()
            groups = [(cu[i], cu[i + 1], slot) for i, slot in enumerate(ctx.seq_slots.tolist())]
        else:
            groups = [(0, len(ids), ctx.slot)]
        snapshots = {}
        if ctx.mode == "prefill":
            seqs = ctx.checkpoint_seqs or [0] * len(ctx.checkpoint_positions)
            snapshots = {(g, p): slot for g, p, slot in zip(seqs, ctx.checkpoint_positions, ctx.checkpoint_slots)}
        rows = []
        for g, (lo, hi, slot) in enumerate(groups):
            s = self.state(slot)
            for i in range(lo, hi):
                s = (s * 31 + ids[i]) % 97
                if ctx.mode == "verify":
                    ctx.inter_ssm[0][g, i - lo] = s
                    ctx.verify_conv_windows[0][g, i - lo] = s
                elif (snap := snapshots.get((g, positions[i] + 1))) is not None:
                    self.set_state(snap, s)
                rows.append((s, positions[i]))
            if ctx.mode != "verify":
                self.set_state(slot, s)
        return {"norm": torch.tensor(rows, dtype=torch.float32).reshape(-1, 2)}

    @staticmethod
    def logits(hidden):
        out = torch.zeros((len(hidden), 128))
        out[:, TOKEN] = 10
        return out


class Head:
    """The MTP layer. `write_kv` (prefill and accepted tokens) checks each row against the exact value for its position
    p: the target's row of position p - 1. A draft step writes its own row as speculative and counts reads of committed
    rows (positions 1 .. the round's start - 1) that are unwritten, or inexact or speculative."""
    bytes = 0
    target = FLASHNEXT

    def __init__(self, pages):
        self.status = torch.zeros(pages, dtype=torch.int8)
        self.writer = None  # the request whose rows the proposer is writing
        self.round_start = []  # per row of a drafting round, its first uncommitted position
        self.states = {}  # request id -> recurrence after p tokens, p = 0, 1, ...
        self.inexact = []  # (request id, position) of each inexact row written
        self.unwritten_reads = self.inexact_reads = 0
        self.draft_head = TargetHead(self.logits)

    def exact_rows(self, key, seq, positions):
        states = self.states.setdefault(key, [0])
        for p in range(len(states), max(positions) + 1):
            states.append((states[-1] * 31 + seq[p - 1]) % 97)
        return torch.tensor([(states[p], p - 1) for p in positions], dtype=torch.float32)

    def write_kv(self, tokens, hidden, positions, pages):
        req = self.writer
        assert req is not None, "draft KV written outside prefill and accepted-token feedback"
        pos, idx = positions.tolist(), pages.long()
        exact = (hidden.float() == self.exact_rows(req.request_id, req.prompt_ids + req.output_ids, pos)).all(dim=1)
        self.status[idx] = EXACT
        self.status[idx[~exact]] = INEXACT
        self.inexact += [(req.request_id, pos[i]) for i in (~exact).nonzero().flatten().tolist()]

    def forward(self, tokens, hidden, positions, pages, qo_lens, kv_pages):
        self.status[pages.long()] = SPECULATIVE
        for start, kv in zip(self.round_start, kv_pages):
            committed = self.status[kv[:start - 1].long()]
            self.unwritten_reads += int((committed == UNWRITTEN).any())
            self.inexact_reads += int(((committed == INEXACT) | (committed == SPECULATIVE)).any())
        return hidden

    logits = staticmethod(Target.logits)

    @staticmethod
    def head_input(hidden):
        return hidden

    def freed(self, pages):
        if pages:
            self.status[torch.tensor(pages, dtype=torch.long)] = UNWRITTEN


class CountingCache(RadixCache):
    """The radix cache, counting every snapshot slot pressure takes: an interior one, or an unlocked leaf that went whole
    (`snapshot_evictions` counts only the former)."""
    taken = 0

    def _evict_snapshot(self, node, reason):
        self.taken += 1
        super()._evict_snapshot(node, reason)


class HostKvPool(KvPool):
    """KvPool's allocator without K/V storage or its pinned (CUDA) upload; freed pages are unwritten draft KV again."""

    def __init__(self, num_pages, on_free):
        super().__init__(num_pages, CPU, layer_ids=[])
        self.on_free = on_free

    def alloc_with_ids(self, n):
        if n > len(self._free):
            raise MemoryError(f"KV pool exhausted: need {n} pages, {len(self._free)} free")
        cut = len(self._free) - n
        pages = self._free[cut:][::-1]
        del self._free[cut:]
        return torch.tensor(pages, dtype=torch.int32), pages

    def free(self, pages):
        ids = pages.tolist()
        super().free(pages)
        self.on_free(ids)

    def free_ids(self, pages):
        super().free_ids(pages)
        self.on_free(list(pages))


@dataclasses.dataclass(eq=False)
class Turn:
    request: object
    resume: int = 0
    state_exact: bool = True
    evicted: int = 0  # snapshots slot pressure took while the turn ran
    admitted_at: int = 0  # cache.taken when it was admitted
    window: int | None = None  # snapshots evicted between its admission and its first prefill ingest
    snapshot: bool | None = None  # its matched node still held its snapshot at the first prefill ingest
    row: bool | None = None  # ... and the drafter's row for the resume position (no row with a snapshot: a cascade)
    locked_at_ingest: bool | None = None  # the request's lock was still on its matched node at that ingest
    lock_after: int | None = None  # where the request's lock was once its first prefill chunk had finished
    locks: list = dataclasses.field(default_factory=list)  # locks in the whole tree after each of its prefill chunks
    drafter: str = "fresh"  # or hit / MISS
    ready: bool | None = None

    @property
    def drafted(self):
        return bool(self.ready) and self.request.spec_rounds > 0

    @property
    def kept(self):
        return bool(self.snapshot and self.row)


def has_row(node, ident, position):
    value = node.snapshot.auxiliary.get(ident) if node.snapshot is not None else None
    return value is not None and value[0] == position


def nodes(cache):
    stack = list(cache.root.children.values())
    while stack:
        n = stack.pop()
        stack.extend(n.children.values())
        yield n


def locked(cache, running):
    """Every lock in the tree is one running request's, on the node its `lock` says, which has a snapshot."""
    holders = [r for r in running if r.lock is not None]
    assert sorted(map(id, (lock.owner for lock in cache.locks()))) == sorted(map(id, holders)), (cache.locks(), len(holders))
    for r in holders:
        assert r.lock.owner is r and r.lock.node.state_lock_ref >= 1 and r.lock.node.snapshot is not None
    return len(holders)


def depth(req):
    return req.lock.node.depth if req.lock is not None else None


class RecordingExecutor(Executor):
    def admit(self, prompt_ids, params, req=None):
        req = super().admit(prompt_ids, params, req)
        if not any(r is req for r in self.admitted):  # a retracted request is admitted again
            self.admitted.append(req)
        turn = self.turns.setdefault(req.request_id, Turn(req))
        turn.resume = req.prefix_len
        turn.state_exact = self.model.state(req.slot) == recurrence(req.prefill_ids[:req.prefix_len])
        turn.admitted_at = self.cache.taken
        return req

    def _finish_prefill(self, batch, out):
        first = [r for r, _ in batch.chunks if r.prefix_len and r.prefill_pos == r.prefix_len]
        token = super()._finish_prefill(batch, out)
        for req in first:
            self.turns[req.request_id].lock_after = depth(req)
        live = [r for r in self.live_requests() if not r.released]
        locked(self.cache, live)
        self.state.check()
        for req, _ in batch.chunks:
            if (turn := self.turns.get(req.request_id)) is not None:
                turn.locks.append(self.cache.locks())
        return token

    def live_requests(self):
        return [r for r in self.admitted if r.slot >= 0]


class RecordingMtp(MtpProposer):
    def ingest_prefill(self, req, new_ids, start, result):
        turn = self.turns.get(req.request_id)
        first = start and (req.proposer_state is None or req.proposer_state.last_hidden is None)
        if turn is not None and first:
            node = req.match.node
            turn.snapshot = node.snapshot is not None
            turn.row = has_row(node, self.cache_identity, start)
            turn.locked_at_ingest = req.lock is not None and req.lock.node is node and node.state_lock_ref >= 1
            turn.window = self.radix.taken - turn.admitted_at
        hits, misses = self.cache_hits, self.cache_fallbacks
        self.head.writer = req
        try:
            super().ingest_prefill(req, new_ids, start, result)
        finally:
            self.head.writer = None
        if turn is not None and first:
            turn.drafter = ("MISS" if self.cache_fallbacks > misses
                            else "hit" if self.cache_hits > hits else turn.drafter)

    def _accepted(self, req, *args, **kwargs):
        self.head.writer = req
        try:
            return super()._accepted(req, *args, **kwargs)
        finally:
            self.head.writer = None

    def propose(self, requests, *args, **kwargs):
        self.head.round_start = [r.context_len - 1 for r in requests]
        return super().propose(requests, *args, **kwargs)

    def start(self, req, token):
        super().start(req, token)
        turn = self.turns.get(req.request_id)
        if turn is not None and turn.ready is None:
            turn.ready = self.ready(req)


def harness(slots, max_batch, pages=200_000, k=3):
    """The executor as Executor.__init__ wires it, around the stubs, with the real Scheduler on top."""
    head = Head(pages)
    ex = RecordingExecutor.__new__(RecordingExecutor)
    ex.target, ex.device, ex.max_context, ex.max_batch = FLASHNEXT, CPU, 65_536, max_batch
    ex.kv_pool = HostKvPool(pages, head.freed)
    ex.state_pool = StatePool(slots, CPU, target=STATE)
    ex.state = StateManager(ex.kv_pool, ex.state_pool, CountingCache(ex.kv_pool, ex.state_pool, CPU))
    ex.sampler = Sampler(CPU)
    ex.spec, ex.graphs, ex.use_graphs = None, None, False
    ex._paged_prefill = ex._decode_wrapper = SimpleNamespace(plan=lambda *a, **kw: None)
    ex.model = Target(ex.state_pool)
    proposer = RecordingMtp(head, ProposerContext(CPU, max_batch, ex.max_context), k=k)
    ex.set_proposer(proposer)
    ex.turns = proposer.turns = {}
    ex.admitted = []  # every request admitted, in order (direct admits share request id 0)
    proposer.radix = ex.cache
    sched = Scheduler(ex, SchedulerConfig(max_running=max_batch, chunk_size=4096, prefill_budget=4096,
                                          solo_chunk_size=8192))
    return SimpleNamespace(ex=ex, proposer=proposer, head=head, sched=sched, ident=proposer.cache_identity, unrelated=0)


def draw(rng, n):
    return [rng.randrange(TOKEN + 1, 50_000) for _ in range(n)]


def conversation(seed, header=600, message=50, turns=8, reply=70, tools=(300, 150, 500, 200)):
    """Agent-replay-shaped prompts: each turn's is the previous one plus a recorded reply and a tool result, so it
    departs from the server's own output (all TOKEN) at the previous prompt's end."""
    rng = random.Random(seed)
    prompts = [draw(rng, header + message)]
    for t in range(1, turns):
        prompts.append(prompts[-1] + draw(rng, reply) + draw(rng, tools[(t - 1) % len(tools)]))
    return prompts


def run_turn(h, prompt, out=72, others=0, boundary=0):
    """One agent turn, with `others` unrelated 600-token prompts submitted alongside (each its own first token).
    `boundary`: the start of the prompt's last message, as the chat API reports it (0: a raw prompt)."""
    params = SamplingParams(max_new_tokens=out, ignore_eos=True)
    req = h.sched.submit(prompt, SamplingParams(max_new_tokens=out, ignore_eos=True, message_boundary=boundary))
    h.ex.turns[req.request_id] = turn = Turn(req)
    for _ in range(others):
        h.unrelated += 1
        body = [TOKEN + 1 + (h.unrelated * 7919 + i) % 40_000 for i in range(599)]
        h.sched.submit([50_000 + h.unrelated] + body, params)
    before = h.ex.cache.taken
    h.sched.run_until_idle()
    turn.evicted = h.ex.cache.taken - before
    assert req.output_ids == [TOKEN] * out
    return turn


def table(h, turns):
    yes = {None: "-", True: "yes", False: "NO"}
    rows = [f"sharar: {BUILD}", "turn  prompt resume evicted window snapshot@ingest row@ingest locked@ingest lock-after "
                                "drafter drafted rounds"]
    dash = lambda v: "-" if v is None else v  # noqa: E731
    rows += [f"{i:4d} {t.request.prefill_len:7d} {t.resume:6d} {t.evicted:7d} {dash(t.window):>6} {yes[t.snapshot]:>15} "
             f"{yes[t.row]:>10} {yes[t.locked_at_ingest]:>11} {dash(t.lock_after):>10} {t.drafter:7} "
             f"{'yes' if t.drafted else 'NO':7} {t.request.spec_rounds:6d}" for i, t in enumerate(turns)]
    rows.append(f"drafter hits {h.proposer.cache_hits}, misses {h.proposer.cache_fallbacks}; snapshots taken "
                f"{h.ex.cache.taken} ({h.ex.cache.snapshot_evictions} interior); draft KV: inexact rows {h.head.inexact}, steps reading unwritten "
                f"rows {h.head.unwritten_reads}, inexact {h.head.inexact_reads}")
    return "\n".join(rows)


@pytest.mark.parametrize("slots, max_batch, others", [
    (12, 8, 0),  # a wide batch (--throughput: 128) under the slot floor: prefill destinations evict
    (8, 4, 0),  # four requests (--fast): a full pool
    (16, 4, 2),  # unrelated prompts prefilled with each turn: their checkpoints evict
], ids=["wide-batch", "full-pool", "concurrent"])
def test_every_resumed_turn_drafts_from_its_matched_node_under_slot_pressure(slots, max_batch, others):
    """One agent's turns, each resuming from the previous prompt's end, while prefill checkpoints (the turn's own, or
    those of prompts prefilled with it) evict snapshots. The matched node keeps its snapshot and drafter row while the
    turn's lock is there, until its first prefill chunk has been ingested: every resume is a drafter hit."""
    h = harness(slots, max_batch)
    turns = [run_turn(h, prompt, others=others) for prompt in conversation("race")]
    resumed = turns[1:]
    assert all(t.resume for t in resumed), "every turn after the first resumes from the cache\n" + table(h, turns)
    assert sum(t.window for t in resumed), (
        f"precondition failed: no snapshot was evicted between a resumed turn's admission and its first prefill "
        f"ingest ({slots} slots, max batch {max_batch}, {others} unrelated prompts a turn), so this case no longer "
        "puts the cache under slot pressure where the matched node is exposed, and cannot show that the node survives "
        "it (without the lock there, that eviction strips it). Retune the slot counts until resumed turns "
        "evict snapshots in that window again (column 'window'; 'evicted' counts the whole turn).\n" + table(h, turns))
    assert [i for i, t in enumerate(turns) if t.resume and t.drafter != "hit"] == [], \
        "resumes that missed the drafter's row\n" + table(h, turns)
    assert [i for i, t in enumerate(resumed, 1) if not t.kept] == [], \
        "resumes whose matched node lost its snapshot or drafter row before the first prefill ingest\n" \
        + table(h, turns)
    assert [i for i, t in enumerate(turns) if not t.drafted] == [], "turns decoded without drafts\n" + table(h, turns)
    assert (h.proposer.cache_hits, h.proposer.cache_fallbacks) == (len(resumed), 0), table(h, turns)
    assert (h.head.inexact, h.head.unwritten_reads, h.head.inexact_reads) == ([], 0, 0), table(h, turns)
    assert all(t.state_exact for t in turns), table(h, turns)
    # The lock was at the matched node through the first ingest and moved on with that chunk (each turn adds more than
    # 64 tokens, so its first chunk places a snapshot past the match); none is left once every request is done.
    assert [i for i, t in enumerate(resumed, 1) if not t.locked_at_ingest] == [], table(h, turns)
    assert [i for i, t in enumerate(resumed, 1) if not t.lock_after or t.lock_after <= t.resume] == [], \
        "locks that stayed at the matched node after a first chunk that placed a newer snapshot\n" + table(h, turns)
    assert not h.ex.state.requests and not h.sched.running, table(h, turns)
    assert not h.ex.cache.locks() and all(n.lock_ref == 0 for n in nodes(h.ex.cache)), table(h, turns)


def test_the_matched_node_keeps_its_snapshot_and_row_until_the_first_prefill_chunk():
    """Admit a request resuming at a cached prompt's end, then evict every snapshot slot pressure may take before its
    prefill: the matched node, where the request's lock is, keeps its snapshot and MTP row, the prefill is a drafter
    hit, and once the lock moved on (the checkpoint after that chunk's ingest) the snapshot is evictable like any other
    on a running request's path. Requests released or retracted before their prefill end their locks."""
    h = harness(slots=16, max_batch=4)
    first = run_turn(h, conversation("lifecycle", turns=1)[0], boundary=600).request  # its last message at 600
    seq = first.prompt_ids + first.output_ids[:-1]  # cached: its message boundary (576), last chunk end and reply end
    rng, params = random.Random("lifecycle"), SamplingParams(max_new_tokens=8, ignore_eos=True)
    # A request continuing the whole reply, admitted and not prefilled: it locks the path past the prompt's end, so
    # the snapshots before it stay interior nodes.
    below = h.ex.admit(seq + draw(rng, 40), params)
    req = h.ex.admit(first.prompt_ids + draw(rng, 300), params)
    node, end = req.match.node, below.match.node
    assert req.prefix_len == 640 and node.children and end.depth == len(seq)  # the first prompt's last chunk end
    assert node.snapshot is not None and has_row(node, h.ident, req.prefix_len)
    assert (req.lock.node, below.lock.node) == (node, end) and (node.state_lock_ref, end.state_lock_ref) == (1, 1)
    assert locked(h.ex.cache, [req, below]) == 2
    evictions = h.ex.cache.taken
    h.ex.cache.evict(need_slots=h.ex.state_pool.free_count + 1024)
    assert h.ex.cache.taken > evictions, "precondition: the eviction took the path's other snapshots"
    assert node.snapshot is not None and has_row(node, h.ident, req.prefix_len), (
        f"eviction between admission and the first prefill chunk stripped the matched node (snapshot "
        f"{node.snapshot}, locked {node.state_lock_ref})")
    hits, misses = h.proposer.cache_hits, h.proposer.cache_fallbacks
    h.ex.prefill(req)
    assert (h.proposer.cache_hits - hits, h.proposer.cache_fallbacks - misses) == (1, 0), "drafter miss at the resume"
    assert h.ex.turns[req.request_id].kept and h.proposer.ready(req)
    last = req.prefix_len + (req.prefill_len - 1 - req.prefix_len) // 64 * 64  # the forward's last grid point < n
    assert depth(req) == last and node.state_lock_ref == 0, "the lock moved to the request's last chunk end"
    assert not req.private and locked(h.ex.cache, [req, below]) == 2
    h.ex.cache.evict(need_slots=h.ex.state_pool.free_count + 1)
    assert node.snapshot is None, "the snapshot is evictable once the lock moved on"
    assert req.lock.node.snapshot is not None, "the request's last chunk end is not"

    # Released or retracted before their prefill: the locks end.
    again = h.ex.admit(seq + draw(rng, 30), params)
    assert again.match.node is end and end.state_lock_ref == 2
    h.ex.retract(again)
    assert again.lock is None and end.state_lock_ref == 1
    h.ex.release(below)
    assert end.state_lock_ref == 0
    h.ex.release(req)
    assert not h.ex.state.requests
    assert not h.ex.cache.locks() and [(n.depth, n.lock_ref) for n in nodes(h.ex.cache) if n.lock_ref] == []


def test_an_admission_failing_after_its_match_ends_its_lock(monkeypatch):
    """The KV allocation fails after admission matched the node, locked it and copied its snapshot: the lock and the
    slot come back, and the retry resumes there and drafts."""
    h = harness(slots=16, max_batch=4)
    first = run_turn(h, conversation("failure", turns=1)[0]).request
    prompt = first.prompt_ids + draw(random.Random("failure"), 300)
    node = h.ex.cache.match(prompt[:-1], touch=False).node
    assert node is not None and node.depth == 640  # the first prompt's last chunk end
    slots, pages = h.ex.state_pool.free_count, h.ex.kv_pool.free_count
    seen = []  # (state_lock_ref, lock_ref) at the matched node, and every lock in the tree, when the allocation fails

    def fail(n):
        seen.append((node.state_lock_ref, node.lock_ref, len(h.ex.cache.locks())))
        raise MemoryError("injected allocation failure")
    monkeypatch.setattr(h.ex.kv_pool, "alloc_with_ids", fail)
    req = Request(prompt, SamplingParams(max_new_tokens=8, ignore_eos=True))
    with pytest.raises(MemoryError):
        h.ex.admit(req.prompt_ids, req.params, req)
    assert seen == [(1, 1, 1)], f"precondition: the allocation fails while admission locks the node: {seen}"
    assert req.slot == -1 and req.pages is None and req.match is None and req.lock is None
    assert (node.state_lock_ref, node.lock_ref, len(h.ex.cache.locks())) == (0, 0, 0)
    assert (h.ex.state_pool.free_count, h.ex.kv_pool.free_count) == (slots, pages)
    monkeypatch.undo()
    h.ex.admit(req.prompt_ids, req.params, req)  # the retry resumes there and drafts
    assert req.lock.node is node and len(h.ex.cache.locks()) == 1
    h.ex.prefill(req)
    assert (h.proposer.cache_hits, h.proposer.cache_fallbacks, node.state_lock_ref) == (1, 0, 0)
    h.ex.release(req)
    assert not h.ex.cache.locks() and [(n.depth, n.lock_ref) for n in nodes(h.ex.cache) if n.lock_ref] == []
