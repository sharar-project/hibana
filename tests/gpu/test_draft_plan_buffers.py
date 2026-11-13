"""MTP draft graphs: each step of a round plans into its own pinned buffer.

plan() writes the schedule into the wrapper's pinned buffer and queues its copy behind the previous step's replay; the
host plans every step before the GPU reaches those copies, so steps sharing one buffer could read a later step's
schedule. Each step index owns a buffer, reused only by the next round's same step."""
from types import SimpleNamespace

import torch

from hibana.proposers.mtp_graphs import DraftGraphs


def _graphs(seen):
    g = object.__new__(DraftGraphs)
    g._pinned, g.max_context, g.device = [], 1 << 20, torch.device("cuda")

    def paged_prefill(wrapper, qo_indptr, kv_indptr, pages, *args, **kwargs):
        seen.append((wrapper._pin_memory_int_workspace_buffer.data_ptr(), pages.tolist()))
    g.mtp = SimpleNamespace(attention=SimpleNamespace(paged_prefill=paged_prefill))
    return g


def test_each_step_of_a_round_plans_into_its_own_pinned_buffer():
    seen = []
    g = _graphs(seen)
    own = torch.empty(64, dtype=torch.uint8, pin_memory=True)
    st = {"wrapper": SimpleNamespace(_pin_memory_int_workspace_buffer=own)}
    kv = [torch.arange(5, dtype=torch.int32), torch.arange(7, dtype=torch.int32)]
    for _ in range(2):  # two rounds of three steps
        for step in range(3):
            g._plan(st, [p[:p.shape[0] - 2 + step] for p in kv], step)
    ptrs = [p for p, _ in seen]
    assert ptrs[0] == own.data_ptr()  # step 0 keeps the wrapper's own buffer
    assert len(set(ptrs[:3])) == 3  # distinct buffers within a round
    assert ptrs[3:] == ptrs[:3]  # the next round's same step reuses its buffer
    assert all(b.is_pinned() and b.numel() == own.numel() for b in g._pinned)
    assert seen[1][1] == list(range(4)) + list(range(6))  # the batch's pages, concatenated


def test_one_request_plans_its_pages_without_a_concatenation():
    seen = []
    g = _graphs(seen)
    st = {"wrapper": SimpleNamespace(_pin_memory_int_workspace_buffer=torch.empty(64, dtype=torch.uint8, pin_memory=True))}
    g._plan(st, [torch.arange(9, dtype=torch.int32)], 0)
    assert seen[0][1] == list(range(9))
