"""Reduced draft vocabulary: ranking and the draft-head index mapping."""
import json
from types import SimpleNamespace

import torch

from sharar.proposers.draft_vocab import draft_vocab
from sharar.proposers.mtp import draft_argmax


def test_specials_first_then_construction_order(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 90}, {"id": 95}, {"id": 200}]}))
    ids = draft_vocab(tmp_path, 5, 100)
    assert ids.tolist() == [0, 1, 2, 90, 95]  # specials inside the vocabulary always kept, then the lowest ids


def test_reduced_head_maps_rows_to_global_ids():
    from sharar.model.flashnext.linear import TrunkLinear
    from sharar.proposers.draft_vocab import ReducedHead
    torch.manual_seed(0)
    weight = torch.randn(10, 4).to(torch.bfloat16)
    x = torch.randn(3, 4).to(torch.bfloat16)
    reduced = ReducedHead(TrunkLinear(weight), torch.tensor([7, 2, 5])).argmax(x)
    assert torch.equal(reduced, torch.tensor([2, 5, 7])[(x @ weight[[2, 5, 7]].T).argmax(-1)])


def test_draft_argmax_prefers_the_heads_reduced_pick():
    full = SimpleNamespace(logits=lambda h: torch.tensor([[0., 3., 1.]]))
    assert draft_argmax(full, None).tolist() == [1]
    reduced = SimpleNamespace(logits=None, draft_argmax=lambda h: torch.tensor([42]))
    assert draft_argmax(reduced, None).tolist() == [42]


def test_pad_rows_repeats_the_last_id_to_the_tile():
    from sharar.proposers.draft_vocab import _pad_rows
    ids = torch.arange(130)
    padded = _pad_rows(ids, 128)
    assert padded.numel() == 256 and torch.equal(padded[:130], ids) and (padded[130:] == 129).all()
    assert _pad_rows(torch.arange(128), 128).numel() == 128


def test_full_fp8_draft_head_is_the_targets_own():
    from sharar.model.flashnext.linear import TrunkLinear
    from sharar.proposers.draft_vocab import ReducedHead
    lm_head = TrunkLinear(torch.randn(10, 4).to(torch.bfloat16))
    head = ReducedHead(lm_head)
    assert head.head is lm_head and head.bytes == 0
    x = torch.randn(3, 4).to(torch.bfloat16)
    assert torch.equal(head.argmax(x), (x @ lm_head.weight.T).argmax(-1))
