"""Speculator parts (#45): declared capabilities, shared draft heads, the depth policy."""
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.contracts import Capabilities, ProposerRequirements, check_capabilities
from sharar.engine.depth import FixedDepth
from sharar.proposers.draft_head import ShortlistHead, TargetHead


def test_declared_capabilities_need_their_methods():
    class Bare:
        requirements = ProposerRequirements("bare", capabilities=Capabilities(reloadable=True))
    with pytest.raises(TypeError, match="reloadable"):
        check_capabilities(Bare())

    class Reload(Bare):
        def reload(self, snapshot):
            return str(snapshot)
    check_capabilities(Reload())

    class Head:
        requirements = ProposerRequirements("mtp", capabilities=Capabilities(draft_head=True))
        head = SimpleNamespace()
    with pytest.raises(TypeError, match="draft_head"):
        check_capabilities(Head())


def test_the_served_proposers_declare_what_they_implement():
    from sharar.engine.contracts import DeviceFeedback, Reloadable, TraceFeatures
    from sharar.proposers.parallel import ParallelProposer
    from sharar.proposers.ngram import NGramProposer
    for protocol in (DeviceFeedback, Reloadable, TraceFeatures):
        assert all(callable(getattr(ParallelProposer, name, None)) for name in protocol.__dict__
                   if not name.startswith("_") and callable(getattr(protocol, name, None)))
    assert NGramProposer.requirements.capabilities == Capabilities()


def test_target_head_is_the_logits_function():
    logits = torch.randn(5, 40)
    head = TargetHead(lambda x: logits)
    assert head.bytes == 0 and torch.equal(head.argmax(None), logits.argmax(-1))
    values, ids = head.topk(None, 4)
    ref = logits.topk(4, dim=-1)
    assert torch.equal(values, ref.values) and torch.equal(ids, ref.indices)


def test_shortlist_topk_maps_rows_to_global_ids():
    from sharar.model.flashnext.linear import TrunkLinear
    torch.manual_seed(3)
    weight = torch.randn(20, 8).to(torch.bfloat16)
    x = torch.randn(2, 3, 8).to(torch.bfloat16)
    ids = torch.tensor([17, 2, 5, 11, 0])
    head = ShortlistHead(TrunkLinear(weight), ids)
    values, got = head.topk(x, 2)
    rows = ids.sort().values
    ref = (x.reshape(-1, 8) @ weight[rows].T).topk(2, dim=-1)
    assert got.shape == (2, 3, 2) and torch.equal(got.reshape(-1, 2), rows[ref.indices])
    assert torch.equal(values.reshape(-1, 2), ref.values)


def test_fixed_depth_and_the_verify_depth_setter():
    assert FixedDepth(4).choose([object()]) == 4
    with pytest.raises(ValueError):
        FixedDepth(0)
    from sharar.engine.spec import SpecDecoder
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.k = 7
    assert spec.verify_depth == 7  # no policy yet: the proposer's k
    spec.verify_depth = 3
    assert isinstance(spec.depth_policy, FixedDepth) and spec.verify_depth == 3
