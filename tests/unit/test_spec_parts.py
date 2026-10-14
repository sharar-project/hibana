"""Speculator parts (#45): declared capabilities, shared draft heads, the depth policy."""
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.contracts import Capabilities, ProposerRequirements, check_capabilities
from sharar.engine.depth import FixedDepth
from sharar.proposers.draft_head import ShortlistHead, TargetHead


def test_declared_capabilities_need_their_methods():
    class Bare:
        requirements = ProposerRequirements("bare", capabilities=Capabilities(stored_payload=True))
    with pytest.raises(TypeError, match="stored_payload"):
        check_capabilities(Bare())

    class Cold(Bare):
        def auxiliary_codec(self):
            return None
    check_capabilities(Cold())

    class Head:
        requirements = ProposerRequirements("mtp", capabilities=Capabilities(draft_head=True))
        head = SimpleNamespace()
    with pytest.raises(TypeError, match="draft_head"):
        check_capabilities(Head())


def test_the_served_proposers_declare_what_they_implement():
    from sharar.engine.contracts import CheckpointCache, StoredPayload, DeviceFeedback
    from sharar.proposers.parallel import ParallelProposer
    from tests.ngram_proposer import NGramProposer
    for protocol in (DeviceFeedback, CheckpointCache, StoredPayload):
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


def test_batch_depth_by_bucket():
    from sharar.engine.depth import BatchDepth, parse_depths
    sizes = (1, 2, 4, 8, 16)
    policy = parse_depths("8:1, 16:0", sizes, 4)
    assert [policy.choose([None] * n) for n in (1, 2, 3, 4, 5, 8, 9, 16)] == [4, 4, 4, 4, 1, 1, 0, 0]
    assert policy.verify_depths == (1, 4) and policy.depth == 4
    assert policy.sizes_for(4, sizes) == (1, 2, 4) and policy.sizes_for(1, sizes) == (8,)
    assert parse_depths(None, sizes, 4) is None and parse_depths("", sizes, 4) is None
    assert parse_depths("16:4", sizes, 4) is None  # every bucket keeps k: the fixed policy
    for bad in ("16", "16:x", "3:1", "16:5", "16:-1"):
        with pytest.raises(ValueError):
            parse_depths(bad, sizes, 4)
    with pytest.raises(ValueError):
        BatchDepth({4: 0}, 4).choose([None] * 5)  # beyond every bucket
    assert FixedDepth(3).sizes_for(3, sizes) == sizes


def test_profile_buckets_up_to_128():
    from sharar.engine.profile import ServingProfile
    assert ServingProfile().graph_sizes == (1, 2, 4)  # the default profile is unchanged
    assert ServingProfile(max_batch=16).graph_sizes == (1, 2, 4, 8, 16)
    assert ServingProfile(max_batch=12).graph_sizes == (1, 2, 4, 8, 12)
    assert ServingProfile(max_batch=128).graph_sizes == (1, 2, 4, 8, 16, 32, 64, 128)
    assert ServingProfile(max_batch=48).graph_sizes == (1, 2, 4, 8, 16, 32, 48)
    with pytest.raises(ValueError):
        ServingProfile(max_batch=129)



def test_acceptance_depth_follows_each_requests_rate_under_the_cap():
    from types import SimpleNamespace

    from sharar.engine.depth import AcceptanceDepth, BatchDepth
    sizes = (1, 2, 4, 8)
    policy = AcceptanceDepth(FixedDepth(8), step_cost=0.14)
    assert policy.verify_depths == tuple(range(1, 9)) and policy.sizes_for(8, sizes) == sizes and policy.sizes_for(9, sizes) == ()
    easy, hard, fresh = (SimpleNamespace(spec_rate=None) for _ in range(3))
    for _ in range(40):  # code-like: every draft accepted; prose-like: one in two
        policy.observe(easy, 8, 8)
        policy.observe(hard, 8, 1)
    assert policy.choose([easy]) == 8 and policy.choose([hard]) <= 2 and 2 < policy.choose([fresh]) < 8
    assert policy.choose([hard, easy]) == 8  # a batch verifies the deepest depth any request wants
    capped = AcceptanceDepth(BatchDepth({1: 8, 2: 4, 4: 4, 8: 0}, 8), step_cost=0.14)
    assert capped.choose([easy]) == 8 and capped.choose([easy, easy]) == 4 and capped.choose([easy] * 5) == 0
    assert capped.sizes_for(5, sizes) == (1,) and capped.sizes_for(3, sizes) == (1, 2, 4)
    # above four requests a batch verifies its bucket's depth, with that depth's graphs only (throughput profile memory)
    wide = AcceptanceDepth(BatchDepth({1: 8, 2: 4, 4: 4, 8: 4, 16: 3, 32: 3}, 8), step_cost=0.14)
    assert wide.choose([hard] * 8) == 4 and wide.choose([hard] * 20) == 3 and wide.choose([hard]) <= 2
    assert wide.verify_depths == tuple(range(1, 9)) and wide.sizes_for(3, (1, 2, 4, 8, 16, 32)) == (1, 2, 4, 16, 32)
    assert wide.sizes_for(2, (1, 2, 4, 8, 16, 32)) == (1, 2, 4) and wide.sizes_for(4, (1, 2, 4, 8, 16, 32)) == (1, 2, 4, 8)
    with pytest.raises(ValueError):
        AcceptanceDepth(FixedDepth(4), 0.0)
