"""Production CuTe routing, ragged TP geometry, checkpoints and continuation."""
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from hibana.engine.config import current, using
from hibana.engine.context import ForwardContext
from hibana.model.gdn_plan import prepare_gdn_prefill
from hibana.kernels.gdn_prefill import prefill, backend


def plan_for(lengths, slots, checkpoints=()):
    t = sum(lengths)
    ctx = ForwardContext(
        mode="prefill", kv_pool=None, state_pool=SimpleNamespace(),
        positions=torch.arange(t, device="cuda"), seq_lens=list(lengths),
        seq_prefix_lens=[64] * len(lengths),
        cu_seqlens=torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="cuda"),
        seq_slots=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        checkpoint_positions=[64 + local for _, local, _ in checkpoints],
        checkpoint_slots=[slot for _, _, slot in checkpoints],
        checkpoint_seqs=[seq for seq, _, _ in checkpoints],
    )
    return prepare_gdn_prefill(ctx, t, "cuda")


def inputs(t, hk, hv):
    # Serving splits the convolution output; token strides span all Q/K/V heads.
    mixed = torch.randn(1, t, 2 * hk + hv, 128, dtype=torch.bfloat16, device="cuda")
    q, k, v = mixed.split([hk, hk, hv], dim=2)
    v.mul_(.1)
    g = -torch.rand(1, t, hv, device="cuda") * .1
    beta = torch.rand(1, t, hv, dtype=torch.bfloat16, device="cuda")
    return dict(q=q, k=k, v=v, g=g, beta=beta)


@pytest.mark.parametrize("hk,hv", [(2, 6), (16, 48), (8, 24)])
@torch.inference_mode()
def test_ragged_checkpoints_and_untouched_slots(hk, hv):
    torch.manual_seed(913)
    lengths = [129, 193]
    args = inputs(sum(lengths), hk, hv)
    initial = torch.randn(9, hv, 128, 128, device="cuda") * .001
    pool = initial.clone()
    # Reordered destinations; one sequence has two snapshots.
    plan = plan_for(lengths, [3, 1], [(1, 128, 7), (0, 64, 5), (1, 64, 6)])
    with using(current().replace(gdn_prefill="cute")):
        out = prefill(**args, state=pool, plan=plan)
        for seq, local, destination in [(1, 128, 7), (0, 64, 5), (1, 64, 6)]:
            start = sum(lengths[:seq])
            short = {key: value[:, start:start + local].contiguous() for key, value in args.items()}
            reference = initial.clone()
            expected = prefill(**short, state=reference, plan=plan_for([local], [[3, 1][seq]]))
            assert torch.equal(pool[destination], reference[[3, 1][seq]])
            assert torch.equal(out[:, start:start + local], expected)
        for seq, slot in enumerate([3, 1]):
            start = sum(lengths[:seq])
            single = {key: value[:, start:start + lengths[seq]].contiguous() for key, value in args.items()}
            reference = initial.clone()
            expected = prefill(**single, state=reference, plan=plan_for([lengths[seq]], [slot]))
            assert torch.equal(out[:, start:start + lengths[seq]], expected)
            assert torch.equal(pool[slot], reference[slot])
    for slot in [0, 2, 4, 8]:
        assert torch.equal(pool[slot], initial[slot])
    assert torch.isfinite(out).all() and torch.isfinite(pool).all()


@torch.inference_mode()
def test_checkpoint_continuation_matches_full_sequence():
    torch.manual_seed(22)
    args = inputs(193, 2, 6)
    initial = torch.randn(5, 6, 128, 128, device="cuda") * .001
    pool = initial.clone()
    with using(current().replace(gdn_prefill="cute")):
        full = prefill(**args, state=pool, plan=plan_for([193], [1], [(0, 64, 3)]))
        resumed = initial.clone()
        resumed[2] = pool[3]
        tail = {key: value[:, 64:].contiguous() for key, value in args.items()}
        actual = prefill(**tail, state=resumed, plan=plan_for([129], [2]))
    assert torch.equal(full[:, 64:], actual)
    assert torch.equal(pool[1], resumed[2])


@torch.inference_mode()
def test_cute_against_fp32_recurrence():
    from hibana.kernels.fla.l2norm import l2norm_fwd
    torch.manual_seed(77)
    lengths, hk, hv = [1, 31, 32, 33, 65], 2, 6
    args = inputs(sum(lengths), hk, hv)
    initial = torch.randn(7, hv, 128, 128, device="cuda") * .001
    pool = initial.clone()
    slots = [5, 1, 4, 2, 3]
    with using(current().replace(gdn_prefill="cute")):
        actual = prefill(**args, state=pool, plan=plan_for(lengths, slots))
    q = l2norm_fwd(args["q"].contiguous())[0].float().repeat_interleave(hv // hk, dim=1)
    k = l2norm_fwd(args["k"].contiguous())[0].float().repeat_interleave(hv // hk, dim=1)
    reference = torch.empty_like(actual, dtype=torch.float32)
    start = 0
    for length, slot in zip(lengths, slots):
        h = initial[slot].clone()
        for t in range(start, start + length):
            h *= args["g"][0, t].exp()[:, None, None]
            delta = (args["v"][0, t].float() - (h * k[t, :, None, :]).sum(-1)) * args["beta"][0, t, :, None].float()
            h += delta[:, :, None] * k[t, :, None, :]
            reference[0, t] = (h * q[t, :, None, :]).sum(-1) * 128 ** -.5
        torch.testing.assert_close(pool[slot], h, atol=.001, rtol=.04)
        start += length
    torch.testing.assert_close(actual.float(), reference, atol=.0002, rtol=.04)


@torch.inference_mode()
def test_prepared_graph_changed_inputs():
    torch.manual_seed(39)
    args = inputs(65, 2, 6)
    initial = torch.randn(4, 6, 128, 128, device="cuda") * .001
    pool = initial.clone()
    plan = plan_for([65], [1], [(0, 64, 3)])
    with using(current().replace(gdn_prefill="cute")):
        for _ in range(2):
            pool.copy_(initial)
            prefill(**args, state=pool, plan=plan)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = prefill(**args, state=pool, plan=plan)
        args["v"].mul_(.3)
        pool.copy_(initial)
        graph.replay()
        actual, state = output.clone(), pool.clone()
        pool.copy_(initial)
        expected = prefill(**args, state=pool, plan=plan)
        assert torch.equal(actual, expected)
        assert torch.equal(state, pool)


def test_auto_and_stable_backend_policy():
    args = inputs(1, 2, 6)
    state = torch.empty(1, 6, 128, 128, device="cuda")
    with using(current().replace(gdn_prefill="auto", stable_numerics=False)):
        assert backend(args["q"], args["k"], args["v"], state) == "cute"
    with using(current().replace(gdn_prefill="auto", stable_numerics=True)):
        assert backend(args["q"], args["k"], args["v"], state) == "triton"


@pytest.mark.parametrize("length", [16384, 24576])
@torch.inference_mode()
def test_long_sparse_checkpoint_continuation(length):
    torch.manual_seed(length)
    args = inputs(length, 8, 24)
    initial = torch.randn(5, 24, 128, 128, device="cuda") * .001
    pool = initial.clone()
    plan = plan_for([length], [1], [(0, 8192, 3), (0, length - 64, 4)])
    # A64-token lookup cadence must allocate only two state snapshots.
    assert plan.checkpoint_interval == 64 and plan.checkpoint_count == 2
    with using(current().replace(gdn_prefill="cute")):
        full = prefill(**args, state=pool, plan=plan)
        resumed = initial.clone()
        resumed[2] = pool[3]
        tail = {key: value[:, 8192:].contiguous() for key, value in args.items()}
        actual = prefill(**tail, state=resumed, plan=plan_for([length - 8192], [2]))
    assert torch.equal(full[:, 8192:], actual)
    assert torch.equal(pool[1], resumed[2])
