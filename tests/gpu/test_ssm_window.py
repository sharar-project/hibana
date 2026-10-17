"""The per-window quantized recurrent state (kernels/ssm_window.py) against FP64 references of the delta rule."""
import pytest
import torch

from sharar.cache.ssm_format import WindowLayout, WindowPool
from sharar.kernels import ssm_window as W

H, HV, K, V = 4, 12, 128, 128  # the 27B's head shapes, a quarter of its heads


RANK = {8: 4, 4: 8}  # the paper's compensator counts
# Relative error of one boundary quantization of `_state`, and of the commits that fold one.
QUANT = {8: 2e-3, 4: 3e-2}
FOLD = {8: 5e-3, 4: 6e-2}  # random full-rank records: near plain per-row noise (2.9 / qmax / sqrt(12) of a row)


def _pool(window=16, rank=None, layers=2, slots=6, bits=8):
    lay = WindowLayout(H, HV, K, V, window, RANK[bits] if rank is None else rank, bits)
    raw = torch.zeros((layers, slots, lay.nbytes), dtype=torch.uint8, device="cuda")
    return lay, raw, WindowPool(lay, raw, list(range(layers)))


def _state(n, seed=0):
    """Recurrent-state-like matrices: a few dominant rank-one terms, outlier key columns, small noise."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    st = torch.randn((n, HV, V, K), device="cuda", generator=g) * 0.02
    for _ in range(3):
        st += (torch.randn((n, HV, V, 1), device="cuda", generator=g) * torch.randn((n, HV, 1, K), device="cuda", generator=g)) * 0.5
    st[..., 5] *= 30
    st[..., 77] *= 12
    return st


def _inputs(tokens, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, device="cuda", generator=g).to(torch.bfloat16)
    return dict(q=rand(1, tokens, H, K), k=rand(1, tokens, H, K), v=rand(1, tokens, HV, V), a=rand(tokens, HV),
                b=rand(tokens, HV), A_log=torch.randn(HV, device="cuda", generator=g) * 0.5 - 1, dt_bias=rand(HV))


def _ring(rows, steps, layers=None):
    lead = () if layers is None else (layers,)
    return (torch.zeros((*lead, rows, H, steps, K), dtype=torch.float16, device="cuda"),
            torch.zeros((*lead, rows, HV, steps, V), dtype=torch.float16, device="cuda"),
            torch.zeros((*lead, rows, HV, steps), dtype=torch.float32, device="cuda"))


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.mark.parametrize("bits", [8, 4])
def test_quantize_keeps_outliers_out_of_the_integer_range(bits):
    lay, _, pool = _pool(bits=bits)
    st = _state(2)
    slots = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
    W.quantize(lay, pool.layer[0], st, slots)
    for i, s in enumerate((1, 4)):
        assert _rel(W.reference_state(lay, pool.layer[0], s), st[i]) < QUANT[bits]
        assert int(pool.layer[0]["log_len"][s].abs().sum()) == 0
    if bits == 4:  # two values per byte, each in -7..7
        values = W.reference_dequant(lay, pool.layer[0]["z"][1])
        assert values.abs().max() == 7 and values.shape == (HV, V, K)
    x = st[0].double()  # plain per-row quantization of the same states
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(-1, keepdim=True)
    assert _rel(torch.round(x / amax * qmax) * amax / qmax, x) > 5 * _rel(W.reference_state(lay, pool.layer[0], 1), st[0])
    # Without compensators and smoothing the residual keeps the outliers.
    lay0, _, pool0 = _pool(rank=0, bits=bits)
    W.quantize(lay0, pool0.layer[0], st, slots)
    assert _rel(W.reference_state(lay0, pool0.layer[0], 1), st[0]) > 2 * _rel(W.reference_state(lay, pool.layer[0], 1), st[0])
    assert not pool.layer[1]["z"].any()  # other layers untouched


@pytest.mark.parametrize("steps", [1, 5, 16])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("logged", [0, 7])
@pytest.mark.parametrize("bits", [8, 4])
def test_verify_matches_the_recurrence(steps, batch, logged, bits):
    lay, raw, pool = _pool(bits=bits)
    views = pool.layer[0]
    slots = torch.tensor([2, 5, 1][:batch], dtype=torch.int32, device="cuda")
    W.quantize(lay, views, _state(batch), slots)
    if logged:  # a partly filled window: commit `logged` tokens of an earlier round
        pre = _inputs(batch * logged, seed=9)
        rk, ru, rg = _ring(batch, logged)
        W.window_verify(lay, views, slots=slots, steps=logged, ring_k=rk, ring_u=ru, ring_g=rg, **pre)
        W.window_commit(lay, pool.single[0], rk[None], ru[None], rg[None], slots,
                        torch.full((batch,), logged, dtype=torch.int32, device="cuda"))
        assert int(views["log_len"][int(slots[0])][0]) == logged
    before = raw.clone()
    ins = _inputs(batch * steps)
    rk, ru, rg = _ring(batch, 16)
    out = W.window_verify(lay, views, slots=slots, steps=steps, ring_k=rk, ring_u=ru, ring_g=rg, **ins)
    assert torch.equal(raw, before)  # verification never writes the state
    for i in range(batch):
        s0 = W.reference_state(lay, views, int(slots[i]))
        rows = slice(i * steps, (i + 1) * steps)
        ref, (kn, us, g), _ = W.reference_block(s0, ins["q"][0, rows], ins["k"][0, rows], ins["v"][0, rows],
                                                ins["a"][rows], ins["b"][rows], ins["A_log"], ins["dt_bias"])
        assert _rel(out[0, rows], ref) < 4e-3  # BF16 outputs
        assert _rel(ru[i, :, :steps].transpose(0, 1), us) < 1e-3  # FP16 records
        assert _rel(rk[i, :, :steps].transpose(0, 1), kn) < 1e-3
        assert torch.allclose(rg[i, :, :steps].T.double(), g, rtol=1e-5, atol=1e-6)


def test_verify_solve_holds_on_repeated_keys():
    """Sixteen identical keys with beta ~1 and no decay: the in-block inverse's series (I - A)(I + A^2)... has
    binomial-size intermediate terms that cancel; the records stay within FP16 rounding."""
    lay, _, pool = _pool()
    views = pool.layer[0]
    slots = torch.tensor([1], dtype=torch.int32, device="cuda")
    W.quantize(lay, views, _state(1), slots)
    ins = _inputs(16)
    ins["k"][:] = ins["k"][:, :1]
    ins["b"][:] = 12.
    ins["a"][:] = -30.
    rk, ru, rg = _ring(1, 16)
    out = W.window_verify(lay, views, slots=slots, steps=16, ring_k=rk, ring_u=ru, ring_g=rg, **ins)
    ref, (_, us, _), _ = W.reference_block(W.reference_state(lay, views, 1), ins["q"][0], ins["k"][0], ins["v"][0],
                                           ins["a"], ins["b"], ins["A_log"], ins["dt_bias"])
    assert _rel(out[0], ref) < 4e-3 and _rel(ru[0].transpose(0, 1), us) < 1e-3


@pytest.mark.parametrize("steps", [1, 16])
@pytest.mark.parametrize("bits", [8, 4])
def test_fused_norm_epilogue_matches_the_separate_kernels(steps, bits):
    from sharar.kernels.fp8_quant import quant_fp8
    from sharar.kernels.verify_norm import verify_gated_norm
    lay, raw, pool = _pool(bits=bits)
    views = pool.layer[0]
    batch = 3
    slots = torch.tensor([2, 5, 1], dtype=torch.int32, device="cuda")
    W.quantize(lay, views, _state(batch), slots)
    ins = _inputs(batch * steps)
    tokens = batch * steps
    g = torch.Generator(device="cuda").manual_seed(4)
    gate = torch.randn((tokens, lay.value_heads * lay.v), device="cuda", generator=g).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(lay.v, device="cuda", generator=g)).to(torch.bfloat16)
    out_scale = torch.tensor([0.004], device="cuda")
    rk, ru, rg = _ring(batch, 16)
    core = W.window_verify(lay, views, slots=slots, steps=steps, ring_k=rk, ring_u=ru, ring_g=rg, **ins)
    want = quant_fp8(verify_gated_norm(core.reshape(tokens, -1).contiguous(), gate, weight, 1e-6, lay.v), out_scale)
    rk2, ru2, rg2 = _ring(batch, 16)
    got = W.window_verify(lay, views, slots=slots, steps=steps, ring_k=rk2, ring_u=ru2, ring_g=rg2,
                          norm=(gate, weight, 1e-6, out_scale, "swish"), **ins)
    assert got.dtype == torch.float8_e4m3fn and got.shape == want.shape
    assert torch.equal(ru2, ru) and torch.equal(rk2, rk) and torch.equal(rg2, rg)  # the same records
    # Only the norm's scalars differ (FP32 here, FP64 there): at most one FP8 step, and rarely.
    got, want = got.float(), want.float()
    step = torch.maximum(want.abs(), got.abs()) / 8 + 2 ** -9
    assert ((got - want).abs() <= step).all()
    assert (got != want).float().mean() < 0.01


@pytest.mark.parametrize("steps", [1, 5, 16])
def test_fused_sigmoid_norm_epilogue_matches_the_separate_kernel(steps):
    """Flash-Next's epilogue: the sigmoid-gated RMS norm of `rms_norm_gated`, in BF16 (its out projection quantizes)."""
    from sharar.kernels.fla.layernorm_gated import rms_norm_gated
    lay, raw, pool = _pool()
    views = pool.layer[0]
    batch = 3
    slots = torch.tensor([2, 5, 1], dtype=torch.int32, device="cuda")
    W.quantize(lay, views, _state(batch), slots)
    ins = _inputs(batch * steps)
    tokens = batch * steps
    g = torch.Generator(device="cuda").manual_seed(5)
    gate = torch.randn((tokens, lay.value_heads * lay.v), device="cuda", generator=g).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(lay.v, device="cuda", generator=g)).to(torch.bfloat16).repeat(lay.value_heads)
    rk, ru, rg = _ring(batch, 16)
    core = W.window_verify(lay, views, slots=slots, steps=steps, ring_k=rk, ring_u=ru, ring_g=rg, **ins)
    want = rms_norm_gated(x=core.reshape(tokens, -1), weight=weight, z=gate, eps=1e-6, group_size=lay.v,
                          activation="sigmoid")
    rk2, ru2, rg2 = _ring(batch, 16)
    got = W.window_verify(lay, views, slots=slots, steps=steps, ring_k=rk2, ring_u=ru2, ring_g=rg2,
                          norm=(gate, weight, 1e-6, None, "sigmoid"), **ins)
    assert got.dtype == torch.bfloat16 and got.shape == want.shape
    assert torch.equal(ru2, ru) and torch.equal(rk2, rk) and torch.equal(rg2, rg)  # the same records
    # Only the order of the variance sum may differ: at most one BF16 step, and rarely.
    got, want = got.float(), want.float()
    step = torch.maximum(want.abs(), got.abs()) / 128 + 2 ** -126
    assert ((got - want).abs() <= step).all()
    assert (got != want).float().mean() < 0.01


@pytest.mark.parametrize("steps", [1, 5, 16])
def test_fused_group_quantization_is_quant_groups_of_the_bf16_epilogue(steps):
    """Flash-Next's 3-bit out projection input: E4M3 per (token, 32-group) and the group scales, as `w3.quant_groups`
    of the epilogue's BF16 output (its arithmetic, silu_mul_quant_groups'). The two compiled epilogues may round a
    rare normalized value one BF16 step apart, which moves that value by at most one E4M3 step."""
    from sharar.kernels import w3
    lay, raw, pool = _pool()
    views = pool.layer[0]
    batch = 3
    slots = torch.tensor([2, 5, 1], dtype=torch.int32, device="cuda")
    W.quantize(lay, views, _state(batch), slots)
    ins = _inputs(batch * steps)
    tokens = batch * steps
    g = torch.Generator(device="cuda").manual_seed(6)
    gate = torch.randn((tokens, lay.value_heads * lay.v), device="cuda", generator=g).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(lay.v, device="cuda", generator=g)).to(torch.bfloat16)
    rk, ru, rg = _ring(batch, 16)
    bf16 = W.window_verify(lay, views, slots=slots, steps=steps, ring_k=rk, ring_u=ru, ring_g=rg,
                           norm=(gate, weight, 1e-6, None, "sigmoid"), **ins)
    want_q, want_s = w3.quant_groups(bf16)
    rk2, ru2, rg2 = _ring(batch, 16)
    got_q, got_s = W.window_verify(lay, views, slots=slots, steps=steps, ring_k=rk2, ring_u=ru2, ring_g=rg2,
                                   norm=(gate, weight, 1e-6, "groups", "sigmoid"), **ins)
    assert torch.equal(ru2, ru) and torch.equal(rk2, rk) and torch.equal(rg2, rg)
    assert got_q.shape == want_q.shape and got_s.shape == want_s.shape
    assert ((got_s - want_s).abs() <= want_s / 128).all() and (got_s != want_s).float().mean() < 1e-2
    got, want = got_q.float(), want_q.float()
    step = torch.maximum(want.abs(), got.abs()) / 8 + 2 ** -9
    assert ((got - want).abs() <= step).all()
    assert (got != want).float().mean() < 1e-3


@pytest.mark.parametrize("window", [8, 16, 32])
@pytest.mark.parametrize("bits", [8, 4])
def test_commits_append_until_the_window_fills_then_fold(window, bits):
    lay, raw, pool = _pool(window=window, bits=bits)
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    for layer in range(2):
        W.quantize(lay, pool.layer[layer], _state(2, seed=layer), slots)
    steps, lengths = 16, [0, 0]
    rng = torch.Generator().manual_seed(window)
    for round_ in range(8):
        counts = [int(torch.randint(1, steps + 1, (1,), generator=rng)) for _ in range(2)]
        if round_ == 2:
            counts[1] = 0  # device acceptance's padded rows commit nothing
        ins = [_inputs(2 * steps, seed=100 * round_ + layer) for layer in range(2)]
        rk, ru, rg = _ring(2, steps, layers=2)
        expected, untouched = [], raw[:, [0, 2, 4, 5]].clone()
        for layer in range(2):
            views = pool.layer[layer]
            W.window_verify(lay, views, slots=slots, steps=steps, ring_k=rk[layer], ring_u=ru[layer], ring_g=rg[layer], **ins[layer])
            for i in range(2):
                s0 = W.reference_state(lay, views, int(slots[i]))
                rows = slice(i * steps, i * steps + counts[i])
                x = ins[layer]
                expected.append(W.reference_block(s0, x["q"][0, rows], x["k"][0, rows], x["v"][0, rows], x["a"][rows],
                                                  x["b"][rows], x["A_log"], x["dt_bias"])[2] if counts[i] else s0)
        W.window_commit(lay, pool.all, rk, ru, rg, slots, torch.tensor(counts, dtype=torch.int32, device="cuda"))
        assert torch.equal(raw[:, [0, 2, 4, 5]], untouched)  # the dummy slot and other slots
        for layer in range(2):
            for i in range(2):
                n, c = lengths[i], counts[i]
                folded = c and n + c >= window
                got = W.reference_state(lay, pool.layer[layer], int(slots[i]))
                # An append only adds FP16 records (keys rounded before their corrections are computed).
                assert _rel(got, expected[2 * layer + i]) < (FOLD[bits] if folded else 5e-4), (round_, layer, i, n, c)
                want = n if not c else (0 if folded else n + c)
                assert pool.layer[layer]["log_len"][int(slots[i])].tolist() == [want] * HV
        lengths = [n if not c else (0 if c and n + c >= window else n + c) for n, c in zip(lengths, counts)]


@pytest.mark.parametrize("bits", [8, 4])
def test_materialize_is_the_committed_state(bits):
    lay, _, pool = _pool(bits=bits)
    views = pool.layer[1]
    slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    W.quantize(lay, views, _state(2), slots)
    rk, ru, rg = _ring(2, 9)
    W.window_verify(lay, views, slots=slots, steps=9, ring_k=rk, ring_u=ru, ring_g=rg, **_inputs(18))
    W.window_commit(lay, pool.single[1], rk[None], ru[None], rg[None], slots, torch.tensor([9, 4], dtype=torch.int32, device="cuda"))
    states = W.materialize(lay, views, slots)
    for i in range(2):
        assert _rel(states[i], W.reference_state(lay, views, int(slots[i]))) < 1e-6
    # The prefill round trip: back into the slots as new boundary states.
    W.quantize(lay, views, states, slots)
    assert _rel(W.materialize(lay, views, slots), states) < QUANT[bits]
    assert not views["log_len"][1].any()


@pytest.mark.parametrize("bits", [8, 4])
def test_graph_replay_matches_eager(bits):
    lay, raw, pool = _pool(bits=bits)
    slots = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    for layer in range(2):
        W.quantize(lay, pool.layer[layer], _state(2, seed=layer), slots)
    start = raw.clone()
    ins = [_inputs(32, seed=layer) for layer in range(2)]
    rk, ru, rg = _ring(2, 16, layers=2)
    counts = torch.tensor([16, 6], dtype=torch.int32, device="cuda")
    outs = []

    def round_():
        outs.clear()
        for layer in range(2):
            outs.append(W.window_verify(lay, pool.layer[layer], slots=slots, steps=16, ring_k=rk[layer], ring_u=ru[layer],
                                        ring_g=rg[layer], **ins[layer]))
        W.window_commit(lay, pool.all, rk, ru, rg, slots, counts)

    round_()
    eager_out, eager_state = [o.clone() for o in outs], raw.clone()
    raw.copy_(start)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        round_()
    raw.copy_(start)
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(outs, eager_out)) and torch.equal(raw, eager_state)


@pytest.mark.parametrize("bits", [8, 4])
def test_per_window_error_stays_far_below_per_token_quantization(bits):
    """Over 768 tokens committed in speculative rounds, the window state tracks the FP64 trajectory, while
    requantizing a plain state of the same width after every token drifts away."""
    lay, _, pool = _pool(bits=bits)
    views = pool.layer[0]
    slot = torch.tensor([1], dtype=torch.int32, device="cuda")
    s0 = _state(1)
    W.quantize(lay, views, s0, slot)
    exact = W.reference_state(lay, views, 1)
    naive = exact.clone()
    tokens, steps, pos = 768, 16, 0
    x = _inputs(tokens + steps, seed=3)
    rng = torch.Generator().manual_seed(5)

    qmax = 2 ** (bits - 1) - 1

    def per_token(st):
        amax = st.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        return torch.round(st / amax * qmax) * amax / qmax

    rk, ru, rg = _ring(1, steps)
    while pos < tokens:
        count = int(torch.randint(1, 9, (1,), generator=rng))
        rows = slice(pos, pos + steps)
        W.window_verify(lay, views, q=x["q"][:, rows], k=x["k"][:, rows], v=x["v"][:, rows], a=x["a"][rows], b=x["b"][rows],
                        A_log=x["A_log"], dt_bias=x["dt_bias"], slots=slot, steps=steps, ring_k=rk, ring_u=ru, ring_g=rg)
        W.window_commit(lay, pool.single[0], rk[None], ru[None], rg[None], slot, torch.tensor([count], dtype=torch.int32, device="cuda"))
        for t in range(pos, pos + count):
            one = slice(t, t + 1)
            args = (x["q"][0, one], x["k"][0, one], x["v"][0, one], x["a"][one], x["b"][one], x["A_log"], x["dt_bias"])
            exact = W.reference_block(exact, *args)[2]
            naive = per_token(W.reference_block(naive, *args)[2])
        pos += count
    window_error = _rel(W.reference_state(lay, views, 1), exact)
    naive_error = _rel(naive, exact)
    assert window_error < FOLD[bits] * 1.5 and window_error * 4 < naive_error, (window_error, naive_error)
