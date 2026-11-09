# SPDX-License-Identifier: Apache-2.0
"""Mamba-2 mixer kernels (`sharar/kernels/mamba2.py`) against a plain FP32 sequential recurrence.

The reference convolves each token from its three-token window (rounding the activation to BF16, as every path
does before the scan), then steps h' = exp(dt A) h + dt x B^T, y = h' C + D x token by token. Prefill (chunked
SSD) is compared within tolerance; decode, verify and replay share one recurrence and are compared bitwise with each
other. Parameters are layer 0's when the checkpoint is present (random otherwise).

Every state test runs in both storage formats: FP32 and FP16 stochastically rounded ("sr", keyed on the token
positions and layer index; the served `SHARAR_SSM_STATE=fp16_sr`). Within a format the bitwise contracts hold
unchanged (verify == decode, replay == decode, 128-aligned prefill splits, checkpoints); against the FP32 reference
the FP16 states carry their rounding error. The stochastic rounding itself is checked for unbiasedness and
determinism.

The quantized window formats (int8, int4) are checked against the FP32 recurrence from the state a slot holds: verify
and the commit of every count, decode across several folds, the prefill on FP32 copies, and CUDA-graph capture.
"""

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sharar.kernels import mamba2 as m

FORMATS = ("fp32", "sr")
DTYPE = {"fp32": torch.float32, "sr": torch.float16}
# Relative error of the state against the FP32 reference after a few steps: each FP16 store adds at most one FP16
# ulp (2^-10 relative).
STATE_TOL = {"fp32": 1e-6, "sr": 2e-3}
PREFILL_STATE_TOL = {"fp32": 5e-5, "sr": 2e-3}

DEV = "cuda"
H, P, N, G, C, DI = m.HEADS, m.HEAD_DIM, m.STATE, m.GROUPS, m.CONV_DIM, m.D_INNER


def _layer0():
    """Layer 0's mixer tensors and its pre-norm weight, or None without the checkpoint."""
    import os
    from pathlib import Path
    from sharar.target import NEMOTRON
    try:
        snap = NEMOTRON.snapshot_dir()
    except FileNotFoundError:  # the shard is enough here: fall back to the default HF cache, index or not
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        snap = hub / NEMOTRON.model_cache_key / "snapshots" / NEMOTRON.model_revision
    shard = snap / "model-00001-of-00088.safetensors"
    if not shard.exists():
        return None
    from safetensors import safe_open
    p = "language_model.backbone.layers.0."
    with safe_open(str(shard), "pt", device="cpu") as f:
        names = ("A_log", "D", "dt_bias", "conv1d.weight", "conv1d.bias", "norm.weight", "in_proj.weight",
                 "in_proj.weight_scale")
        t = {k: f.get_tensor(p + "mixer." + k) for k in names}
        t["prenorm"] = f.get_tensor(p + "norm.weight")
    return t


@pytest.fixture(scope="module")
def layer():
    torch.manual_seed(0)
    t = _layer0()
    if t is None:
        params = m.prepare_params(torch.randn(C, 1, 4, device=DEV) * 0.5, torch.randn(C, device=DEV) * 0.1,
                                  torch.log(torch.rand(H, device=DEV) * 15 + 1), torch.randn(H, device=DEV),
                                  torch.rand(H, device=DEV) * -4, torch.randn(DI, device=DEV) * 0.2 + 1, 1e-5)
        return params, None
    params = m.prepare_params(*(t[k].to(DEV) for k in ("conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias",
                                                        "norm.weight")), 1e-5)
    w = (t["in_proj.weight"].to(DEV).float() * t["in_proj.weight_scale"].to(DEV).float()).bfloat16()
    return params, (w, t["prenorm"].to(DEV))


def _inputs(layer, t: int, seed: int) -> torch.Tensor:
    """[t, z | xBC | dt] bf16: layer 0's in_proj of RMS-normalized random hidden states, or Gaussian rows."""
    params, proj = layer
    g = torch.Generator(device=DEV).manual_seed(seed)
    if proj is None:
        return torch.randn(t, m.PROJ_DIM, device=DEV, generator=g).bfloat16()
    w, prenorm = proj
    h = torch.randn(t, w.shape[1], device=DEV, generator=g)
    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16() * prenorm
    return h @ w.T


def _reference(params, zx, hist, state):
    """Sequential FP32 mixer core: (y [T, d_inner], final state, final conv window, state after every token)."""
    xbc, dtr = zx[:, DI:DI + C].float(), zx[:, DI + C:].float()
    w, b = params.conv_weight.float(), params.conv_bias.float()
    win = torch.cat([hist.float().T, xbc])
    s, ys, states = state.clone().float(), [], []
    for t in range(zx.shape[0]):
        cv = F.silu(b + (w * win[t:t + 4].T).sum(1)).bfloat16().float()
        x = cv[:DI].view(H, P)
        bm = cv[DI:DI + G * N].view(G, N).repeat_interleave(H // G, 0)
        cm = cv[DI + G * N:].view(G, N).repeat_interleave(H // G, 0)
        dt = F.softplus(dtr[t] + params.dt_bias)
        s = s * torch.exp(dt * params.A)[:, None, None] + (dt[:, None] * x)[:, :, None] * bm[:, None, :]
        ys.append(((s * cm[:, None, :]).sum(-1) + params.D[:, None] * x).reshape(-1))
        states.append(s)
    return torch.stack(ys), s, win[-3:].T.bfloat16(), states


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _pools(slots, seed=1, fmt="fp32"):
    g = torch.Generator(device=DEV).manual_seed(seed)
    conv = torch.randn(slots, C, 3, device=DEV, generator=g).bfloat16()
    ssm = (torch.randn(slots, H, P, N, device=DEV, generator=g) * 0.05).to(DTYPE[fmt])
    return conv, ssm


def _kw(fmt, layer=0):
    return dict(layer=layer)


def _positions(start, rows, steps=1):
    """int64 positions of `rows` sequences continuing at start[i], `steps` tokens each, row-major."""
    return torch.tensor([s + t for s in start[:rows] for t in range(steps)], dtype=torch.int64, device=DEV)


@pytest.mark.parametrize("lengths,prefixes", [([2048], [0]), ([2048], [2048]),
                                           ([2048, 257], [17, 2180]), ([2048, 1, 3], [0, 7, 128])])
@pytest.mark.parametrize("fmt", FORMATS)
def test_normalized_prefill_matches_separate_scan_and_norm(layer, lengths, prefixes, fmt, monkeypatch):
    """Exercise the served fusion, ragged tails, interior checkpoints and untouched slots."""
    from sharar.kernels import mamba2_prefill
    params, _ = layer
    slots = [3, 1, 5][:len(lengths)]
    checkpoints = [(i, min(127, n - 1), 6 + i) for i, n in enumerate(lengths) if n > 128]
    plan = m.plan_prefill(lengths, prefixes, slots, DEV, checkpoints)
    zx = _inputs(layer, sum(lengths), 1237)
    conv, ssm = _pools(10, fmt=fmt)
    conv0, ssm0 = conv.clone(), ssm.clone()
    expected = m.gated_rmsnorm(m.prefill(zx, params, conv, ssm, plan), zx[:, :DI], params)
    ref_conv, ref_ssm = conv.clone(), ssm.clone()
    conv.copy_(conv0)
    ssm.copy_(ssm0)
    calls = []
    original = mamba2_prefill.scan_norm
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(mamba2_prefill, "scan_norm", counted)
    actual = m.prefill(zx, params, conv, ssm, plan, normalize=True)
    assert bool(calls) == (torch.cuda.get_device_capability() == (12, 1))
    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=4e-3)
    assert torch.equal(conv, ref_conv)
    assert torch.equal(ssm, ref_ssm)
    # Graph capture must use the same slot/plan addresses without host device-value reads.
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        conv.copy_(conv0)
        ssm.copy_(ssm0)
        captured = m.prefill(zx, params, conv, ssm, plan, normalize=True)
    g.replay()
    torch.testing.assert_close(captured, expected, rtol=4e-3, atol=4e-3)
    assert torch.equal(conv, ref_conv)
    assert torch.equal(ssm, ref_ssm)


@pytest.mark.parametrize("fmt,t,quant", [("fp32", 17, False), ("sr", 17, False),
                                      ("fp32", 2048, True)])
def test_normalized_prefill_fallbacks(layer, fmt, t, quant, monkeypatch):
    from sharar.kernels import mamba2_prefill
    params, _ = layer
    plan = m.plan_prefill([t], [17], [1], DEV)
    zx = _inputs(layer, t, 421)
    conv, ssm = _pools(2, fmt=fmt)
    conv0, ssm0 = conv.clone(), ssm.clone()
    scale = torch.tensor(0.125, device=DEV) if quant else None
    expected = m.gated_rmsnorm(m.prefill(zx, params, conv, ssm, plan), zx[:, :DI], params, scale)
    ref_conv, ref_ssm = conv.clone(), ssm.clone()
    conv.copy_(conv0)
    ssm.copy_(ssm0)
    def fail(*args, **kwargs):
        raise AssertionError("unsupported contract must use the separate kernels")
    monkeypatch.setattr(mamba2_prefill, "scan_norm", fail)
    actual = m.prefill(zx, params, conv, ssm, plan, normalize=True, out_scale=scale)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(conv, ref_conv)
    assert torch.equal(ssm, ref_ssm)


@pytest.mark.parametrize("fmt", FORMATS)
def test_decode_matches_reference(layer, fmt):
    params, _ = layer
    conv, ssm = _pools(8, fmt=fmt)
    conv0, ssm0 = conv.clone(), ssm.clone()
    slots = torch.tensor([5, 2, 7], dtype=torch.int32, device=DEV)
    steps = 6
    zx = _inputs(layer, 3 * steps, 11).view(3, steps, -1)
    ys = [m.decode(zx[:, t].contiguous(), params, conv, ssm, slots, _positions([40, 3, 900], 3) + t, **_kw(fmt))
          for t in range(steps)]
    for i, slot in enumerate(slots.tolist()):
        y_ref, s_ref, h_ref, _ = _reference(params, zx[i], conv0[slot], ssm0[slot])
        y = torch.stack([ys[t][i] for t in range(steps)])
        assert _rel(y, y_ref) < 4e-3  # y is BF16
        assert _rel(ssm[slot], s_ref) < STATE_TOL[fmt]
        assert torch.equal(conv[slot], h_ref)
    untouched = [0, 1, 3, 4, 6]
    assert torch.equal(ssm[untouched], ssm0[untouched]) and torch.equal(conv[untouched], conv0[untouched])


@pytest.mark.parametrize("fmt", FORMATS)
def test_prefill_varlen_matches_reference(layer, fmt):
    """Fresh and continued sequences in one batch (prefixes off the 128 grid), with checkpoints; fresh slots hold
    garbage that must be ignored."""
    params, _ = layer
    conv, ssm = _pools(12, fmt=fmt)
    conv0, ssm0 = conv.clone(), ssm.clone()
    lens, prefixes, slots = [300, 1, 129, 133, 2], [0, 5, 0, 70, 3], [3, 4, 5, 6, 7]
    checkpoints = [(0, 128, 8), (0, 200, 9), (3, 58, 10), (3, 3, 11)]  # (seq, local end, slot)
    zx = _inputs(layer, sum(lens), 12)
    plan = m.plan_prefill(lens, prefixes, slots, DEV, checkpoints)
    y = m.prefill(zx, params, conv, ssm, plan, **_kw(fmt))
    tol = PREFILL_STATE_TOL[fmt]
    start = 0
    zero_h, zero_s = torch.zeros(C, 3, device=DEV, dtype=torch.bfloat16), torch.zeros(H, P, N, device=DEV)
    for i, (n, p, slot) in enumerate(zip(lens, prefixes, slots)):
        h0, s0 = (conv0[slot], ssm0[slot]) if p else (zero_h, zero_s)
        y_ref, s_ref, h_ref, states = _reference(params, zx[start:start + n], h0, s0)
        assert _rel(y[start:start + n], y_ref) < 4e-3, i
        assert _rel(ssm[slot], s_ref) < tol, i
        assert torch.equal(conv[slot], h_ref), i
        for seq, local, ck in checkpoints:
            if seq == i:
                assert _rel(ssm[ck], states[local - 1]) < tol, (i, local)
                window = torch.cat([h0.T, zx[start:start + n, DI:DI + C]])[local:local + 3]
                assert torch.equal(conv[ck], window.T), (i, local)
        start += n
    assert torch.equal(ssm[[0, 1, 2]], ssm0[[0, 1, 2]])


@pytest.mark.parametrize("fmt", FORMATS)
def test_prefill_equals_stepwise_decode(layer, fmt):
    params, _ = layer
    conv, ssm = _pools(4, fmt=fmt)
    conv[1], ssm[1] = conv[0], ssm[0]
    zx = _inputs(layer, 260, 13)
    y_p = m.prefill(zx, params, conv, ssm, m.plan_prefill([260], [17], [0], DEV), **_kw(fmt))
    slot = torch.tensor([1], dtype=torch.int32, device=DEV)
    y_d = torch.cat([m.decode(zx[t:t + 1], params, conv, ssm, slot, _positions([17 + t], 1), **_kw(fmt))
                     for t in range(260)])
    assert _rel(y_p, y_d) < 4e-3
    # 260 decode stores against 3 chunk-end roundings: the FP16 paths differ by their rounding noise
    assert _rel(ssm[0], ssm[1]) < {"fp32": 5e-5, "sr": 5e-3}[fmt]
    assert torch.equal(conv[0], conv[1])


@pytest.mark.parametrize("fmt", FORMATS)
def test_chunked_prefill_continuity(layer, fmt):
    """Chunked prefill against one pass: bitwise at 128-aligned splits (the chunking, and in FP16 the chunk-end
    roundings, are identical), within tolerance elsewhere; the conv state carries the window across every split."""
    params, _ = layer
    zx = _inputs(layer, 700, 14)
    conv, ssm = _pools(4, fmt=fmt)
    y_full = m.prefill(zx, params, conv, ssm, m.plan_prefill([700], [0], [1], DEV), **_kw(fmt))
    for splits, exact in (([256, 384, 60], True), ([255, 301, 144], False), ([1, 2, 697], False)):
        conv2, ssm2 = _pools(4, seed=2, fmt=fmt)
        ys, done = [], 0
        for n in splits:
            ys.append(m.prefill(zx[done:done + n], params, conv2, ssm2, m.plan_prefill([n], [done], [1], DEV),
                                **_kw(fmt)))
            done += n
        y = torch.cat(ys)
        assert torch.equal(conv2[1], conv[1])
        if exact:
            assert torch.equal(y, y_full) and torch.equal(ssm2[1], ssm[1])
        else:
            assert _rel(y, y_full) < 4e-3 and _rel(ssm2[1], ssm[1]) < PREFILL_STATE_TOL[fmt]


@pytest.mark.parametrize("fmt", FORMATS)
def test_checkpoint_is_the_shorter_prefill(layer, fmt):
    """A checkpoint holds bitwise the state a prefill ending there leaves (it ends a chunk, and in FP16 is rounded
    where the shorter run rounds its final state), so a prefix hit continues exactly as the cached run did."""
    params, _ = layer
    zx = _inputs(layer, 400, 15)
    for local, ck in ((200, 2), (256, 3)):
        conv, ssm = _pools(6, fmt=fmt)
        m.prefill(zx, params, conv, ssm, m.plan_prefill([400], [0], [1], DEV, [(0, local, ck)]), **_kw(fmt))
        conv2, ssm2 = _pools(6, seed=3, fmt=fmt)
        m.prefill(zx[:local], params, conv2, ssm2, m.plan_prefill([local], [0], [4], DEV), **_kw(fmt))
        assert torch.equal(ssm[ck], ssm2[4]) and torch.equal(conv[ck], conv2[4]), local


class _Pool:
    def __init__(self, layers, slots, fmt="fp32"):
        self.layer_ids = [2 * j for j in range(layers)]
        self.conv_all = torch.stack([_pools(slots, seed=10 + j)[0] for j in range(layers)])
        self.ssm_all = torch.stack([_pools(slots, seed=10 + j, fmt=fmt)[1] for j in range(layers)])
        self.conv = {lid: self.conv_all[j] for j, lid in enumerate(self.layer_ids)}
        self.ssm = {lid: self.ssm_all[j] for j, lid in enumerate(self.layer_ids)}


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("batch,steps", [(1, 2), (3, 4), (2, 5), (4, 9)])
def test_verify_snapshots_replay_bitwise(layer, batch, steps, fmt):
    """verify == stepwise decode (outputs, per-step states, conv windows) with the committed slots untouched, and the
    replay of every count == that many decode steps, for every layer in one launch; snapshots in the pool's dtype
    or FP32 hold the same values."""
    params, _ = layer
    pool = _Pool(2, 10, fmt)
    ssm0, conv0 = pool.ssm_all.clone(), pool.conv_all.clone()
    capacity, max_steps = 5, 9
    replay = m.AcceptedState(pool, capacity, max_steps, G)
    windows = torch.full((2, capacity, max_steps, C, 3), -7.0, device=DEV, dtype=torch.bfloat16)
    snapshots = torch.zeros(capacity, max_steps, H, P, N, device=DEV, dtype=DTYPE[fmt])
    snapshots32 = torch.zeros(capacity, max_steps, H, P, N, device=DEV)
    slots = torch.tensor([7, 2, 9, 4][:batch], dtype=torch.int32, device=DEV)
    rows = torch.arange(batch, dtype=torch.int32, device=DEV)
    start = [100, 5, 77, 4000]
    pos = _positions(start, batch, steps)
    zx = torch.stack([_inputs(layer, batch * steps, 20 + j) for j in range(2)])
    ys = [m.verify(zx[j], params, pool.conv_all[j], pool.ssm_all[j], slots, steps, windows[j],
                   replay=replay.layer_inputs[lid], positions=pos, **_kw(fmt, j))
          for j, lid in enumerate(pool.layer_ids)]
    y_snap = m.verify(zx[0], params, pool.conv_all[0], pool.ssm_all[0], slots, steps,
                      snapshots=snapshots, snapshot_indices=rows, positions=pos, **_kw(fmt))
    m.verify(zx[0], params, pool.conv_all[0], pool.ssm_all[0], slots, steps,
             snapshots=snapshots32, snapshot_indices=rows, positions=pos, **_kw(fmt))
    assert torch.equal(pool.ssm_all, ssm0) and torch.equal(pool.conv_all, conv0)
    assert torch.equal(y_snap, ys[0]) and torch.equal(snapshots32, snapshots.float())
    decoded = []
    for j in range(2):
        conv, ssm, states = conv0[j].clone(), ssm0[j].clone(), []
        for t in range(steps):
            y = m.decode(zx[j].view(batch, steps, -1)[:, t].contiguous(), params, conv, ssm, slots,
                         _positions(start, batch) + t, **_kw(fmt, j))
            assert torch.equal(y, ys[j].view(batch, steps, -1)[:, t])
            assert torch.equal(windows[j, :batch, t], conv[slots.long()])
            states.append(ssm[slots.long()].clone())
        decoded.append(states)
    for t in range(steps):
        assert torch.equal(snapshots[:batch, t], decoded[0][t])
    y_ref, _, _, _ = _reference(params, zx[0][:steps], conv0[0, slots[0]], ssm0[0, slots[0]])
    assert _rel(ys[0][:steps], y_ref) < 4e-3
    for count in range(steps + 1):
        counts = torch.tensor([(count + i) % (steps + 1) for i in range(batch)], dtype=torch.int32, device=DEV)
        pool.ssm_all.copy_(ssm0)
        pool.conv_all.copy_(conv0)
        replay.run(pool, slots, counts)
        m.commit_conv(pool.conv_all, windows, slots, counts)
        for j in range(2):
            for i, (slot, c) in enumerate(zip(slots.tolist(), counts.tolist())):
                assert torch.equal(pool.ssm_all[j, slot], decoded[j][c - 1][i] if c else ssm0[j, slot])
                assert torch.equal(pool.conv_all[j, slot], windows[j, i, c - 1] if c else conv0[j, slot])
        others = [s for s in range(10) if s not in slots.tolist()]
        assert torch.equal(pool.ssm_all[:, others], ssm0[:, others])
        assert torch.equal(pool.conv_all[:, others], conv0[:, others])


@pytest.mark.parametrize("fmt", FORMATS)
def test_verify_and_replay_under_graphs(layer, fmt):
    params, _ = layer
    pool = _Pool(2, 6, fmt)
    replay = m.AcceptedState(pool, 4, 5, G)
    windows = torch.zeros(2, 4, 5, C, 3, device=DEV, dtype=torch.bfloat16)
    slots = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    counts = torch.tensor([2, 5], dtype=torch.int32, device=DEV)
    pos = _positions([12, 300], 2, 5)
    zx = torch.stack([_inputs(layer, 10, 30 + j) for j in range(2)])
    ssm0, conv0 = pool.ssm_all.clone(), pool.conv_all.clone()

    def body():
        out = [m.verify(zx[j], params, pool.conv_all[j], pool.ssm_all[j], slots, 5, windows[j],
                        replay=replay.layer_inputs[lid], positions=pos, **_kw(fmt, j))
               for j, lid in enumerate(pool.layer_ids)]
        replay.run(pool, slots, counts)
        m.commit_conv(pool.conv_all, windows, slots, counts)
        return out

    eager = body()
    ssm1, conv1 = pool.ssm_all.clone(), pool.conv_all.clone()
    pool.ssm_all.copy_(ssm0)
    pool.conv_all.copy_(conv0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = body()
    pool.ssm_all.copy_(ssm0)
    pool.conv_all.copy_(conv0)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))
    assert torch.equal(pool.ssm_all, ssm1) and torch.equal(pool.conv_all, conv1)


def test_gated_rmsnorm(layer):
    """transformers' Zamba2RMSNormGated (norm_before_gate=False, group 1024) and the FP8 out-projection input."""
    params, _ = layer
    y = torch.randn(37, DI, device=DEV).bfloat16()
    zxbcdt = torch.randn(37, m.PROJ_DIM, device=DEV).bfloat16()
    z = zxbcdt[:, :DI]
    h = y.float() * F.silu(z.float())
    hg = h.view(37, G, -1)
    h = (hg * torch.rsqrt(hg.pow(2).mean(-1, keepdim=True) + params.eps)).view(37, DI)
    ref = params.norm_weight * h.bfloat16()
    out = m.gated_rmsnorm(y, z, params)
    assert (out.float() - ref.float()).abs().max().item() <= 2 ** -7 * ref.float().abs().max().item()
    assert _rel(out, ref) < 2e-3
    scale = torch.tensor(0.02, device=DEV)
    q = m.gated_rmsnorm(y, z, params, scale)
    expected = (out.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))


def test_plan_rejects_bad_checkpoints():
    with pytest.raises(ValueError):
        m.plan_prefill([100], [0], [1], DEV, [(0, 2, 3)])
    with pytest.raises(ValueError):
        m.plan_prefill([100], [0], [1], DEV, [(0, 100, 3)])
    plan = m.plan_prefill([300, 50], [100, 0], [1, 2], DEV, [(0, 64, 3)])
    # sequence 0 starts at absolute 100: chunks end at local 28, 64 (checkpoint), 156, 284 and 300
    assert plan.cu_chunks.tolist() == [0, 28, 64, 156, 284, 300, 350]
    assert plan.chunk_dest.tolist() == [-1, 3, -1, -1, -1, -1]


# ---------------------------------------------------------------------------------------------------------------------
# Stochastic rounding


@triton.jit
def _sr_probe(X, R, O, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    r = tl.load(R + i, mask, 0).to(tl.uint32, bitcast=True)
    tl.store(O + i, m._sr_fp16(tl.load(X + i, mask, 0.0), r), mask)


def _sr(x, r):
    out = torch.empty(x.shape, dtype=torch.float16, device=DEV)
    _sr_probe[(triton.cdiv(x.numel(), 1024),)](x, r, out, x.numel(), 1024)
    return out


def test_stochastic_rounding_is_unbiased_between_neighbours():
    """Every result is one of the value's two FP16 neighbours, FP16 values are kept, and the mean over random words
    is the value itself, through normal, subnormal and below-subnormal magnitudes; out-of-range values saturate."""
    vals = torch.tensor([1.0, 1.0001, -1.0001, 3.14159, 1e-3, -7e-5, 6.0e-5, 3e-6, -1e-7, 3e-8, 9e-8, 0.0, 6.5e4,
                         0.0999, 2.0 ** -14, -(2.0 ** -20) * 3], device=DEV)
    reps = 1 << 17
    x = vals.repeat_interleave(reps)
    r = torch.randint(-2 ** 31, 2 ** 31, x.shape, dtype=torch.int64, device=DEV,
                      generator=torch.Generator(device=DEV).manual_seed(0)).to(torch.int32)
    y = _sr(x, r).float().view(len(vals), reps)
    for v, row in zip(vals.tolist(), y):
        lo = torch.tensor(v, device=DEV).half()
        lo_f = lo.float().item()
        if lo_f != v and abs(lo_f) > abs(v):  # torch rounds to nearest; step towards zero for the lower neighbour
            lo = (lo.view(torch.int16) - 1).view(torch.float16)
        hi = (lo.view(torch.int16) + 1).view(torch.float16) if lo.float().item() != v else lo
        assert set(torch.unique(row).tolist()) <= {lo.float().item(), hi.float().item()}, v
        spacing = abs(hi.float().item() - lo.float().item())
        if spacing:  # the mean of a Bernoulli over the spacing: within 5 standard errors
            assert abs(row.double().mean().item() - v) < 5 * 0.5 * spacing / reps ** 0.5, v
        else:
            assert torch.all(row == v), v
    assert _sr(torch.tensor([7e4, -1e6, 65519.0], device=DEV),
               torch.full((3,), -1, dtype=torch.int32, device=DEV)).float().tolist() == [65504.0, -65504.0, 65504.0]


def _stagnation(fmt, steps=400):
    """How far the mean state of one slot grows from 1 with no decay (A ~ 0) and per-step increments dt x b ~ 5e-5, a
    tenth of half an FP16 ulp at 1 (2^-11): round to nearest would drop every one, stochastic rounding keeps them in
    expectation. x and B come from the convolution bias (zero inputs, zero taps)."""
    bias = torch.zeros(C, device=DEV)
    bias[:DI + G * N] = 0.1
    params = m.prepare_params(torch.zeros(C, 1, 4, device=DEV), bias, torch.full((H,), -30.0, device=DEV),
                              torch.zeros(H, device=DEV), torch.full((H,), -4.0, device=DEV),
                              torch.ones(DI, device=DEV), 1e-5)
    conv = torch.zeros(1, C, 3, device=DEV, dtype=torch.bfloat16)
    ssm = torch.ones(1, H, P, N, device=DEV, dtype=DTYPE[fmt])
    zx = torch.zeros(1, m.PROJ_DIM, device=DEV, dtype=torch.bfloat16)
    slot = torch.zeros(1, dtype=torch.int32, device=DEV)
    for t in range(steps):
        m.decode(zx, params, conv, ssm, slot, torch.tensor([t], device=DEV), **_kw(fmt))
    return ssm.double().mean().item() - 1.0


def test_stochastic_rounding_keeps_sub_ulp_updates():
    exact = _stagnation("fp32")
    assert 0.01 < exact < 0.03  # 400 updates of ~5e-5: about 20 FP16 ulps at 1, each below half an ulp
    assert _stagnation("sr") == pytest.approx(exact, rel=0.01)


def test_stochastic_rounding_is_deterministic_and_keyed():
    """The same (state, inputs, position, layer) round to the same bits whatever the slot and the launch; another
    position or layer draws other bits."""
    params, _ = _random_params()
    zx = torch.randn(2, m.PROJ_DIM, device=DEV, generator=torch.Generator(device=DEV).manual_seed(4)).bfloat16()
    zx[1] = zx[0]
    outs = {}
    for key, (slots, pos, lay) in {"a": ([1, 3], [50, 50], 0), "b": ([4, 2], [50, 50], 0),
                                   "pos": ([1, 3], [50, 51], 0), "layer": ([1, 3], [50, 50], 7)}.items():
        conv, ssm = _pools(5, fmt="sr")
        ssm[:] = ssm[1]
        conv[:] = conv[1]
        s = torch.tensor(slots, dtype=torch.int32, device=DEV)
        m.decode(zx, params, conv, ssm, s, torch.tensor(pos, device=DEV), layer=lay)
        outs[key] = (ssm[s[0]].clone(), ssm[s[1]].clone())
    assert torch.equal(outs["a"][0], outs["a"][1]) and torch.equal(outs["a"][0], outs["b"][0])
    assert not torch.equal(outs["pos"][0], outs["pos"][1])
    assert not torch.equal(outs["layer"][0], outs["a"][0])
    differ = (outs["pos"][0] != outs["pos"][1]).float().mean().item()
    assert 0.05 < differ < 0.95  # independent draws disagree on a fraction of the elements, not all


def _random_params():
    torch.manual_seed(0)
    return m.prepare_params(torch.randn(C, 1, 4, device=DEV) * 0.5, torch.randn(C, device=DEV) * 0.1,
                            torch.log(torch.rand(H, device=DEV) * 15 + 1), torch.randn(H, device=DEV),
                            torch.rand(H, device=DEV) * -4, torch.randn(DI, device=DEV) * 0.2 + 1, 1e-5), None


def test_fp16_state_needs_positions_for_stochastic_rounding(layer):
    params, _ = layer
    conv, ssm = _pools(2, fmt="sr")
    zx = _inputs(layer, 1, 5)
    slot = torch.zeros(1, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError, match="positions"):
        m.decode(zx, params, conv, ssm, slot)


# ---- Quantized window state (SsmFormat int8 / int4) ----

WINDOW = 16
RANK = {8: 4, 4: 8}  # the format's default compensators


class _WindowPool:
    """The state pool a WindowState and the window paths read: FP32-free byte rows plus BF16 conv windows."""

    def __init__(self, layers, slots, bits):
        from sharar.cache.ssm_format import WindowLayout, WindowPool
        self.layer_ids = [2 * j for j in range(layers)]
        self.layout = WindowLayout(G, H, N, P, WINDOW, RANK[bits], bits)
        self.ssm_all = torch.zeros((layers, slots, self.layout.nbytes), dtype=torch.uint8, device=DEV)
        self.window = WindowPool(self.layout, self.ssm_all, self.layer_ids)
        self.conv_all = torch.stack([_pools(slots, seed=10 + j)[0] for j in range(layers)])

    def views(self, j=0):
        return self.window.layer[self.layer_ids[j]]

    def arg(self, j=0):
        return (self.layout, self.views(j))

    def seed(self, slots, seed=1, j=0):
        """Quantize random states into `slots` and return the states they now hold, FP32 [len(slots), H, P, N]."""
        from sharar.kernels import ssm_window as W
        t = torch.tensor(slots, dtype=torch.int32, device=DEV)
        W.quantize(self.layout, self.views(j), _pools(len(slots), seed=seed)[1], t)
        return self.state(slots, j)

    def state(self, slots, j=0):
        from sharar.kernels import ssm_window as W
        return torch.stack([W.reference_state(self.layout, self.views(j), s).float() for s in slots])


@pytest.mark.parametrize("bits", [8, 4])
def test_window_normalized_prefill_preserves_checkpoint_bytes(layer, bits, monkeypatch):
    from sharar.kernels import mamba2_prefill
    params, _ = layer
    pool = _WindowPool(1, 10, bits)
    pool.seed([3, 4, 5], seed=19)
    lengths, prefixes, slots = [2048, 3, 257], [0, 17, 2180], [3, 4, 5]
    plan = m.plan_prefill(lengths, prefixes, slots, DEV, [(0, 127, 6), (0, 128, 7), (2, 3, 8)])
    zx = _inputs(layer, sum(lengths), 927)
    conv0, bytes0 = pool.conv_all.clone(), pool.ssm_all.clone()
    expected = m.gated_rmsnorm(m.prefill(zx, params, pool.conv_all[0], pool.ssm_all[0], plan,
                                        window=pool.arg()), zx[:, :DI], params)
    ref_conv, ref_bytes = pool.conv_all.clone(), pool.ssm_all.clone()
    pool.conv_all.copy_(conv0)
    pool.ssm_all.copy_(bytes0)
    calls = []
    original = mamba2_prefill.scan_norm
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(mamba2_prefill, "scan_norm", counted)
    actual = m.prefill(zx, params, pool.conv_all[0], pool.ssm_all[0], plan,
                       window=pool.arg(), normalize=True)
    assert bool(calls) == (torch.cuda.get_device_capability() == (12, 1))
    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=4e-3)
    assert torch.equal(pool.conv_all, ref_conv)
    assert torch.equal(pool.ssm_all, ref_bytes)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        pool.conv_all.copy_(conv0)
        pool.ssm_all.copy_(bytes0)
        captured = m.prefill(zx, params, pool.conv_all[0], pool.ssm_all[0], plan,
                             window=pool.arg(), normalize=True)
    g.replay()
    torch.testing.assert_close(captured, expected, rtol=4e-3, atol=4e-3)
    assert torch.equal(pool.conv_all, ref_conv)
    assert torch.equal(pool.ssm_all, ref_bytes)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("logged", [0, 7, 12])  # 12: the larger counts fold
@pytest.mark.parametrize("batch,steps", [(1, 1), (3, 4), (2, 6)])
def test_window_verify_and_commit_follow_the_recurrence(layer, batch, steps, logged, bits):
    """From a quantized boundary state with `logged` decoded tokens in its window: verify's outputs against the FP32
    recurrence from the state the slot holds, the pool untouched, the ring's records the recurrence's own, and the
    commit of every count leaves the recurrence's state (exactly up to the FP16 records while the window has room,
    within one quantization when it folds)."""
    from sharar.kernels import ssm_window as W
    params, _ = layer
    pool = _WindowPool(1, 8, bits)
    slots_l = [5, 2, 7][:batch]
    slots = torch.tensor(slots_l, dtype=torch.int32, device=DEV)
    s0 = pool.seed(slots_l)
    conv0 = pool.conv_all[0].clone()
    zx = _inputs(layer, batch * (logged + steps), 40).view(batch, logged + steps, -1)
    for t in range(logged):
        m.decode(zx[:, t].contiguous(), params, pool.conv_all[0], pool.ssm_all[0], slots, window=pool.arg())
    assert int(pool.views()["log_len"][slots_l[0]].max()) == logged
    refs = [_reference(params, zx[i], conv0[s], s0[i]) for i, s in enumerate(slots_l)]
    for i, s in enumerate(slots_l):  # the decoded tokens' records are the recurrence's
        if logged:
            assert _rel(pool.state([s])[0], refs[i][3][logged - 1]) < 2e-3
    replay = W.WindowState(pool, 4, 8, G)
    before = pool.ssm_all.clone()
    block = zx[:, logged:].reshape(batch * steps, -1)
    y = m.verify(block, params, pool.conv_all[0], pool.ssm_all[0], slots, steps,
                 replay=replay.layer_inputs[0], window=pool.arg())
    assert torch.equal(pool.ssm_all, before)
    for i in range(batch):
        assert _rel(y.view(batch, steps, -1)[i], refs[i][0][logged:]) < 4e-3, i
    for count in range(steps + 1):
        counts = torch.tensor([(count + i) % (steps + 1) for i in range(batch)], dtype=torch.int32, device=DEV)
        pool.ssm_all.copy_(before)
        replay.run(pool, slots, counts)
        for i, (s, c) in enumerate(zip(slots_l, counts.tolist())):
            got = pool.state([s])[0]
            if not c:
                assert torch.equal(pool.ssm_all[0, s], before[0, s])
                continue
            want = refs[i][3][logged + c - 1]
            # records only (FP16 dt x): the window's own error; a fold adds one quantization of the state
            tol = 2e-3 if logged + c < WINDOW else {8: 3e-3, 4: 2e-2}[bits]
            assert _rel(got, want) < tol, (i, c)
            assert int(pool.views()["log_len"][s][0]) == (logged + c if logged + c < WINDOW else 0)
        others = [s for s in range(8) if s not in slots_l]
        assert torch.equal(pool.ssm_all[:, others], before[:, others])


@pytest.mark.parametrize("bits", [8, 4])
def test_window_decode_folds_and_tracks_the_recurrence(layer, bits):
    """60 decode steps (three folds of a 16-token window) against the FP32 recurrence from the seeded state: the
    quantization error enters once per fold and does not grow with the steps (the state decays)."""
    params, _ = layer
    pool = _WindowPool(1, 4, bits)
    slots_l = [1, 3]
    slots = torch.tensor(slots_l, dtype=torch.int32, device=DEV)
    s0 = pool.seed(slots_l, seed=4)
    conv0 = pool.conv_all[0].clone()
    steps = 60
    zx = _inputs(layer, 2 * steps, 41).view(2, steps, -1)
    ys = torch.stack([m.decode(zx[:, t].contiguous(), params, pool.conv_all[0], pool.ssm_all[0], slots,
                               window=pool.arg()) for t in range(steps)], 1)
    for i, s in enumerate(slots_l):
        y_ref, s_ref, h_ref, _ = _reference(params, zx[i], conv0[s], s0[i])
        assert torch.equal(pool.conv_all[0, s], h_ref)
        assert int(pool.views()["log_len"][s][0]) == steps % WINDOW
        assert _rel(pool.state([s])[0], s_ref) < {8: 2e-3, 4: 2e-2}[bits], i
        assert _rel(ys[i], y_ref) < {8: 4e-3, 4: 1e-2}[bits], i  # BF16 outputs: 4e-3 before any fold


@pytest.mark.parametrize("bits", [8, 4])
def test_window_prefill_quantizes_the_fp32_prefill(layer, bits):
    """The window prefill runs the FP32 prefill on the states the slots hold: outputs bitwise the FP32 pool's from those
    states, final and checkpoint states its states after one quantization; fresh slots ignore what they held."""
    params, _ = layer
    pool = _WindowPool(1, 12, bits)
    lens, prefixes, slots = [300, 1, 129, 133, 2], [0, 5, 0, 70, 3], [3, 4, 5, 6, 7]
    checkpoints = [(0, 128, 8), (0, 200, 9), (3, 58, 10), (3, 3, 11)]
    from sharar.kernels import ssm_window as W
    pool.seed(slots, seed=6)
    held = W.materialize(pool.layout, pool.views(), torch.tensor(slots, dtype=torch.int32, device=DEV))
    conv = pool.conv_all[0]
    conv_fp32, ssm_fp32 = conv.clone(), torch.zeros(12, H, P, N, device=DEV)
    for s, st, p in zip(slots, held, prefixes):
        ssm_fp32[s] = st if p else 0
    zx = _inputs(layer, sum(lens), 42)
    plan = m.plan_prefill(lens, prefixes, slots, DEV, checkpoints)
    assert plan.scratch_rows == len(lens) + len(checkpoints)
    y = m.prefill(zx, params, conv, pool.ssm_all[0], plan, window=pool.arg())
    y_fp32 = m.prefill(zx, params, conv_fp32, ssm_fp32, plan)
    assert torch.equal(y, y_fp32) and torch.equal(conv, conv_fp32)
    tol = {8: 5e-3, 4: 8e-2}[bits]  # one quantization of a prefilled state (measured 2.5e-3, 5.4e-2)
    for s in slots + [ck for _, _, ck in checkpoints]:
        assert _rel(pool.state([s])[0], ssm_fp32[s]) < tol, s
        assert int(pool.views()["log_len"][s].abs().sum()) == 0
    assert not pool.ssm_all[0, [0, 1, 2]].any()


def test_window_paths_under_graphs(layer):
    """verify + commit and decode on a quantized pool capture into CUDA graphs and replay as they ran eagerly."""
    from sharar.kernels import ssm_window as W
    params, _ = layer
    pool = _WindowPool(2, 6, 8)
    for j in range(2):
        pool.seed([1, 3], seed=7 + j, j=j)
    replay = W.WindowState(pool, 4, 5, G)
    windows = torch.zeros(2, 4, 5, C, 3, device=DEV, dtype=torch.bfloat16)
    slots = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    counts = torch.tensor([2, 5], dtype=torch.int32, device=DEV)
    zx = torch.stack([_inputs(layer, 10, 50 + j) for j in range(2)])
    zd = _inputs(layer, 2, 52)
    ssm0, conv0 = pool.ssm_all.clone(), pool.conv_all.clone()

    def body():
        out = [m.verify(zx[j], params, pool.conv_all[j], pool.ssm_all[j], slots, 5, windows[j],
                        replay=replay.layer_inputs[lid], window=pool.arg(j))
               for j, lid in enumerate(pool.layer_ids)]
        replay.run(pool, slots, counts)
        m.commit_conv(pool.conv_all, windows, slots, counts)
        out.append(m.decode(zd, params, pool.conv_all[0], pool.ssm_all[0], slots, window=pool.arg(0)))
        return out

    eager = body()
    ssm1, conv1 = pool.ssm_all.clone(), pool.conv_all.clone()
    pool.ssm_all.copy_(ssm0)
    pool.conv_all.copy_(conv0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = body()
    pool.ssm_all.copy_(ssm0)
    pool.conv_all.copy_(conv0)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))
    assert torch.equal(pool.ssm_all, ssm1) and torch.equal(pool.conv_all, conv1)
