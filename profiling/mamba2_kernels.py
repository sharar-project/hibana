# SPDX-License-Identifier: Apache-2.0
"""Per-layer cost of the Nemotron-3.5-Super Mamba-2 mixer (`sharar/kernels/mamba2.py`) on layer 0's real weights.

    uv run python -m profiling.mamba2_kernels [--reps 9] [--no-flashinfer] [--json out.json]

Every row splits one layer's mixer into the FP8 in_proj (with the input quantization), the mixer core (convolution,
recurrence) and the gated RMS norm fused with the out projection's FP8 input quantization, then the FP8 out_proj;
"x40" is the 40 Mamba layers of one forward. Decode: B = 1, 4, 16, 64 rows. Verify: B = 1, 4, 16 rows of S = 2, 4,
5, 9 tokens, the core in both state strategies (compact replay inputs, or every step's state as snapshots), and the
accept step's replay of all S tokens of every row (one launch for all 40 layers, timed as such and divided by 40).
Prefill: one sequence of 2048 and of 8192 tokens, and four of 512.

Decode, verify and replay are timed as CUDA graphs of one call per layer, each layer with its own state slots and
inputs (rotated across as many copies as keep the working set out of L2), so the recurrent state streams from DRAM
as in a real forward; prefill is timed eagerly. Times are medians of --reps replays (CUDA events). The GPU may be
shared: `nvidia-smi` utilization is recorded before and after each section. FlashInfer's selective_state_update
(the CUDA SSU, which runs on SM121) is timed beside the decode and snapshot-verify recurrence for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharar.kernels import mamba2 as m  # noqa: E402

LAYERS = 40
STATE_BYTES = m.HEADS * m.HEAD_DIM * m.STATE * 4  # one slot's FP32 recurrent state, per layer


def utilization() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{out}%"
    except (OSError, subprocess.SubprocessError):
        return "?"


def shard_path() -> Path:
    from sharar.target import NEMOTRON
    try:
        snap = NEMOTRON.snapshot_dir()
    except FileNotFoundError:
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        snap = hub / NEMOTRON.model_cache_key / "snapshots" / NEMOTRON.model_revision
    return snap / "model-00001-of-00088.safetensors"


def load_layer0(device):
    from safetensors import safe_open
    from sharar.weights.modelopt import Fp8Linear
    p = "language_model.backbone.layers.0."
    with safe_open(str(shard_path()), "pt", device="cpu") as f:
        t = lambda name: f.get_tensor(p + name).to(device)  # noqa: E731
        params = m.prepare_params(t("mixer.conv1d.weight"), t("mixer.conv1d.bias"), t("mixer.A_log"), t("mixer.D"),
                                  t("mixer.dt_bias"), t("mixer.norm.weight"), 1e-5)
        lin = {name: Fp8Linear(t(f"mixer.{name}.weight").contiguous(),
                               t(f"mixer.{name}.weight_scale").float().reshape(()),
                               t(f"mixer.{name}.input_scale").float().reshape(()))
               for name in ("in_proj", "out_proj")}
        prenorm = t("norm.weight")
    return params, lin, prenorm


class Timer:
    def __init__(self, reps: int):
        self.reps = reps

    def events(self, fn) -> float:
        ts = []
        for _ in range(self.reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return statistics.median(ts)

    def graph(self, fn, calls: int) -> float:
        """ms per call of `fn` (which makes `calls` calls), captured once and replayed."""
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        return self.events(g.replay) / calls


def copies_for(nbytes: int, cap: int = LAYERS) -> int:
    """Copies whose combined size (>= 192 MB) keeps a rotated working set out of the 24 MB L2."""
    return max(2, min(cap, -(-192 * 2**20 // max(nbytes, 1))))


def hidden(rows, prenorm, seed=0):
    g = torch.Generator(device=prenorm.device).manual_seed(seed)
    h = torch.randn(rows, prenorm.numel(), device=prenorm.device, generator=g)
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16() * prenorm


def projections(timer, lin, params, prenorm, rows: int) -> tuple[float, float, float, torch.Tensor]:
    """(in_proj incl. quantization, gated norm + quant, out_proj) ms per layer at `rows` rows, and a zxbcdt."""
    from sharar.kernels import ops
    from sharar.kernels.fp8_quant import quant_fp8
    x = hidden(rows, prenorm)
    w_in, w_out = lin["in_proj"], lin["out_proj"]
    zx = ops.fp8_linear(None, w_in, x_q=quant_fp8(x, w_in.input_scale))
    y = torch.randn(rows, m.D_INNER, device=x.device).bfloat16()
    xq = m.gated_rmsnorm(y, zx[:, :m.D_INNER], params, w_out.input_scale)
    graphs = rows <= 4096
    run = (lambda f: timer.graph(f, 1)) if graphs else timer.events
    t_in = run(lambda: ops.fp8_linear(None, w_in, x_q=quant_fp8(x, w_in.input_scale)))
    t_norm = run(lambda: m.gated_rmsnorm(y, zx[:, :m.D_INNER], params, w_out.input_scale))
    t_out = run(lambda: ops.fp8_linear(None, w_out, x_q=xq))
    return t_in, t_norm, t_out, zx


def fi_args(params, conv, rows, steps=None):
    """FlashInfer SSU views of a conv output (x, B, C) and dt with tied head dims."""
    h, p, n, g = params.heads, params.head_dim, params.state, params.groups
    lead = (rows,) if steps is None else (rows, steps)
    conv = conv.view(*lead, -1)
    x = conv[..., :m.D_INNER].unflatten(-1, (h, p))
    b = conv[..., m.D_INNER:m.D_INNER + g * n].unflatten(-1, (g, n))
    c = conv[..., m.D_INNER + g * n:].unflatten(-1, (g, n))
    dt = torch.randn(*lead, h, device=conv.device)[..., None].expand(*lead, h, p)
    A = params.A[:, None, None].expand(h, p, n)
    return x, dt, A, b, c, params.D[:, None].expand(h, p), params.dt_bias[:, None].expand(h, p)


def decode_section(timer, params, lin, prenorm, flashinfer: bool) -> list[dict]:
    dev = prenorm.device
    rows_out = []
    for b in (1, 4, 16, 64):
        t_in, t_norm, t_out, zx = projections(timer, lin, params, prenorm, b)
        k = copies_for(b * STATE_BYTES)
        convs = [torch.zeros(b + 1, m.CONV_DIM, 3, device=dev, dtype=torch.bfloat16) for _ in range(k)]
        ssms = [torch.zeros(b + 1, m.HEADS, m.HEAD_DIM, m.STATE, device=dev) for _ in range(k)]
        slots = torch.arange(1, b + 1, dtype=torch.int32, device=dev)
        t_core = timer.graph(lambda: [m.decode(zx, params, convs[i], ssms[i], slots) for i in range(k)], k)
        row = dict(B=b, in_proj=t_in, core=t_core, norm=t_norm, out_proj=t_out)
        row["state_GBps"] = b * 2 * STATE_BYTES / t_core / 1e6
        if flashinfer:
            from flashinfer.mamba import selective_state_update
            conv = torch.randn(b, m.CONV_DIM, device=dev).bfloat16()
            x, dt, A, bm, cm, D, dtb = fi_args(params, conv, b)
            row["flashinfer_ssu"] = timer.graph(lambda: [selective_state_update(
                ssms[i], x, dt, A, bm, cm, D, dt_bias=dtb, dt_softplus=True, state_batch_indices=slots)
                for i in range(k)], k)
        rows_out.append(row)
    return rows_out


def verify_section(timer, params, lin, prenorm, flashinfer: bool) -> list[dict]:
    dev = prenorm.device
    out = []

    class Pool:  # the replay's view of a state pool: k layers of b + 1 slots
        pass

    for b in (1, 4, 16):
        for s in (2, 4, 5, 9):
            t_in, t_norm, t_out, zx = projections(timer, lin, params, prenorm, b * s)
            k = copies_for(b * STATE_BYTES)
            pool = Pool()
            pool.layer_ids = list(range(k))
            pool.ssm_all = torch.zeros(k, b + 1, m.HEADS, m.HEAD_DIM, m.STATE, device=dev)
            pool.conv_all = torch.zeros(k, b + 1, m.CONV_DIM, 3, device=dev, dtype=torch.bfloat16)
            replay = m.AcceptedState(pool, b, s, m.GROUPS)
            windows = torch.zeros(k, b, s, m.CONV_DIM, 3, device=dev, dtype=torch.bfloat16)
            slots = torch.arange(1, b + 1, dtype=torch.int32, device=dev)
            rows = torch.arange(b, dtype=torch.int32, device=dev)
            counts = torch.full((b,), s, dtype=torch.int32, device=dev)
            t_rec = timer.graph(lambda: [m.verify(zx, params, pool.conv_all[i], pool.ssm_all[i], slots, s, windows[i],
                                                  replay=replay.layer_inputs[i]) for i in range(k)], k)
            snaps = [torch.zeros(b, s, m.HEADS, m.HEAD_DIM, m.STATE, device=dev) for _ in range(min(k, 4))]
            t_snap = timer.graph(lambda: [m.verify(zx, params, pool.conv_all[i], pool.ssm_all[i], slots, s, windows[i],
                                                   snapshots=snaps[i % len(snaps)], snapshot_indices=rows)
                                          for i in range(k)], k)
            # The replay is one launch for every layer: time it over the k layers and scale to one.
            t_replay = timer.graph(lambda: replay.run(pool, slots, counts), k)
            t_commit = timer.graph(lambda: m.commit_conv(pool.conv_all, windows, slots, counts), k)
            row = dict(B=b, S=s, in_proj=t_in, core_reconstruct=t_rec, core_snapshots=t_snap, norm=t_norm,
                       out_proj=t_out, replay_all_S=t_replay, conv_commit=t_commit)
            if flashinfer:
                from flashinfer.mamba import selective_state_update
                conv = torch.randn(b * s, m.CONV_DIM, device=dev).bfloat16()
                x, dt, A, bm, cm, D, dtb = fi_args(params, conv, b, s)
                row["flashinfer_mtp_snapshots"] = timer.graph(lambda: [selective_state_update(
                    pool.ssm_all[i], x, dt, A, bm, cm, D, dt_bias=dtb, dt_softplus=True, state_batch_indices=slots,
                    disable_state_update=True, intermediate_states_buffer=snaps[i % len(snaps)],
                    intermediate_state_indices=rows, cache_steps=s) for i in range(k)], k)
                row["flashinfer_mtp_no_update"] = timer.graph(lambda: [selective_state_update(
                    pool.ssm_all[i], x, dt, A, bm, cm, D, dt_bias=dtb, dt_softplus=True, state_batch_indices=slots,
                    disable_state_update=True, cache_steps=s) for i in range(k)], k)
            out.append(row)
            del pool, replay, windows, snaps
    return out


def prefill_section(timer, params, lin, prenorm) -> list[dict]:
    dev = prenorm.device
    out = []
    for lens in ([2048], [8192], [512] * 4):
        t = sum(lens)
        t_in, t_norm, t_out, zx = projections(timer, lin, params, prenorm, t)
        conv = torch.zeros(len(lens) + 1, m.CONV_DIM, 3, device=dev, dtype=torch.bfloat16)
        ssm = torch.zeros(len(lens) + 1, m.HEADS, m.HEAD_DIM, m.STATE, device=dev)
        plan = m.plan_prefill(lens, [0] * len(lens), list(range(1, len(lens) + 1)), dev)
        m.prefill(zx, params, conv, ssm, plan)
        t_core = timer.events(lambda: m.prefill(zx, params, conv, ssm, plan))
        out.append(dict(seqs="x".join(map(str, lens)) if len(set(lens)) > 1 else f"{len(lens)}x{lens[0]}", tokens=t,
                        in_proj=t_in, core=t_core, norm=t_norm, out_proj=t_out))
        del zx
    return out


def total(row, core_key="core"):
    return row["in_proj"] + row[core_key] + row["norm"] + row["out_proj"]


def show(decode, verify, prefill, util):
    print(f"\nSharar Mamba-2 mixer, layer 0 weights, ms per layer (x40 = per forward). GPU utilization {util}")
    print("\n[Sharar] decode")
    print(f"{'B':>3} {'in_proj':>8} {'core':>8} {'norm+q':>8} {'out_proj':>8} {'layer':>8} {'x40':>8} {'state GB/s':>10}"
          f" {'FI ssu':>8}")
    for r in decode:
        print(f"{r['B']:>3} {r['in_proj']:8.4f} {r['core']:8.4f} {r['norm']:8.4f} {r['out_proj']:8.4f} {total(r):8.4f}"
              f" {40 * total(r):8.3f} {r['state_GBps']:10.0f} {r.get('flashinfer_ssu', float('nan')):8.4f}")
    print("\n[Sharar] verify (core = conv + recurrence; replay = the accept step advancing every row by all S tokens)")
    print(f"{'B':>3} {'S':>2} {'in_proj':>8} {'core rec':>8} {'core snp':>8} {'norm+q':>8} {'out_proj':>8} {'layer':>8}"
          f" {'x40':>8} {'replay':>8} {'commit':>7} {'FI snap':>8} {'FI noupd':>8}")
    for r in verify:
        layer = total(r, "core_reconstruct")
        print(f"{r['B']:>3} {r['S']:>2} {r['in_proj']:8.4f} {r['core_reconstruct']:8.4f} {r['core_snapshots']:8.4f}"
              f" {r['norm']:8.4f} {r['out_proj']:8.4f} {layer:8.4f} {40 * layer:8.3f} {r['replay_all_S']:8.4f}"
              f" {r['conv_commit']:7.4f} {r.get('flashinfer_mtp_snapshots', float('nan')):8.4f}"
              f" {r.get('flashinfer_mtp_no_update', float('nan')):8.4f}")
    print("\n[Sharar] prefill")
    print(f"{'seqs':>8} {'in_proj':>8} {'core':>8} {'norm+q':>8} {'out_proj':>8} {'layer':>8} {'x40':>8} {'us/token':>8}")
    for r in prefill:
        print(f"{r['seqs']:>8} {r['in_proj']:8.3f} {r['core']:8.3f} {r['norm']:8.3f} {r['out_proj']:8.3f}"
              f" {total(r):8.3f} {40 * total(r):8.1f} {1000 * 40 * total(r) / r['tokens']:8.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--no-flashinfer", action="store_true")
    ap.add_argument("--sections", default="decode,verify,prefill")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    timer = Timer(args.reps)
    with torch.inference_mode():
        params, lin, prenorm = load_layer0(dev)
        util, results = {}, {"decode": [], "verify": [], "prefill": []}
        for name in args.sections.split(","):
            before = utilization()
            if name == "decode":
                results[name] = decode_section(timer, params, lin, prenorm, not args.no_flashinfer)
            elif name == "verify":
                results[name] = verify_section(timer, params, lin, prenorm, not args.no_flashinfer)
            elif name == "prefill":
                results[name] = prefill_section(timer, params, lin, prenorm)
            util[name] = f"{before} -> {utilization()}"
            torch.cuda.empty_cache()
    show(results["decode"], results["verify"], results["prefill"], util)
    if args.json:
        args.json.write_text(json.dumps(dict(results, utilization=util), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
