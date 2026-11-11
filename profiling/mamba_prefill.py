# SPDX-License-Identifier: Apache-2.0
"""Compare separate and fused Mamba prefill cores through their public entry point.

Example: python -m profiling.mamba_prefill --snapshot /cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f
Use --random without weights. GPU timing reuses CUDA graphs; direct-call wall
timing also includes allocations and Python dispatch. Projection GEMMs and plan
construction are excluded. Run with exclusive GPU access.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from hibana.kernels import mamba2 as m
from hibana.launch.guard import start_thread


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    weights = ap.add_mutually_exclusive_group(required=True)
    weights.add_argument("--snapshot", type=Path)
    weights.add_argument("--random", action="store_true")
    ap.add_argument("--tokens", nargs="+", type=int, default=[2048])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--prefixes", nargs="+", type=int, default=[0, 2048])
    ap.add_argument("--state-format", choices=["fp32", "fp16_sr", "int8", "int4"], default="fp32")
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--replays", type=int, default=100)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if min(args.tokens + args.batches + [args.trials, args.replays]) < 1 or min(args.prefixes) < 0:
        ap.error("tokens, batches, trials and replays must be positive; prefixes nonnegative")
    start_thread()
    torch.manual_seed(1271)
    if args.snapshot:
        from hibana.weights.safetensors import SnapshotWeights
        w = SnapshotWeights(args.snapshot)
        names = ("conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias", "norm.weight")
        params = m.prepare_params(*(w.get("language_model.backbone.layers.0.mixer." + n).to("cuda")
                                    for n in names), 1e-5)
        w.close()
    else:
        params = m.prepare_params(torch.randn(m.CONV_DIM, 4, device="cuda") * .5,
                                  torch.randn(m.CONV_DIM, device="cuda") * .1,
                                  torch.log(torch.rand(m.HEADS, device="cuda") * 15 + 1),
                                  torch.randn(m.HEADS, device="cuda"),
                                  torch.rand(m.HEADS, device="cuda") * -4,
                                  torch.randn(m.D_INNER, device="cuda") * .2 + 1, 1e-5)
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "snapshot": str(args.snapshot) if args.snapshot else None, "state_format": args.state_format,
              "scope": "prefill core and norm, excluding projection GEMMs and plan preparation",
              "cases": []}

    def save():
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    with torch.inference_mode():
        for n in args.tokens:
            for batch in args.batches:
                for prefix in args.prefixes:
                    lengths = [n] * batch
                    checkpoints = [(i, min(128, n - 1), batch + i) for i in range(batch) if n > 128]
                    plan = m.plan_prefill(lengths, [prefix] * batch, list(range(batch)), "cuda", checkpoints)
                    zx = (torch.randn(n * batch, m.PROJ_DIM, device="cuda") * .4).bfloat16()
                    conv = torch.randn(batch * 2, m.CONV_DIM, 3, device="cuda", dtype=torch.bfloat16) * .1
                    state = torch.randn(batch * 2, m.HEADS, m.HEAD_DIM, m.STATE, device="cuda") * .02
                    window = None
                    if args.state_format == "fp16_sr":
                        state = state.half()
                    elif args.state_format in ("int8", "int4"):
                        from hibana.cache.ssm_format import WindowLayout, WindowPool
                        from hibana.kernels.ssm_window import quantize
                        bits = 8 if args.state_format == "int8" else 4
                        layout = WindowLayout(m.GROUPS, m.HEADS, m.STATE, m.HEAD_DIM, 16,
                                              4 if bits == 8 else 8, bits)
                        packed = torch.zeros((1, batch * 2, layout.nbytes), dtype=torch.uint8, device="cuda")
                        views = WindowPool(layout, packed, [0]).layer[0]
                        quantize(layout, views, state, torch.arange(batch * 2, device="cuda", dtype=torch.int32))
                        state = packed[0]
                        window = (layout, views)
                    conv0, state0 = conv.clone(), state.clone()

                    def call(optimized):
                        y = m.prefill(zx, params, conv, state, plan, normalize=optimized, window=window)
                        return y if optimized else m.gated_rmsnorm(y, zx[:, :m.D_INNER], params)

                    expected = call(False)
                    ref_conv, ref_state = conv.clone(), state.clone()
                    conv.copy_(conv0)
                    state.copy_(state0)
                    actual = call(True)
                    diff = actual.float() - expected.float()
                    row = {"tokens_per_sequence": n, "batch": batch, "prefix": prefix,
                           "max_abs": float(diff.abs().max()),
                           "relative_l2": float(diff.norm() / expected.float().norm().clamp_min(1e-20)),
                           "output_bitwise": torch.equal(actual, expected),
                           "state_bitwise": torch.equal(state, ref_state),
                           "history_bitwise": torch.equal(conv, ref_conv)}
                    assert row["state_bitwise"] and row["history_bitwise"], row
                    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=4e-3)
                    graphs = []
                    for optimized in [False, True]:
                        call(optimized)
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            call(optimized)
                        graphs.append(g)
                    for _ in range(30):
                        for g in graphs:
                            g.replay()

                    def gpu(g):
                        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        a.record()
                        for _ in range(args.replays):
                            g.replay()
                        b.record()
                        b.synchronize()
                        return a.elapsed_time(b) * 1000 / args.replays

                    def wall(optimized):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(10):
                            call(optimized)
                        torch.cuda.synchronize()
                        return (time.perf_counter() - start) * 1e6 / 10

                    gpu_pairs, wall_pairs = [], []
                    for trial in range(args.trials):
                        order = [0, 1] if trial % 2 == 0 else [1, 0]
                        times = {i: gpu(graphs[i]) for i in order}
                        gpu_pairs.append({"baseline_us": times[0], "optimized_us": times[1]})
                        times = {i: wall(bool(i)) for i in order}
                        wall_pairs.append({"baseline_us": times[0], "optimized_us": times[1]})
                    row.update(gpu_trials=gpu_pairs, wall_trials=wall_pairs,
                               gpu_speedup=statistics.median(x["baseline_us"] / x["optimized_us"] for x in gpu_pairs),
                               wall_speedup=statistics.median(x["baseline_us"] / x["optimized_us"] for x in wall_pairs))
                    report["cases"].append(row)
                    save()
                    print(json.dumps({k: v for k, v in row.items() if not k.endswith("trials")}), flush=True)
                    del graphs, expected, actual, conv, state, conv0, state0, ref_conv, ref_state
                    torch.cuda.empty_cache()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
