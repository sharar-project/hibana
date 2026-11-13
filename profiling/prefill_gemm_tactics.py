"""Every CUTLASS FP4 tactic for the 27B's MLP GEMMs at prefill row counts: equal to the default output? how fast?

The input to `hibana/kernels/prefill_gemm.py`'s table. Synthetic NVFP4 operands with the real shapes; each tactic's
output is compared bitwise with FlashInfer's default dispatch, and its time is the mean of 20 calls after 3.

    uv run python -m profiling.prefill_gemm_tactics [--rows 512,1024,2048,4096]
"""

import argparse

import torch


def _timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def main():
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf, get_gemm_sm120_module_cutlass_fp4

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="256,512,1024,1536,2048,3072,4096")
    rows = [int(r) for r in ap.parse_args().rows.split(",")]
    torch.manual_seed(0)
    dev = torch.device("cuda")
    runner = get_gemm_sm120_module_cutlass_fp4().cutlass_fp4_gemm_runner()
    workspace = _get_cache_buf("mm_fp4_workspace", DEFAULT_WORKSPACE_SIZE, dev)

    def quantize(x):
        scale = (448 * 6 / x.abs().amax().float()).reshape(1)
        q, sf = nvfp4_quantize(x, scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        return q, sf, scale

    for n, k in ((34816, 5120), (5120, 17408)):
        wq, wsf, ws = quantize(torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02)
        for m in rows:
            xq, xsf, xs = quantize(torch.randn(m, k, device=dev, dtype=torch.bfloat16))
            alpha = (1.0 / (xs * ws)).float()
            default = lambda: mm_fp4(xq, wq.T, xsf, wsf.T, alpha, torch.bfloat16, backend="cutlass")  # noqa: E731
            reference, base = default(), _timed(default)
            results = []
            for tactic in runner.get_valid_tactics([], None):
                out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
                call = lambda t=tactic, o=out: runner.forward(  # noqa: E731
                    [xq, wq.T, xsf, wsf.T, alpha, None, o, None, None, workspace], tactic=t)
                call()
                results.append((_timed(call), tactic, torch.equal(out.view(torch.int16), reference.view(torch.int16))))
            results.sort()
            equal = [r for r in results if r[2]]
            best = equal[0] if equal else None
            print(f"N={n} K={k} M={m}: default {base:.3f} ms; {len(equal)}/{len(results)} tactics equal; "
                  + (f"fastest equal: tactic {best[1]} {best[0]:.3f} ms ({base / best[0]:.2f}x)" if best else "none equal"),
                  flush=True)


if __name__ == "__main__":
    main()
