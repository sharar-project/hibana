"""KV page size 1 versus 16: FlashInfer decode and verify attention over FP8 pages on the GB10.

The engine's pools use page size 1 (one token per page), so every plan()
uploads an index list as long as the context. This measures, in isolation,
what page 16 would change: the attention kernels' time at 10K context for the
16 full-attention layers, and the host cost of plan() with 16x fewer indices.
Pages are a random permutation, as a free-list allocator hands them out.

    python bench/kv_page_size.py --context 10000 --batch 1 4 --page-size 1 16
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import flashinfer

H_Q, H_KV, D = 24, 4, 256
LAYERS = 16
KV_DTYPE = torch.float8_e4m3fn
VERIFY_ROWS = 16


def make_pool(page_size: int, num_pages: int, device):
    shape = (num_pages, page_size, H_KV, D)
    return ([torch.randn(shape, device=device, dtype=torch.bfloat16).to(KV_DTYPE) for _ in range(LAYERS)],
            [torch.randn(shape, device=device, dtype=torch.bfloat16).to(KV_DTYPE) for _ in range(LAYERS)])


def layout(page_size: int, batch: int, context: int, generator):
    """Per-sequence page lists over a random free-list order; returns host indptr, indices, last_page_len."""
    per_seq = -(-context // page_size)
    order = torch.randperm(batch * per_seq + 1, generator=generator)[: batch * per_seq] + 1  # page 0 reserved
    indptr = torch.tensor([i * per_seq for i in range(batch + 1)], dtype=torch.int32)
    last = context - (per_seq - 1) * page_size
    return indptr, order.to(torch.int32), torch.full((batch,), last, dtype=torch.int32)


def time_ms(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run(page_size: int, batch: int, context: int, iters: int, device) -> dict:
    g = torch.Generator().manual_seed(0)
    per_seq = -(-(context + VERIFY_ROWS) // page_size)
    k, v = make_pool(page_size, batch * per_seq + 1, device)
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=device)
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
    indptr, indices, last = layout(page_size, batch, context, g)
    plan_dec = lambda: dec.plan(indptr, indices, last, H_Q, H_KV, D, page_size, pos_encoding_mode="NONE",  # noqa: E731
                                q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=D ** -0.5)
    t0 = time.perf_counter()
    for _ in range(iters):
        plan_dec()
    torch.cuda.synchronize()
    plan_decode_ms = (time.perf_counter() - t0) / iters * 1000
    q = torch.randn(batch, H_Q, D, device=device, dtype=torch.bfloat16)
    decode_ms = time_ms(lambda: [dec.run(q, (k[l], v[l])) for l in range(LAYERS)], iters)

    ws2 = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    ver = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws2, "NHD")
    vindptr, vindices, vlast = layout(page_size, batch, context + VERIFY_ROWS, g)
    qo = torch.arange(0, (batch + 1) * VERIFY_ROWS, VERIFY_ROWS, dtype=torch.int32)
    plan_ver = lambda: ver.plan(qo, vindptr, vindices, vlast, H_Q, H_KV, D, page_size, causal=True,  # noqa: E731
                                q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=D ** -0.5)
    t0 = time.perf_counter()
    for _ in range(iters):
        plan_ver()
    torch.cuda.synchronize()
    plan_verify_ms = (time.perf_counter() - t0) / iters * 1000
    qv = torch.randn(batch * VERIFY_ROWS, H_Q, D, device=device, dtype=torch.bfloat16)
    verify_ms = time_ms(lambda: [ver.run(qv, (k[l], v[l])) for l in range(LAYERS)], iters)
    bytes_per_layer = 2 * batch * context * H_KV * D
    return {"page_size": page_size, "batch": batch, "context": context,
            "decode_ms_16_layers": round(decode_ms, 3), "verify_ms_16_layers": round(verify_ms, 3),
            "plan_decode_ms": round(plan_decode_ms, 3), "plan_verify_ms": round(plan_verify_ms, 3),
            "indices": int(indices.numel()),
            "decode_gbps": round(bytes_per_layer * LAYERS / decode_ms / 1e6, 1),
            "verify_gbps": round(bytes_per_layer * LAYERS / verify_ms / 1e6, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--page-size", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    device = torch.device("cuda")
    rows = []
    for batch in args.batch:
        for page_size in args.page_size:
            row = run(page_size, batch, args.context, args.iters, device)
            rows.append(row)
            print(json.dumps(row), flush=True)
    for batch in args.batch:
        base = next(r for r in rows if r["batch"] == batch and r["page_size"] == args.page_size[0])
        for r in rows:
            if r["batch"] == batch and r is not base:
                print(f"batch {batch}: page {r['page_size']} vs {base['page_size']}: decode x{r['decode_ms_16_layers'] / base['decode_ms_16_layers']:.3f}, "
                      f"verify x{r['verify_ms_16_layers'] / base['verify_ms_16_layers']:.3f}, plan(decode) {base['plan_decode_ms']:.2f} -> {r['plan_decode_ms']:.2f} ms, "
                      f"plan(verify) {base['plan_verify_ms']:.2f} -> {r['plan_verify_ms']:.2f} ms")


if __name__ == "__main__":
    main()
