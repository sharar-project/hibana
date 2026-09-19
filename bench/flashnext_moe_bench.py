# SPDX-License-Identifier: Apache-2.0
"""FlashInfer cutlass_fused_moe at decode shapes in the Sharar image: is the 158 us/call of the profile the kernel's speed?"""

import torch
from flashinfer.fused_moe import cutlass_fused_moe

from sharar.model.flashnext.moe import swizzle_block_scales_batched

E, I, K, TOPK = 512, 640, 2560, 10
dev = torch.device("cuda")
torch.manual_seed(0)
w13 = torch.randint(0, 256, (E, 2 * I, K // 2), device=dev, dtype=torch.uint8)
w2 = torch.randint(0, 256, (E, K, I // 2), device=dev, dtype=torch.uint8)
s13 = swizzle_block_scales_batched((torch.rand((E, 2 * I, K // 16), device=dev) * 2 + 0.5).to(torch.float8_e4m3fn))
s2 = swizzle_block_scales_batched((torch.rand((E, K, I // 16), device=dev) * 2 + 0.5).to(torch.float8_e4m3fn))
g1 = torch.full((E,), 1e-3, device=dev, dtype=torch.float32)
g2 = torch.full((E,), 1e-3, device=dev, dtype=torch.float32)
a13 = torch.tensor(1.0, device=dev, dtype=torch.float32)
a2 = torch.tensor(1.0, device=dev, dtype=torch.float32)
quant_scales = [a13, s13, g1, a2, s2, g2]
bytes_per_expert = 2 * I * (K // 2) + K * (I // 2) + 2 * I * (K // 16) + K * (I // 16)


def bench(M, tune, use_output, graphs=True, iters=40):
    x = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
    experts = [torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32) for _ in range(8)]
    scales = torch.softmax(torch.rand((M, TOPK), device=dev), dim=-1).to(torch.float32)
    out = torch.empty((M, K), device=dev, dtype=torch.bfloat16)
    kw = {"tune_max_num_tokens": tune} if tune else {}
    if use_output:
        kw["output"] = out

    def call(ids):
        return cutlass_fused_moe(input=x, token_selected_experts=ids, token_final_scales=scales,
                                 fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
                                 output_dtype=torch.bfloat16, quant_scales=quant_scales, **kw)
    for ids in experts:
        call(ids)
    torch.cuda.synchronize()
    if graphs:
        gs = []
        for ids in experts:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call(ids)
            gs.append(g)
        for g in gs:
            g.replay()
        torch.cuda.synchronize()
        run = lambda i: gs[i % 8].replay()  # noqa: E731
    else:
        run = lambda i: call(experts[i % 8])  # noqa: E731
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters):
        run(i)
    b.record(); torch.cuda.synchronize()
    ms = a.elapsed_time(b) / iters
    exp_distinct = E * (1 - (1 - TOPK / E) ** M)
    print(f"M={M:2d} tune={str(tune):5s} output={use_output!s:5s} graphs={graphs!s:5s}: {ms * 1000:7.0f} us per layer -> x48 = {ms * 48:5.1f} ms; {exp_distinct * bytes_per_expert / ms / 1e6:.0f} GB/s", flush=True)


for M in (1, 4):
    for tune in (None, 1 << (M - 1).bit_length(), 8192):
        for use_output in (True, False):
            bench(M, tune, use_output)
bench(1, 1, True, graphs=False)
print("done")
