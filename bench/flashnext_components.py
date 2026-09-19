# SPDX-License-Identifier: Apache-2.0
"""Flash-Next engine modules against the validated plain-torch reference, one layer each.

Loads only what each check needs (layer 0's experts, GDN and PLE; layer 3's attention), runs
random and embedding-derived inputs through both implementations and prints cosine similarity
and relative error. MoE variants (gate/up order) are tried when the primary layout disagrees.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from sharar.model.flashnext import reference as R
from sharar.model.flashnext.attention import FlashNextAttention
from sharar.model.flashnext.gdn import FlashNextGdn
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.flashnext.moe import MoeLayer
from sharar.model.flashnext.ple import PleLayer
from sharar.model.rope import PartialRotary
from sharar.kernels import ops
from sharar.target import FLASHNEXT
from sharar.weights.safetensors import SnapshotWeights

SNAP_ROOT = Path(os.environ.get("FN_SNAP_ROOT", "/cache/flash-next/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots"))
P = R.PREFIX


def stats(name, a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    cos = F.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    print(f"  {name:34s} cos {cos:.6f}  rel {rel:.4f}  |ref| {b.norm().item():.3f}", flush=True)
    return cos


def main():
    torch.manual_seed(0)
    snap = sorted(SNAP_ROOT.iterdir())[-1]
    d = torch.device("cuda")
    w = SnapshotWeights(snap)
    cfg = R.FlashNextConfig(snap)
    t = FLASHNEXT
    ref = R.FlashNextReference.__new__(R.FlashNextReference)
    ref.cfg, ref.device, ref.w, ref.snap, ref.ablate = cfg, d, w, snap, frozenset()
    T = 40
    ids = torch.randint(1000, 200000, (T,), dtype=torch.int64)
    embed = w.get(f"{P}.embed_tokens.weight")
    x_small = F.embedding(ids, embed).to(d)  # embedding-scale input (std ~0.01)
    x_unit = torch.randn(T, cfg.hidden, device=d, dtype=torch.bfloat16)
    inputs = [("embed-scale", x_small), ("unit", x_unit)]

    # ---- MoE, layer 0 ----
    print("MoE layer 0", flush=True)
    p = f"{P}.layers.0"
    bank = R.ExpertBank(w, f"{p}.mlp.experts", cfg.experts, d)
    L = {"router": w.get(f"{p}.mlp.gate.weight").to(d), "shared_gate": w.get(f"{p}.mlp.shared_expert_gate.weight").to(d),
         "shared": {n: w.get(f"{p}.mlp.shared_expert.{n}.weight").to(d) for n in ("gate_proj", "up_proj", "down_proj")}, "experts": bank}
    eng = MoeLayer(w, f"{p}.mlp", d, t, TrunkLinear, "bf16")
    for name, x in inputs:
        ref_out = ref._moe(x, L)
        stats(f"moe {name}", eng(x), ref_out)
        # routed part only (shared expert removed on both sides) and the gate/up-swapped layout
        ref.ablate = frozenset({"shared"})
        routed_ref = ref._moe(x, L)
        ref.ablate = frozenset()
        gu = eng.shared_gate_up(x)
        shared = eng.shared_down(F.silu(gu[:, : eng.inter]) * gu[:, eng.inter:]).float() * torch.sigmoid(x.float() @ eng.shared_gate.T)
        routed_eng = eng(x).float() - shared
        c = stats(f"moe routed {name}", routed_eng, routed_ref)
        if c < 0.99:
            i = eng.inter
            swapped = MoeLayer.__new__(MoeLayer); swapped.__dict__.update(eng.__dict__)
            swapped.w13 = torch.cat([eng.w13[:, i:], eng.w13[:, :i]], dim=1).contiguous(); swapped.w13_long = swapped.w13.view(torch.long)
            s13 = eng.s13.view(torch.float8_e4m3fn)  # swizzled: recompute from raw instead
            from sharar.model.flashnext.moe import swizzle_block_scales_batched
            raw = torch.stack([torch.cat([w.get(f"{p}.mlp.experts.{e}.up_proj.weight_scale"), w.get(f"{p}.mlp.experts.{e}.gate_proj.weight_scale")], 0) for e in range(cfg.experts)]).to(d)
            swapped.s13 = swizzle_block_scales_batched(raw); swapped.quant_scales = [eng.a13, swapped.s13, eng.g1, eng.a2, eng.s2, eng.g2]
            routed_sw = swapped(x).float() - shared
            stats(f"moe routed swapped {name}", routed_sw, routed_ref)
    del bank, eng, L
    torch.cuda.empty_cache()

    # ---- GDN, layer 0 ----
    print("GDN layer 0", flush=True)
    q = f"{p}.linear_attn"
    L = {n: w.get(f"{q}.{n}").to(d) for n in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight")}
    eng = FlashNextGdn(w, q, 0, d, t, "bf16")
    for name, x in inputs:
        stats(f"gdn {name}", eng.prefill(x), ref._gdn(x, L))
    del eng, L

    # ---- attention, layer 3 ----
    print("attention layer 3", flush=True)
    p3 = f"{P}.layers.3.self_attn"
    L = {n: w.get(f"{p3}.{n}").to(d) for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight")}
    rope = PartialRotary(t.head_dim, t.rotary_dim, t.rope_theta, 4096, d)
    ragged = ops.RaggedAttention(d, t.num_attention_heads, t.num_kv_heads, t.head_dim)
    eng = FlashNextAttention(w, p3, 3, d, rope, ragged, t, "bf16")
    positions = torch.arange(T, device=d)
    for name, x in inputs:
        stats(f"attention {name}", eng.prefill(x, positions, None), ref._attention(x, positions, L))
    del eng, L

    # ---- PLE, layer 1 ----
    print("PLE layer 1", flush=True)
    ref._ple_handles = {}
    Lp = ref._load_ple(f"{P}.layers.1.ple")
    eng = PleLayer(w, snap, f"{P}.layers.1.ple", d, t, TrunkLinear, "bf16")
    for name, x in inputs:
        hyper = torch.cat([x] * cfg.hc, dim=-1)
        stats(f"ple {name}", eng.forward(hyper, ids.to(d), None, cfg.eos), ref._ple(hyper, ids, Lp))
    print("done", flush=True)


if __name__ == "__main__":
    main()
