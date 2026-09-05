"""Block-level parity: feed each block the reference's own dumped input.

From a dump, layer i yields (h_i, r_i) with r_i = stream before the MLP add and
h_i = MLP output, so:
  block_in_i  = h_{i-1} + r_{i-1}        (embed for i = 0)
  mixer_out_i = r_i - block_in_i
  mlp_in_i    = post_norm(r_i), mlp_out_i = h_i
This isolates the GDN block, the attention block and the MLP with no error
accumulation, using only two loaded layers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from sharar import config
from sharar.kernels import ops
from sharar.kernels.conv1d import causal_conv1d_fn
from sharar.kernels.fla.fused_recurrent import fused_recurrent_gated_delta_rule
from sharar.kernels.fla.chunk import chunk_gated_delta_rule
from sharar.kernels.fla.fused_gdn_gating import fused_gdn_gating
from sharar.model.qwen38 import PREFIX, DecoderLayer
from sharar.model.rope import PartialRotary
from sharar.weights.modelopt import load_fp8_linear, load_nvfp4_linear
from sharar.weights.safetensors import SnapshotWeights

FP4_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def rel(mine: torch.Tensor, ref: torch.Tensor) -> str:
    mine = mine.float()
    ref = ref.float().to(mine.device)
    d = (mine - ref).abs()
    s = ref.abs().mean().item() or 1.0
    return f"max {d.max().item() / s:.4f} mean {d.mean().item() / s:.5f} (ref scale {s:.4f})"


def dequant_nvfp4(lin, device) -> torch.Tensor:
    w = lin.weight  # [N, K/2] uint8
    lo = (w & 0xF).long()
    hi = (w >> 4).long()
    vals = FP4_VALUES.to(device)
    full = torch.stack([vals[lo], vals[hi]], dim=-1).reshape(w.shape[0], -1)  # element 2i = low nibble
    # un-swizzle is hard; reload raw scales from the loader's input instead
    return full


def main(reference: str, prompt: str) -> int:
    device = torch.device("cuda")
    snap = config.snapshot_dir()
    w = SnapshotWeights(snap)
    ref = torch.load(Path(reference) / "dump" / f"{prompt}.pt", map_location="cpu")
    ids = ref["input_ids"].to(device)
    t = ids.shape[0]
    print(f"prompt {prompt}: T={t}")
    embed = w.get(f"{PREFIX}.embed_tokens.weight")
    x0 = torch.nn.functional.embedding(ids.cpu(), embed).to(device)
    print("embed:", rel(x0, ref["embed"]))

    rope = PartialRotary(config.HEAD_DIM, 64, config.ROPE_THETA, 4096, device)
    ragged = ops.RaggedAttention(device, config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM)
    l0 = DecoderLayer(w, 0, "linear_attention", device, rope, ragged, fused_silu_quant=True)
    l3 = DecoderLayer(w, 3, "full_attention", device, rope, ragged, fused_silu_quant=True)
    positions = torch.arange(t, device=device)

    # ---- layer 0: GDN block ----
    h0, r0 = ref["layers"][0]
    gdn_ref = r0.float() - ref["embed"].float()
    normed = l0.input_norm(x0)
    with torch.inference_mode():
        gdn_mine = l0.mixer.prefill(normed)
    print("layer0 GDN block:", rel(gdn_mine, gdn_ref))

    # sub-steps of GDN with torch references
    g = l0.mixer
    with torch.inference_mode():
        qkvz = ops.fp8_linear(normed, g.in_qkvz)
        wq = g.in_qkvz.weight.float() * g.in_qkvz.weight_scale
        xq = (normed.float() / g.in_qkvz.input_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * g.in_qkvz.input_scale
        print("  fp8 in_qkvz vs dequant matmul:", rel(qkvz, xq @ wq.T))
        mixed = qkvz[:, : g.conv_dim].contiguous()
        # torch causal depthwise conv + silu
        xpad = torch.nn.functional.pad(mixed.float().T.unsqueeze(0), (config.LINEAR_CONV_KERNEL - 1, 0))
        conv_ref = torch.nn.functional.conv1d(xpad, g.conv_weight.float().unsqueeze(1), groups=g.conv_dim)[0]
        conv_ref = torch.nn.functional.silu(conv_ref).T  # [T, dim]
        qsl = torch.tensor([0, t], dtype=torch.int32, device=device)
        ci = torch.tensor([0], dtype=torch.int32, device=device)
        conv_state = torch.zeros((1, g.conv_dim, 3), dtype=torch.bfloat16, device=device)
        conv_k = causal_conv1d_fn(mixed.transpose(0, 1), g.conv_weight, None, activation="silu", conv_states=conv_state, has_initial_state=torch.tensor([False], device=device), cache_indices=ci, query_start_loc=qsl, seq_lens_cpu=[t]).transpose(0, 1)[:t]
        print("  conv1d kernel vs torch:", rel(conv_k, conv_ref))
        conv_state2 = torch.zeros((1, g.conv_dim, 3), dtype=torch.bfloat16, device=device)
        conv_k2 = causal_conv1d_fn(qkvz[:, : g.conv_dim].transpose(0, 1), g.conv_weight, None, activation="silu", conv_states=conv_state2, has_initial_state=torch.tensor([False], device=device), cache_indices=ci, query_start_loc=qsl, seq_lens_cpu=[t]).transpose(0, 1)[:t]
        print("  conv1d kernel on sliced (non-contiguous) input vs torch:", rel(conv_k2, conv_ref))
        ba = normed @ g.in_ba.T
        b, a = ba.split([48, 48], dim=-1)
        gg, beta = fused_gdn_gating(g.A_log, a.contiguous(), b.contiguous(), g.dt_bias)
        g_ref = -g.A_log.exp() * torch.nn.functional.softplus(a.float() + g.dt_bias.float())
        print("  gating g:", rel(gg[0], g_ref), " beta:", rel(beta[0], torch.sigmoid(b.float())))
        q, k, v = torch.split(conv_ref.to(torch.bfloat16), [g.key_dim, g.key_dim, g.value_dim], dim=-1)
        q = q.reshape(1, t, 16, 128).contiguous()
        k = k.reshape(1, t, 16, 128).contiguous()
        v = v.reshape(1, t, 48, 128).contiguous()
        ssm = torch.zeros((1, 48, 128, 128), dtype=torch.float32, device=device)
        o_chunk, _, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=gg, beta=beta, initial_state=ssm, initial_state_indices=ci, cu_seqlens=qsl, head_first=False, use_qk_l2norm_in_kernel=True)
        ssm2 = torch.zeros((1, 48, 128, 128), dtype=torch.float32, device=device)
        o_rec, _ = fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=gg, beta=beta, initial_state=ssm2, output_final_state=False, cu_seqlens=qsl, use_qk_l2norm_in_kernel=True)
        print("  chunk vs fused_recurrent:", rel(o_chunk, o_rec))
        # naive fp32 recurrence (GVA: value head hv uses key head hv // 3)
        qn = torch.nn.functional.normalize(q[0].float(), dim=-1) * (128 ** -0.5)
        kn = torch.nn.functional.normalize(k[0].float(), dim=-1)
        S = torch.zeros((48, 128, 128), device=device)  # [HV, K, V]
        outs = []
        for i in range(t):
            S = S * gg[0, i].exp().view(48, 1, 1)
            kk = kn[i].repeat_interleave(3, dim=0)  # [48, K]
            qq = qn[i].repeat_interleave(3, dim=0)
            mem = torch.einsum("hk,hkv->hv", kk, S)
            delta = beta[0, i].view(48, 1) * (v[0, i].float() - mem)
            S = S + torch.einsum("hk,hv->hkv", kk, delta)
            outs.append(torch.einsum("hk,hkv->hv", qq, S))
        o_naive = torch.stack(outs).unsqueeze(0)
        print("  chunk vs naive fp32 recurrence:", rel(o_chunk, o_naive))
        z = qkvz[:, g.conv_dim :]
        core = g.norm(o_chunk.reshape(-1, 128), z.reshape(-1, 128).contiguous())
        core_ref = o_chunk.reshape(-1, 128).float()
        core_ref = core_ref * torch.rsqrt(core_ref.pow(2).mean(-1, keepdim=True) + 1e-6) * g.norm.weight.float() * torch.nn.functional.silu(z.reshape(-1, 128).float())
        print("  gated rmsnorm kernel vs torch:", rel(core, core_ref))
        out = ops.fp8_linear(core.reshape(t, -1), g.out)
        print("  full GDN (torch conv path) vs ref block:", rel(out, gdn_ref))

    # ---- layer 0: MLP ----
    with torch.inference_mode():
        mlp_in = l0.post_norm(r0.to(device))
        mlp_mine = l0.mlp(mlp_in)
        print("layer0 MLP (fused silu quant):", rel(mlp_mine, h0))
        l0.mlp.fused_silu_quant = False
        mlp_mine2 = l0.mlp(mlp_in)
        print("layer0 MLP (torch silu + fp4_quantize):", rel(mlp_mine2, h0))
        gu = ops.nvfp4_linear(mlp_in, l0.mlp.gate_up)
        # dequantized reference for gate_up using raw (pre-swizzle) scales
        raw_scale = torch.cat([w.get(f"{PREFIX}.layers.0.mlp.gate_proj.weight_scale"), w.get(f"{PREFIX}.layers.0.mlp.up_proj.weight_scale")], 0).to(device).float()
        ws2 = w.get(f"{PREFIX}.layers.0.mlp.gate_proj.weight_scale_2").float().item()
        full = dequant_nvfp4(l0.mlp.gate_up, device)
        w_deq = (full.reshape(full.shape[0], -1, 16) * raw_scale.unsqueeze(-1)).reshape(full.shape[0], -1) * ws2
        gu_ref = mlp_in.float() @ w_deq.T
        print("  nvfp4 gate_up vs dequant matmul (bf16 act):", rel(gu, gu_ref))

    # ---- layer 3: attention block ----
    h2, r2 = ref["layers"][2]
    h3, r3 = ref["layers"][3]
    in3 = (h2.float() + r2.float()).to(torch.bfloat16).to(device)
    attn_ref = r3.float().to(device) - in3.float()
    with torch.inference_mode():
        attn_mine = l3.mixer.prefill(l3.input_norm(in3), positions)
    print("layer3 attention block:", rel(attn_mine, attn_ref))
    with torch.inference_mode():
        mlp_mine3 = l3.mlp(l3.post_norm(r3.to(device)))
    print("layer3 MLP:", rel(mlp_mine3, h3))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rand8"))
