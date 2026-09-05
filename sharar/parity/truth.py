"""Distance of Sharar and of the reference from an fp32 dequantized 'truth'.

W4A4 and FP8 static activation quantization make block outputs sensitive to
tiny input differences. This measures, per block, how far Sharar's output and
the reference's dumped output each are from a torch fp32 computation with
dequantized weights and unquantized activations. If both are equally far and in
different directions, the mutual gap is quantization noise, not a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from sharar import config
from sharar.kernels import ops
from sharar.model.qwen38 import PREFIX, DecoderLayer
from sharar.model.rope import PartialRotary
from sharar.weights.safetensors import SnapshotWeights

FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def rel(a, b):
    a = a.float()
    b = b.float().to(a.device)
    d = (a - b).abs()
    s = b.abs().mean().item()
    return f"max {d.max().item() / s:.3f} mean {d.mean().item() / s:.4f}"


def cos(a, b):
    a = a.float().flatten()
    b = b.float().to(a.device).flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def deq_nvfp4(w: SnapshotWeights, prefixes, device):
    parts = []
    for p in prefixes:
        wt = w.get(f"{p}.weight").to(device)
        sc = w.get(f"{p}.weight_scale").to(device).float()
        ws2 = w.get(f"{p}.weight_scale_2").float().item()
        lo = (wt & 0xF).long()
        hi = (wt >> 4).long()
        vals = FP4.to(device)
        full = torch.stack([vals[lo], vals[hi]], -1).reshape(wt.shape[0], -1)
        full = (full.reshape(full.shape[0], -1, 16) * sc.unsqueeze(-1)).reshape(full.shape[0], -1) * ws2
        parts.append(full)
    return torch.cat(parts, 0)


def deq_fp8(w: SnapshotWeights, prefixes, device):
    return torch.cat([w.get(f"{p}.weight").to(device).float() * w.get(f"{p}.weight_scale").float().item() for p in prefixes], 0)


def main(reference: str, prompt: str) -> int:
    device = torch.device("cuda")
    snap = config.snapshot_dir()
    w = SnapshotWeights(snap)
    ref = torch.load(Path(reference) / "dump" / f"{prompt}.pt", map_location="cpu")
    ids = ref["input_ids"].to(device)
    t = ids.shape[0]
    rope = PartialRotary(config.HEAD_DIM, 64, config.ROPE_THETA, 16384, device)
    ragged = ops.RaggedAttention(device, config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM)
    l0 = DecoderLayer(w, 0, "linear_attention", device, rope, ragged, True)
    l3 = DecoderLayer(w, 3, "full_attention", device, rope, ragged, True)
    positions = torch.arange(t, device=device)
    p0 = f"{PREFIX}.layers.0"
    p3 = f"{PREFIX}.layers.3"

    # ---------- MLP layer 0 ----------
    h0, r0 = ref["layers"][0]
    x = l0.post_norm(r0.to(device))
    with torch.inference_mode():
        mine = l0.mlp(x)
        Wgu = deq_nvfp4(w, [f"{p0}.mlp.gate_proj", f"{p0}.mlp.up_proj"], device)
        Wd = deq_nvfp4(w, [f"{p0}.mlp.down_proj"], device)
        gu = x.float() @ Wgu.T
        truth = (torch.nn.functional.silu(gu[:, :17408]) * gu[:, 17408:]) @ Wd.T
        # same input perturbed by one bf16 ulp on random 10% of elements
        noise = torch.where(torch.rand_like(x.float()) < 0.1, torch.sign(torch.randn_like(x.float())), torch.zeros_like(x.float()))
        x_pert = (x.float() + noise * x.float().abs() * 2**-8).to(torch.bfloat16)
        mine_pert = l0.mlp(x_pert)
    print(f"MLP layer0:  mine vs truth {rel(mine, truth)} cos {cos(mine, truth):.5f} | ref vs truth {rel(h0, truth)} cos {cos(h0, truth):.5f} | mine vs ref {rel(mine, h0)} cos {cos(mine, h0):.5f}")
    print(f"             mine(x) vs mine(x + 1ulp noise on 10%) {rel(mine_pert, mine)} cos {cos(mine_pert, mine):.5f}")

    # ---------- GDN layer 0 (already matched to 0.14%) ----------
    # ---------- attention layer 3 ----------
    h2, r2 = ref["layers"][2]
    h3, r3 = ref["layers"][3]
    in3 = (h2.float() + r2.float()).to(torch.bfloat16).to(device)
    attn_ref = r3.float().to(device) - in3.float()
    xn = l3.input_norm(in3)
    with torch.inference_mode():
        mine_attn = l3.mixer.prefill(xn, positions)
        # fp32 truth: dequant FP8 weights, no activation quant, naive attention
        Wqkv = deq_fp8(w, [f"{p3}.self_attn.q_proj", f"{p3}.self_attn.k_proj", f"{p3}.self_attn.v_proj"], device)
        Wo = deq_fp8(w, [f"{p3}.self_attn.o_proj"], device)
        qkv = xn.float() @ Wqkv.T
        qg, k, v = qkv.split([24 * 512, 1024, 1024], dim=-1)
        qg = qg.view(t, 24, 512)
        q, gate = qg[..., :256], qg[..., 256:]
        qn_w = 1 + w.get(f"{p3}.self_attn.q_norm.weight").to(device).float()
        kn_w = 1 + w.get(f"{p3}.self_attn.k_norm.weight").to(device).float()
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * qn_w
        k = k.view(t, 4, 256)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * kn_w
        inv_freq = 1.0 / (config.ROPE_THETA ** (torch.arange(0, 64, 2, device=device).float() / 64))
        fr = torch.outer(positions.float(), inv_freq)
        c, s = fr.cos().unsqueeze(1), fr.sin().unsqueeze(1)

        def rot(xx):
            r_, rest = xx[..., :64], xx[..., 64:]
            x1, x2 = r_.chunk(2, -1)
            return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, rest), -1)

        q, k = rot(q), rot(k)
        v = v.view(t, 4, 256)
        kk = k.repeat_interleave(6, dim=1)
        vv = v.repeat_interleave(6, dim=1)
        scores = torch.einsum("thd,shd->hts", q, kk) / 16.0
        mask = torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = scores.softmax(-1)
        o = torch.einsum("hts,shd->thd", probs, vv)
        o = (o * torch.sigmoid(gate)).reshape(t, -1)
        truth_attn = o @ Wo.T
        # flashinfer attention alone vs naive, using the same bf16 q/k/v
        fa = ragged
        fa.plan([t], device)
        o_fi = fa.run(q.to(torch.bfloat16).contiguous(), k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous())
        o_naive = torch.einsum("hts,shd->thd", probs, vv).reshape(t, -1)
    print(f"ATTN layer3: mine vs truth {rel(mine_attn, truth_attn)} cos {cos(mine_attn, truth_attn):.5f} | ref vs truth {rel(attn_ref, truth_attn)} cos {cos(attn_ref, truth_attn):.5f} | mine vs ref {rel(mine_attn, attn_ref)} cos {cos(mine_attn, attn_ref):.5f}")
    print(f"             flashinfer ragged vs naive softmax attention (same q/k/v): {rel(o_fi, o_naive)}")

    # ---------- GDN layer 0 vs truth (FP8 dequant, torch conv, chunk kernel) ----------
    gdn_ref = r0.float().to(device) - ref["embed"].float().to(device)
    x0 = torch.nn.functional.embedding(ids, w.get(f"{PREFIX}.embed_tokens.weight").to(device))
    with torch.inference_mode():
        mine_gdn = l0.mixer.prefill(l0.input_norm(x0))
    print(f"GDN layer0:  mine vs ref {rel(mine_gdn, gdn_ref)} cos {cos(mine_gdn, gdn_ref):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rand8"))
