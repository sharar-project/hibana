# SPDX-License-Identifier: Apache-2.0
"""Plain-torch reference forward for Qwen3.8-Flash-Next (prefill, no cache).

The specification is SGLang's `qwen4_exp.py` (Apache-2.0); this file follows
its equations so the engine's kernels can be judged against it layer by
layer. Everything is BF16 like the checkpoint except the routed experts,
which are ModelOpt NVFP4 W4A4: weights are dequantized exactly, activations
are fake-quantized to E2M1 blocks of 16 with the checkpoint's static input
scale, and the product runs in FP32. While the prompt fits the indexer
budget (2,048 tokens plus the pending group) the top-k over compressed
blocks selects every block, so attention is dense causal. Longer prompts run the sparse indexer (`qsa_token_mask`) and attend
over its selection in row chunks.

Weights are read straight from the safetensors shards by their checkpoint
names; the 51 GB n-gram table stays on the host as a memory map and only the
rows a prompt touches are gathered.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from sharar.model.flashnext.hyper import HyperConnection, grouped_gemma_rms
from sharar.weights.safetensors import SnapshotWeights

PREFIX = "model.language_model"
FP4_MAGNITUDES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


class FlashNextConfig:
    def __init__(self, snap: Path):
        cfg = json.loads((snap / "config.json").read_text())
        t = cfg["text_config"]
        self.hidden = t["hidden_size"]
        self.num_layers = t["num_hidden_layers"]
        self.layer_types = list(t["layer_types"])
        self.heads, self.kv_heads, self.head_dim = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        self.rotary_dim = int(self.head_dim * t["partial_rotary_factor"])
        self.rope_theta = float(t["rope_parameters"]["rope_theta"])
        self.k_heads, self.v_heads = t["linear_num_key_heads"], t["linear_num_value_heads"]
        self.k_dim, self.v_dim = t["linear_key_head_dim"], t["linear_value_head_dim"]
        self.conv_kernel = t["linear_conv_kernel_dim"]
        self.output_gate = t.get("output_gate_type") or "swish"  # GDN output gate: Flash-Next uses sigmoid, the 27B silu
        self.experts, self.topk, self.expert_dim = t["num_experts"], t["num_experts_per_tok"], t["moe_intermediate_size"]
        self.shared_dim = t["shared_expert_intermediate_size"]
        self.norm_topk = t.get("norm_topk_prob", True)
        self.hc, self.hc_rank = t["hc_count"], t["hc_lowrank"]
        self.eps = t["rms_norm_eps"]
        self.vocab = t["vocab_size"]
        self.eos = t["eos_token_id"]
        self.ple_layer = t["ple_layer_ids"][0] - 1  # ple_layer_ids counts from one
        self.ngram, self.heads_per_ngram = t["ngram_size"], t["heads_per_ngram"]
        self.ple_dim, self.ple_conv = t["ple_embed_dim"], t["ple_conv_kernel_size"]
        self.indexer_budget, self.indexer_ratio = t["indexer_budget"], t["indexer_compress_ratio"]
        self.max_dense_prompt = self.indexer_budget + self.indexer_ratio - 1
        self.indexer_heads, self.indexer_dim = t["indexer_n_heads"], t["indexer_head_dim"]


def gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xn * (1.0 + weight.float())).to(x.dtype)


def rope_neox(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    """x [T, H, D]: rotate the first rotary_dim dims, NeoX halves, cos/sin in float32."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim))
    ang = positions.float()[:, None] * inv[None]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    r, rest = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    a, b = r[..., : rotary_dim // 2], r[..., rotary_dim // 2 :]
    rotated = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).to(x.dtype)
    return torch.cat([rotated, rest], dim=-1)


def qsa_token_mask(q: torch.Tensor, keys: torch.Tensor, budget: int, ratio: int) -> torch.Tensor:
    """[T, T] bool, True where query row i attends to token j, for one sequence
    starting at position 0 (Qwen Sparse Attention, as SGLang's qsa package).

    q [T, H, D] are the normed, rotated index queries; keys [G, D] the
    compressed keys of the G = T // ratio complete groups (mean of the group's
    raw index keys, Gemma norm, rotated at the group's first position). Row i
    sees the (i + 1) // ratio groups that end at or before it, scores them
    relu(q . k) summed over heads / sqrt(D), keeps the top budget // ratio,
    and adds the tail of its own incomplete group (at most ratio - 1 tokens,
    itself included). With at most budget // ratio groups everything is kept:
    dense causal attention."""
    t, d = q.shape[0], q.shape[-1]
    g = keys.shape[0]
    rows = torch.arange(t, device=q.device)
    visible = (rows + 1) // ratio
    mask = torch.zeros((t, t), dtype=torch.bool, device=q.device)
    tail_start = visible * ratio
    cols = torch.arange(t, device=q.device)
    mask |= (cols[None, :] >= tail_start[:, None]) & (cols[None, :] <= rows[:, None])
    if g == 0:
        return mask
    top = budget // ratio
    for a in range(0, t, 1024):  # [rows, groups] logits in row chunks
        b = min(t, a + 1024)
        logits = torch.relu(torch.einsum("thd,gd->tgh", q[a:b].float(), keys.float())).sum(-1) / d ** 0.5
        seen = torch.arange(g, device=q.device)[None, :] < visible[a:b, None]
        logits = logits.masked_fill(~seen, -float("inf"))
        width = min(top, g)
        values, chosen = logits.topk(width, dim=-1)
        chosen = torch.where(torch.isfinite(values), chosen, torch.full_like(chosen, -1))
        groups = torch.zeros((b - a, g + 1), dtype=torch.bool, device=q.device)
        groups.scatter_(1, torch.where(chosen >= 0, chosen, torch.full_like(chosen, g)), True)
        member = groups[:, :g].repeat_interleave(ratio, dim=1)  # [rows, g * ratio]
        mask[a:b, : g * ratio] |= member
    return mask


class ExpertBank:
    """One layer's routed experts as stacked NVFP4 tensors on the device; dequantized per call."""

    def __init__(self, w: SnapshotWeights, prefix: str, count: int, device):
        self.device = device
        self.packed, self.scales, self.scale2, self.input_scale = {}, {}, {}, {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            ps, ss, s2, ins = [], [], [], []
            for e in range(count):
                p = f"{prefix}.{e}.{name}"
                ps.append(w.get(f"{p}.weight")); ss.append(w.get(f"{p}.weight_scale"))
                s2.append(w.get(f"{p}.weight_scale_2").float().reshape(())); ins.append(w.get(f"{p}.input_scale").float().reshape(()))
            self.packed[name] = torch.stack(ps).to(device)
            self.scales[name] = torch.stack(ss).to(device)
            self.scale2[name] = torch.stack(s2).to(device)
            self.input_scale[name] = torch.stack(ins).to(device)
        # The fused kernel quantizes activations once per layer with the loudest expert's scale
        # (SGLang ModelOptNvFp4FusedMoE takes the max over experts for w13 and for w2).
        self.act_scale_w13 = torch.maximum(self.input_scale["gate_proj"].max(), self.input_scale["up_proj"].max())
        self.act_scale_w2 = self.input_scale["down_proj"].max()

    @staticmethod
    def dequant(packed: torch.Tensor, scales: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
        codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1).long()
        mags = FP4_MAGNITUDES.to(packed.device)
        values = mags[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
        return values * scales.float().repeat_interleave(16, -1) * scale2.float()

    def weight(self, name: str, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dequant(self.packed[name][e], self.scales[name][e], self.scale2[name][e]), self.input_scale[name][e]


def fake_quant_nvfp4(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """ModelOpt W4A4 activations: per-16 block E4M3 scale from the static global scale, E2M1 values."""
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], -1, 16)
    amax = blocks.abs().amax(-1, keepdim=True)
    gs = 1.0 / input_scale  # global scale = 1 / input_scale (ModelOpt convention)
    # The kernel's cvt to E4M3 saturates at 448: blocks louder than the calibrated amax keep
    # that scale and clip at 6 below. (An unclamped torch cast yields NaN and zeroed the block.)
    sf = (gs * amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    inv = torch.where(sf > 0, gs / sf, torch.zeros_like(sf))
    v = blocks * inv
    mags = FP4_MAGNITUDES.to(x.device)
    a = v.abs().clamp(max=6.0)
    idx = torch.bucketize(a, (mags[:-1] + mags[1:]) / 2)  # nearest magnitude (ties resolved upward; fine for a reference)
    q = mags[idx] * torch.sign(v)
    return (q * sf / gs).flatten(-2)


class FlashNextReference:
    def __init__(self, snap: Path, device="cuda", layers: int | None = None, log=print):
        self.snap, self.device = Path(snap), torch.device(device)
        self.cfg = c = FlashNextConfig(self.snap)
        self.w = w = SnapshotWeights(self.snap)
        self.num_layers = c.num_layers if layers is None else layers
        d = self.device
        self.embed = w.get(f"{PREFIX}.embed_tokens.weight").to(d)
        self.lm_head = w.get("lm_head.weight").to(d)
        self.mixer = HyperConnection.load(w, f"{PREFIX}.hyper_connection_mixer", c.hc, c.hidden, c.eps, d, combine=False, fused=False)
        self.layers = []
        for i in range(self.num_layers):
            p = f"{PREFIX}.layers.{i}"
            layer = {"kind": c.layer_types[i],
                     "attn_hc": HyperConnection.load(w, f"{p}.attn_hyper_connection", c.hc, c.hidden, c.eps, d, fused=False),
                     "mlp_hc": HyperConnection.load(w, f"{p}.mlp_hyper_connection", c.hc, c.hidden, c.eps, d, fused=False),
                     "router": w.get(f"{p}.mlp.gate.weight").to(d),
                     "shared_gate": w.get(f"{p}.mlp.shared_expert_gate.weight").to(d),
                     "shared": {n: w.get(f"{p}.mlp.shared_expert.{n}.weight").to(d) for n in ("gate_proj", "up_proj", "down_proj")},
                     "experts": ExpertBank(w, f"{p}.mlp.experts", c.experts, d)}
            if layer["kind"] == "linear_attention":
                q = f"{p}.linear_attn"
                layer.update({n: w.get(f"{q}.{n}").to(d) for n in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
                                                                    "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight")})
            else:
                q = f"{p}.self_attn"
                layer.update({n: w.get(f"{q}.{n}").to(d) for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                                                    "q_norm.weight", "k_norm.weight")})
                layer.update({f"indexer.{n}": w.get(f"{q}.indexer.{n}").to(d)
                              for n in ("index_qk_proj.weight", "q_layernorm.weight", "k_layernorm.weight")})
            if i == c.ple_layer:
                layer["ple"] = self._load_ple(f"{p}.ple")
            self.layers.append(layer)
            if log and (i + 1) % 8 == 0:
                log(f"  reference: {i + 1}/{self.num_layers} layers")
        # The reader stays open: the PLE table is served from its memory maps.

    # ---- PLE ----
    def _load_ple(self, p: str) -> dict:
        w, d, c = self.w, self.device, self.cfg
        ple = {n: w.get(f"{p}.{n}").to(d) for n in ("key_proj.weight", "value_proj.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight", "conv1d.weight")}
        e = f"{p}.ple_embedding"
        ple["multipliers"] = w.get(f"{e}.layer_multipliers").to(torch.int64)
        ple["vocab_sizes"] = w.get(f"{e}.ngram_heads_vocab_sizes").to(torch.int64)
        ple["offsets"] = w.get(f"{e}.ngram_heads_offsets").to(torch.int64)
        ple["scale"] = w.get(f"{e}.ngram_embedding.weight_scale").float().reshape(())
        heads = (c.ngram - 1) * c.heads_per_ngram
        # Host-side memory maps of the 128 FP8 shards (no copy), concatenated by row offset.
        from safetensors import safe_open
        rows, shards = 0, []
        self._ple_handles = {}
        for s_ in range(128):
            name = f"{e}.ngram_embedding.shard_{s_}.weight"
            shard = w._map[name]
            handle = self._ple_handles.get(shard)
            if handle is None:
                handle = self._ple_handles[shard] = safe_open(str(self.snap / shard), framework="pt", device="cpu")
            t = handle.get_slice(name)
            n = t.get_shape()[0]
            shards.append((rows, rows + n, t))
            rows += n
        ple["shards"], ple["rows"], ple["heads"], ple["head_dim"] = shards, rows, heads, c.ple_dim // heads
        return ple

    def _ngram_ids(self, ids: torch.Tensor, ple: dict) -> torch.Tensor:
        """[T, heads] table rows, following Qwen4ExpNGramEmbedding._hash_contexts.

        The history before a sequence is EOS. For the token n back, SGLang's
        _shift_right_ignore_eos on the (t-2, t-1, t) window yields the token
        itself when no EOS lies in the n-1 tokens between it and the current
        one, else EOS: back-1 is always t-1 (an EOS there reads as EOS anyway),
        back-2 is t-2 unless t-1 or t-2 is EOS.
        """
        c = self.cfg
        assert c.ngram == 3, "the reference spells out the 3-gram rule"
        ids = ids.to(torch.int64)
        eos = torch.full((1,), c.eos, dtype=torch.int64)
        back1 = torch.cat([eos, ids[:-1]])
        back2 = torch.cat([eos, eos, ids[:-2]])
        back2 = torch.where((back1 == c.eos) | (back2 == c.eos), torch.full_like(back2, c.eos), back2)
        shifted = [ids, back1, back2]
        blocks = []
        mult = ple["multipliers"]
        for n in range(2, c.ngram + 1):
            mix = shifted[0] * mult[0]
            for pos in range(1, n):
                mix = torch.bitwise_xor(mix, shifted[pos] * mult[pos])
            lo, hi = (n - 2) * c.heads_per_ngram, (n - 1) * c.heads_per_ngram
            blocks.append(torch.remainder(mix[:, None], ple["vocab_sizes"][lo:hi][None]) + ple["offsets"][lo:hi][None])
        return torch.cat(blocks, dim=-1)

    def _gather_rows(self, rows: torch.Tensor, ple: dict) -> torch.Tensor:
        out = torch.empty((rows.numel(), ple["head_dim"]), dtype=torch.bfloat16)
        flat = rows.reshape(-1)
        for lo, hi, t in ple["shards"]:
            sel = (flat >= lo) & (flat < hi)
            if sel.any():
                local = (flat[sel] - lo).tolist()
                picked = torch.stack([t[r:r + 1][0] for r in local])  # sliced reads from the memory map
                out[sel] = picked.view(torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
        return out.reshape(*rows.shape, ple["head_dim"]).to(self.device)

    def _ple(self, hyper: torch.Tensor, ids: torch.Tensor, ple: dict) -> torch.Tensor:
        c = self.cfg
        rows = self._ngram_ids(ids, ple)  # [T, heads]
        emb = (self._gather_rows(rows, ple).float() * ple["scale"]).to(torch.bfloat16).flatten(-2)  # [T, ple_dim]
        key = F.linear(emb, ple["key_proj.weight"]).reshape(-1, c.hc, c.hidden)
        value = F.linear(emb, ple["value_proj.weight"])  # [T, hidden]
        query = hyper.reshape(-1, c.hc, c.hidden)
        key_n = grouped_gemma_rms(key.flatten(-2), ple["norm_key.weight"], c.eps, c.hidden).unflatten(-1, (c.hc, c.hidden))
        query_n = grouped_gemma_rms(query.flatten(-2), ple["norm_query.weight"], c.eps, c.hidden).unflatten(-1, (c.hc, c.hidden))
        gate = (key_n * query_n).sum(-1, keepdim=True) / math.sqrt(c.hidden)
        gate = torch.sigmoid(gate.abs().clamp_min(1e-6).sqrt() * gate.sign())
        gated = gate * value.unsqueeze(-2)  # [T, hc, hidden]
        gated_n = grouped_gemma_rms(gated.flatten(-2), ple["norm_conv.weight"], c.eps, c.hidden)
        # Depthwise causal conv over time, kernel ple_conv, dilation ngram, zero history.
        x = gated_n.transpose(0, 1).unsqueeze(0)  # [1, C, T]
        pad = (c.ple_conv - 1) * c.ngram
        conv = F.conv1d(F.pad(x, (pad, 0)), ple["conv1d.weight"].to(x.dtype), dilation=c.ngram, groups=x.shape[1])
        conv = F.silu(conv.squeeze(0).transpose(0, 1))
        return (gated.flatten(-2) + conv).to(hyper.dtype)

    # ---- mixers ----
    def _gdn(self, x: torch.Tensor, L: dict) -> torch.Tensor:
        from sharar.kernels.fla.chunk import chunk_gated_delta_rule
        c = self.cfg
        t = x.shape[0]
        key_dim, value_dim = c.k_heads * c.k_dim, c.v_heads * c.v_dim
        qkv = F.linear(x, L["in_proj_qkv.weight"])
        z = F.linear(x, L["in_proj_z.weight"])
        a, b = F.linear(x, L["in_proj_a.weight"]), F.linear(x, L["in_proj_b.weight"])
        conv_w = L["conv1d.weight"].reshape(qkv.shape[-1], c.conv_kernel)
        qkv = F.silu(F.conv1d(F.pad(qkv.transpose(0, 1).unsqueeze(0), (c.conv_kernel - 1, 0)), conv_w.unsqueeze(1), groups=qkv.shape[-1]).squeeze(0).transpose(0, 1))
        q, k, v = torch.split(qkv, [key_dim, key_dim, value_dim], dim=-1)
        q = q.reshape(1, t, c.k_heads, c.k_dim)
        k = k.reshape(1, t, c.k_heads, c.k_dim)
        v = v.reshape(1, t, c.v_heads, c.v_dim)
        g = (-L["A_log"].float().exp() * F.softplus(a.float() + L["dt_bias"].float())).unsqueeze(0)  # [1, T, Hv]
        beta = torch.sigmoid(b.float()).unsqueeze(0)
        # Sharar's kernel is pool-indexed: one zero state, one sequence, 64-token chunk table.
        d = x.device
        init = torch.zeros((1, c.v_heads, c.v_dim, c.k_dim), dtype=torch.float32, device=d)
        chunks = (t + 63) // 64
        core, _, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=init, initial_state_indices=torch.zeros(1, dtype=torch.int32, device=d),
            cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device=d),
            chunk_indices=torch.tensor([[0, ci] for ci in range(chunks)], dtype=torch.int32, device=d),
            chunk_offsets=torch.tensor([0, chunks], dtype=torch.int64, device=d),
            head_first=False, use_qk_l2norm_in_kernel=True)
        core = core.reshape(t, c.v_heads, c.v_dim).float()
        # Gated RMS norm per head with a plain weight (the checkpoint's norm.weight is ~1, not zero-centred),
        # then the output gate named by `output_gate_type`: sigmoid(z) here, silu(z) on the 27B.
        normed = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + c.eps) * L["norm.weight"].float()
        zf = z.float().reshape(t, c.v_heads, c.v_dim)
        gate = torch.sigmoid(zf) if c.output_gate == "sigmoid" else F.silu(zf)
        out = (normed * gate).reshape(t, value_dim).to(x.dtype)
        return F.linear(out, L["out_proj.weight"])

    def _attention(self, x: torch.Tensor, positions: torch.Tensor, L: dict) -> torch.Tensor:
        c = self.cfg
        t = x.shape[0]
        q_gate = F.linear(x, L["q_proj.weight"]).view(t, c.heads, 2 * c.head_dim)
        q, gate = q_gate[..., : c.head_dim], q_gate[..., c.head_dim :]
        k = F.linear(x, L["k_proj.weight"]).view(t, c.kv_heads, c.head_dim)
        v = F.linear(x, L["v_proj.weight"]).view(t, c.kv_heads, c.head_dim)
        q = gemma_rms(q, L["q_norm.weight"], c.eps)
        k = gemma_rms(k, L["k_norm.weight"], c.eps)
        q = rope_neox(q, positions, c.rotary_dim, c.rope_theta)
        k = rope_neox(k, positions, c.rotary_dim, c.rope_theta)
        rep = c.heads // c.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        if t <= c.max_dense_prompt or "dense" in self.ablate:  # every block selected (or the dense control): causal attention
            attn = F.scaled_dot_product_attention(q.transpose(0, 1).float(), k.transpose(0, 1).float(), v.transpose(0, 1).float(), is_causal=True, scale=c.head_dim**-0.5)
            attn = attn.transpose(0, 1)
        else:
            mask = self._qsa_mask(x, positions, L)
            attn = torch.empty((t, c.heads, c.head_dim), dtype=torch.float32, device=x.device)
            kf, vf = k.transpose(0, 1).float(), v.transpose(0, 1).float()
            for a in range(0, t, 512):  # [heads, rows, T] scores in row chunks
                b = min(t, a + 512)
                attn[a:b] = F.scaled_dot_product_attention(q[a:b].transpose(0, 1).float(), kf, vf, attn_mask=mask[a:b],
                                                           scale=c.head_dim**-0.5).transpose(0, 1)
        attn = attn.to(x.dtype) * torch.sigmoid(gate.float()).to(x.dtype)
        return F.linear(attn.reshape(t, -1), L["o_proj.weight"])

    def _qsa_mask(self, x: torch.Tensor, positions: torch.Tensor, L: dict) -> torch.Tensor:
        """The indexer's selection for this layer (index keys are not cached: the reference prefills whole prompts)."""
        c = self.cfg
        t, h, dim, ratio = x.shape[0], c.indexer_heads, c.indexer_dim, c.indexer_ratio
        qk = F.linear(x, L["indexer.index_qk_proj.weight"])
        q = gemma_rms(qk[:, : h * dim].reshape(t, h, dim), L["indexer.q_layernorm.weight"], c.eps)
        q = rope_neox(q, positions, c.rotary_dim, c.rope_theta)
        g = t // ratio
        raw = qk[: g * ratio, h * dim:].reshape(g, ratio, 1, dim)
        keys = raw.float().mean(dim=1).to(x.dtype)  # fp32 mean, stored in the activation dtype
        keys = gemma_rms(keys, L["indexer.k_layernorm.weight"], c.eps)
        keys = rope_neox(keys, positions[: g * ratio : ratio], c.rotary_dim, c.rope_theta)
        return qsa_token_mask(q, keys[:, 0], c.indexer_budget, ratio)

    def _moe(self, x: torch.Tensor, L: dict) -> torch.Tensor:
        c = self.cfg
        logits = F.linear(x.float(), L["router"].float())
        probs = torch.softmax(logits, dim=-1)
        weights, ids = torch.topk(probs, c.topk, dim=-1)
        if c.norm_topk:
            weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros((x.shape[0], c.hidden), dtype=torch.float32, device=x.device)
        for e in ids.unique().tolist():
            rows, slots = (ids == e).nonzero(as_tuple=True)
            bank = L["experts"]
            xe = x[rows]
            (wg, sg), (wu, su), (wd, sd) = bank.weight("gate_proj", e), bank.weight("up_proj", e), bank.weight("down_proj", e)
            if "fq" in self.ablate:
                xq = xe.float()
                h = F.silu(xq @ wg.T) * (xq @ wu.T)
                y = h @ wd.T
            else:
                xq = fake_quant_nvfp4(xe, bank.act_scale_w13)  # one activation scale per layer, like the fused kernel
                h = F.silu(xq @ wg.T) * (xq @ wu.T)
                hq = fake_quant_nvfp4(h.to(torch.bfloat16), bank.act_scale_w2)
                y = hq @ wd.T
            out.index_add_(0, rows, y * weights[rows, slots].unsqueeze(-1))
        s = L["shared"]
        shared = F.linear(F.silu(F.linear(x, s["gate_proj"])) * F.linear(x, s["up_proj"]), s["down_proj"])
        shared = torch.sigmoid(F.linear(x.float(), L["shared_gate"].float())) * shared.float()
        if "shared" in self.ablate:
            return out.to(x.dtype)
        return (out + shared).to(x.dtype)

    # ---- ablation switches ----
    def _mix(self, hc: HyperConnection, hyper: torch.Tensor):
        if "hc" in self.ablate:  # plain average of the streams, no gates
            return hyper.unflatten(-1, (hc.hc, hc.hidden)).mean(-2), (hyper, None)
        return hc.mix(hyper)

    def _combine(self, hc: HyperConnection, block: torch.Tensor, res):
        if "hc" in self.ablate:
            hyper = res[0]
            return (hyper.unflatten(-1, (hc.hc, hc.hidden)) + block.unsqueeze(-2)).flatten(-2)
        return hc.combine(block, res)

    # ---- forward ----
    ablate: set = frozenset()  # diagnostics: {"ple", "fq", "shared", "hc"}

    @torch.inference_mode()
    def forward(self, ids: list[int], capture: bool = False) -> dict:
        c, d = self.cfg, self.device
        tokens = torch.tensor(ids, dtype=torch.int64, device=d)
        positions = torch.arange(len(ids), device=d)
        x = F.embedding(tokens, self.embed)
        hyper = torch.cat([x] * c.hc, dim=-1)
        out = {"layers": []}
        for i, L in enumerate(self.layers):
            if "ple" in L and "ple" not in self.ablate:
                hyper = hyper + self._ple(hyper, tokens.cpu(), L["ple"])
            h, res = self._mix(L["attn_hc"], hyper)
            h = self._gdn(h, L) if L["kind"] == "linear_attention" else self._attention(h, positions, L)
            hyper = self._combine(L["attn_hc"], h, res)
            h, res = self._mix(L["mlp_hc"], hyper)
            h = self._moe(h, L)
            hyper = self._combine(L["mlp_hc"], h, res)
            if capture:
                out["layers"].append(hyper.float().pow(2).mean().sqrt().item())
        final, _ = self.mixer.mix(hyper)
        out["hidden"] = final
        out["logits"] = F.linear(final, self.lm_head).float()
        return out
