"""Bolt text inference with native packed weights and the Nemotron Mamba runtime.

Every block's expert count/width comes from its pinned Puzzle configuration.
IQ2 routed and shared experts stay packed; Q8 attention/head/latent projections
and NVFP4 Mamba projections use weight-only BF16 kernels. No MTP is in this pin.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from sharar.distributed import get_tp_info
from sharar.engine.config import current
from sharar.kernels import ggml_bf16, ops
from sharar.model.bolt.config import BOLT
from sharar.model.bolt.linear import PackedLinear
from sharar.model.nemotron.attention import NemotronAttention
from sharar.model.nemotron.mamba import NemotronMamba
from sharar.model.nemotron.model import BACKBONE, Block, Nemotron, drop_page_cache
from sharar.model.nemotron.moe import route_topk
from sharar.model.nemotron.norm import RMSNorm
from sharar.weights.safetensors import SnapshotWeights


class BoltMoe:
    def __init__(self, w, prefix, spec, device):
        self.experts, self.inter, self.topk = spec.experts, spec.intermediate, spec.topk
        self.router = w.get(f"{prefix}.gate.weight").to(device).contiguous()
        self.bias = w.get(f"{prefix}.gate.e_score_correction_bias").to(device).float().contiguous()
        self.fc1 = PackedLinear.load(w, [f"{prefix}.fc1_latent_proj"], device)
        self.fc2 = PackedLinear.load(w, [f"{prefix}.fc2_latent_proj"], device)
        self.shared_up = PackedLinear.load(w, [f"{prefix}.shared_experts.up_proj"], device)
        self.shared_down = PackedLinear.load(w, [f"{prefix}.shared_experts.down_proj"], device)
        names = [f"{prefix}.experts.{e}.{proj}_proj.weight" for e in range(self.experts) for proj in ("up", "down")]
        tensors = w.get_whole(names, device)
        self.w1 = torch.stack([tensors[f"{prefix}.experts.{e}.up_proj.weight"] for e in range(self.experts)])
        self.w2 = torch.stack([tensors[f"{prefix}.experts.{e}.down_proj.weight"] for e in range(self.experts)])
        if self.w1.shape != (self.experts, self.inter, BOLT.latent // 256, 66) or self.w2.shape != (
                self.experts, BOLT.latent, self.inter // 256, 66):
            raise ValueError(f"{prefix}: encoded expert shapes differ from the pinned Puzzle block")
        if self.router.shape != (self.experts, BOLT.hidden) or self.bias.shape != (self.experts,):
            raise ValueError(f"{prefix}: router shape differs from the pinned Puzzle block")

    @property
    def expert_bytes(self):
        return self.w1.numel() + self.w2.numel()

    @property
    def bytes(self):
        return (self.expert_bytes + self.router.numel() * self.router.element_size() + self.bias.numel() * 4
                + sum(lin.bytes for lin in (self.fc1, self.fc2, self.shared_up, self.shared_down)))

    def __call__(self, x, ctx=None):
        x = x.contiguous()
        logits = torch.mm(x, self.router.T, out_dtype=torch.float32)
        latent = self.fc1(x)
        fused = torch.cat((logits, latent.float()), dim=-1)
        slots, steps = None, 1
        if ctx is not None and ctx.mode in ("decode", "verify"):
            slots, steps = (ctx.slots, 1) if ctx.mode == "decode" else (ctx.seq_slots, ctx.verify_steps)
        ids, weights, latent = route_topk(fused, self.bias, self.topk, BOLT.latent, slots, steps, repeat=False)
        up = ggml_bf16.matmul(latent, self.w1, "IQ2_XXS", ids=ids, repeat_input=True, relu2=True)
        pairs = ggml_bf16.matmul(up, self.w2, "IQ2_XXS", ids=ids)
        routed = ggml_bf16.combine(pairs, weights, BOLT.route_scale)
        shared = self.shared_down(self.shared_up(x, relu2=True))
        return self.fc2(routed) + shared


class BoltBlock(Block):
    def __init__(self, w, index, device, ragged, target):
        self.index, self.kind = index, target.layer_types[index]
        p = f"{BACKBONE}.layers.{index}"
        self.norm = RMSNorm(w.get(f"{p}.norm.weight").to(device), target.rms_norm_eps)
        if self.kind == "linear_attention":
            self.mixer = NemotronMamba(w, f"{p}.mixer", index, device, target, load_linear=PackedLinear.load)
        elif self.kind == "full_attention":
            self.mixer = NemotronAttention(w, f"{p}.mixer", index, device, ragged, target, load_linear=PackedLinear.load)
        else:
            self.mixer = BoltMoe(w, f"{p}.mixer", BOLT.blocks[index], device)


class Bolt(Nemotron):
    def __init__(self, snap: Path, device="cuda", log=print, target=None, **_ignored):
        from sharar.cache.ledger import check_load_headroom
        from sharar.target import NEMOTRON_BOLT
        target = target or NEMOTRON_BOLT
        if get_tp_info().size != 1:
            raise ValueError("Bolt serves on one Spark")
        cfg = current()
        if cfg.w3_decode or cfg.moe_keep or cfg.moe_budget or cfg.moe_floors or cfg.moe_tau or cfg.moe_extra:
            raise ValueError("Bolt uses the checkpoint's packed weights and all nine routed experts; disable W3 and expert pruning")
        if cfg.kv_format not in ("fp8", "sim-bf16"):
            raise ValueError("Bolt currently supports FP8 KV (sim-bf16 for parity)")
        target.load_model_config(snap)
        check_load_headroom(target.weight_load_peak_bytes)
        self.target, self.device = target, torch.device(device)
        self.eos, self.w3_identity, self.mtp_experts3 = 11, None, None
        started = time.perf_counter()
        w = SnapshotWeights(snap)
        self.layers = []
        try:
            self.embed = w.get(f"{BACKBONE}.embeddings.weight").to(self.device).contiguous()
            if self.embed.shape != (BOLT.vocab, BOLT.hidden) or self.embed.dtype != torch.bfloat16:
                raise ValueError("unexpected Bolt BF16 embedding shape")
            self.ragged = ops.RaggedAttention(self.device, target.num_attention_heads, target.num_kv_heads, target.head_dim)
            for i in range(target.num_layers):
                self.layers.append(BoltBlock(w, i, self.device, self.ragged, target))
                w.close()
                if log and (i + 1) % 11 == 0:
                    log(f"  loaded {i + 1}/{target.num_layers} Bolt blocks ({time.perf_counter() - started:.0f}s, "
                        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
            self.final_norm = RMSNorm(w.get(f"{BACKBONE}.norm_f.weight").to(self.device), target.rms_norm_eps)
            self.lm_head = PackedLinear.load(w, ["language_model.lm_head"], self.device)
        finally:
            w.close()
        torch.cuda.synchronize()
        drop_page_cache(snap)
        moes = [layer.mixer for layer in self.layers if layer.kind == "moe"]
        self.stream_bytes = (sum(layer.bytes for layer in self.layers) + self.final_norm.bytes + self.lm_head.bytes
                             + self.embed[0].numel() * self.embed.element_size() - sum(m.expert_bytes for m in moes)
                             + sum(m.expert_bytes * m.topk // m.experts for m in moes))
        if log:
            log(f"  Bolt loaded in {time.perf_counter() - started:.0f}s; {torch.cuda.memory_allocated() / 2**30:.2f} GiB CUDA allocated")

    def embed_ids(self, input_ids):
        return torch.nn.functional.embedding(input_ids, self.embed)

    def linears(self):
        for layer in self.layers:
            for name in ("in_proj", "out_proj", "qkv", "o", "fc1", "fc2", "shared_up", "shared_down"):
                lin = getattr(layer.mixer, name, None)
                if lin is not None:
                    yield f"layers.{layer.index}.mixer.{name}", lin
        yield "lm_head", self.lm_head

    @torch.inference_mode()
    def logits(self, hidden):
        return self.lm_head(hidden.contiguous())

    head_logits = logits
