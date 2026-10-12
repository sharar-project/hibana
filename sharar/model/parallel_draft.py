# SPDX-License-Identifier: Apache-2.0
"""Pinned Qwen DFlash2 backbone, without Transformers or SGLang.

Model equations and checkpoint names follow the publishers' draft models and
the Apache-2.0 SGLang implementation (see THIRD_PARTY_NOTICES.md). The five
target features feed context K/V directly; only the short noise block traverses
the draft layers. Attention planning and cache ownership stay in the adapter.
"""
import json
import math

import torch
import torch.nn.functional as F
from safetensors import safe_open


def rms_reference(x, weight, eps=1e-6):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def rms(x, weight, eps=1e-6):
    if x.is_cuda and x.dtype == weight.dtype == torch.bfloat16:
        from sharar.kernels.draft_norm import rms_bf16
        return rms_bf16(x, weight, eps)
    return rms_reference(x, weight, eps)


def frequencies(cfg, capacity, device):
    rope = cfg['rope_parameters']
    dim, base = cfg['head_dim'], rope['rope_theta']
    inv = 1 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    magnitude = 1.
    if rope['rope_type'] == 'yarn':
        factor, original = rope['factor'], rope['original_max_position_embeddings']
        correction = lambda rotations: dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(0, math.floor(correction(rope['beta_fast'])))
        high = min(dim - 1, math.ceil(correction(rope['beta_slow'])))
        ramp = ((torch.arange(dim // 2, device=device) - low) / max(high - low, .001)).clamp(0, 1)
        inv = inv / factor * ramp + inv * (1 - ramp)
        magnitude = .1 * math.log(factor) + 1 if factor > 1 else 1
    elif rope['rope_type'] != 'default':
        raise ValueError('unsupported draft RoPE')
    angles = torch.arange(capacity, device=device, dtype=torch.float32)[:, None] * inv[None]
    return angles.cos() * magnitude, angles.sin() * magnitude


def rotate_reference(x, positions, cos, sin):
    # HF computes rotations with cos/sin cast to the activation dtype.
    c, s = cos[positions].to(x.dtype)[:, None], sin[positions].to(x.dtype)[:, None]
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * c - b * s, b * c + a * s], dim=-1)


def rotate(x, positions, cos, sin):
    if x.is_cuda and x.dtype == torch.bfloat16:
        from sharar.kernels.draft_rope import rotate_bf16
        return rotate_bf16(x, positions, cos, sin)
    return rotate_reference(x, positions, cos, sin)


def rotate_and_cache(k, v, positions, cos, sin, caches, pages):
    if k.is_cuda and k.dtype == v.dtype == torch.bfloat16:
        from sharar.kernels.draft_rope import rotate_bf16
        rotate_bf16(k, positions, cos, sin, values=v, caches=caches, pages=pages)
    else:
        caches[0][pages, 0] = rotate_reference(k, positions, cos, sin)
        caches[1][pages, 0] = v


def convolve_reference(x, coefficients, base):
    """Block-local two-tap convolution; the anchor's predecessor is zero."""
    batch, steps, hidden = x.shape
    grouped = x.view(batch, steps, hidden // 16, 16)
    previous = F.pad(grouped[:, :-1], (0, 0, 0, 0, 1, 0))
    output = torch.zeros_like(grouped)
    for tap, values in enumerate((grouped, previous)):
        output = output + base[tap].view(1, 1, hidden // 16, 16) * values
        output = torch.addcmul(output, coefficients[:, :, tap, :, None], values)
    return output.flatten(-2)


def convolve(x, coefficients, base):
    if x.is_cuda and x.dtype == coefficients.dtype == base.dtype == torch.bfloat16:
        from sharar.kernels.draft_conv import convolve_bf16
        return convolve_bf16(x, coefficients, base)
    return convolve_reference(x, coefficients, base)


class ParallelDraftModel:
    def __init__(self, snapshot, target, max_context, *, precision='bf16'):
        if precision not in ('bf16', 'fp8'):
            raise ValueError('unsupported draft precision')
        self.precision, self.fp8_weights = precision, {}
        self.cfg = json.loads((snapshot / 'config.json').read_text())
        self.source = str(snapshot)  # identifies a non-pinned checkpoint in cache identities
        cfg = self.cfg
        self.kind = cfg['architectures'][0]
        if self.kind != 'DFlash2DraftModel':
            raise ValueError('unsupported draft architecture')
        # The geometry is the checkpoint's (#49): hidden width, heads, KV heads, head size and layers.
        self.hidden, self.heads, self.kv_heads, self.head_dim = (cfg['hidden_size'], cfg['num_attention_heads'],
                                                                 cfg['num_key_value_heads'], cfg['head_dim'])
        self.layers = cfg['num_hidden_layers']
        if self.hidden != target.embed.shape[1]:
            raise ValueError(f"the draft's hidden width {self.hidden} is not the target's embedding width {target.embed.shape[1]} (the draft reads the target's embeddings)")
        self.target, self.device = target, target.device
        from sharar.proposers.draft_head import TargetHead
        self.draft_head = TargetHead(lambda x: self.target.logits(x))  # the vocabulary projection the selector reads (#45); resolved per call
        self.features = tuple(cfg['dflash_config']['target_layer_ids'])
        self.window = cfg['sliding_window']
        self.cos, self.sin = frequencies(cfg, max_context + 32, self.device)
        self.weights = {name: weight.to(self.device) for name, weight in self.read_weights(snapshot, cfg).items()}
        if precision == 'fp8':
            from sharar.model.draft_linear import DraftFp8Linear
            for name in list(self.weights):
                if self.is_fp8_name(name):
                    self.fp8_weights[name] = DraftFp8Linear.from_bf16(self.weights.pop(name))
            from sharar.kernels import w3
            if w3.enabled():  # the 3-bit decode copies cover the FP8 draft as well
                w3.attach_draft(self)

    @staticmethod
    def is_fp8_name(name):
        return name.endswith(('self_attn.q_proj.weight', 'self_attn.o_proj.weight', 'mlp.gate_up', 'mlp.down_proj.weight'))

    @staticmethod
    def read_weights(snapshot, cfg):
        """The checkpoint's tensors on the CPU with the serving fusion applied (k|v, gate|up)."""
        with safe_open(str(snapshot / 'model.safetensors'), framework='pt', device='cpu') as source:
            # clone() reads the mmap view into RAM first: copying the view to the
            # device page-faults the file through DMA staging at ~200 MB/s on GB10
            # (19 s for this draft); memcpy plus a device copy takes about a second.
            weights = {key: source.get_tensor(key).clone() for key in source.keys()}
        for layer in range(cfg['num_hidden_layers']):
            prefix = f'layers.{layer}.'
            for output, members in [('self_attn.kv', ('self_attn.k_proj.weight', 'self_attn.v_proj.weight')),
                                    ('mlp.gate_up', ('mlp.gate_proj.weight', 'mlp.up_proj.weight'))]:
                weights[prefix + output] = torch.cat([weights.pop(prefix + name) for name in members])
        return weights

    def _linear(self, x, name):
        if name in self.fp8_weights:
            return self.fp8_weights[name](x)
        return x @ self.weights[name].T

    @property
    def bytes(self):
        return (sum(t.numel() * t.element_size() for t in (*self.weights.values(), self.cos, self.sin))
                + sum(w.bytes for w in self.fp8_weights.values()))

    @property
    def feature_width(self) -> int:
        """The width of each tapped target layer's rows: what `fc` projects, per tap."""
        return self.weights['fc.weight'].shape[1] // len(self.features)

    def project_features(self, features):
        joined = torch.cat([features[i] for i in self.features], dim=-1)
        return rms(joined @ self.weights['fc.weight'].T, self.weights['hidden_norm.weight'])

    def context_kv(self, hidden, positions, caches, pages):
        for layer, (keys, values) in enumerate(caches):
            p = f'layers.{layer}.self_attn.'
            k, v = (hidden @ self.weights[p + 'kv'].T).chunk(2, dim=-1)
            k = rms(k.view(-1, self.kv_heads, self.head_dim), self.weights[p + 'k_norm.weight'])
            rotate_and_cache(k, v.view(-1, self.kv_heads, self.head_dim), positions, self.cos, self.sin, (keys, values), pages)

    def _prepare_conv(self, hidden, name):
        coefficients = (hidden @ self.weights[name + '.kernel_projection.weight'].T).view(*hidden.shape[:-1], 2, 2, self.hidden // 16)
        return convolve(hidden, coefficients[:, :, 0], self.weights[name + '.base_kernel'][0]), coefficients[:, :, 1]

    def _finish_conv(self, hidden, coefficients, name):
        return convolve(hidden, coefficients, self.weights[name + '.base_kernel'][1])

    def forward(self, tokens, positions, pages, caches, attention):
        """Static noise-block forward. attention is a planned FlashInfer wrapper."""
        batch, steps = tokens.shape
        hidden = F.embedding(tokens, self.target.embed)
        for layer in range(self.layers):
            p = f'layers.{layer}.'
            residual = hidden
            x = rms(hidden, self.weights[p + 'input_layernorm.weight'])
            x, coefficients = self._prepare_conv(x, p + 'attention_conv')
            flat = x.view(-1, self.hidden)
            q = rms(self._linear(flat, p + 'self_attn.q_proj.weight').view(-1, self.heads, self.head_dim), self.weights[p + 'self_attn.q_norm.weight'])
            k, v = (flat @ self.weights[p + 'self_attn.kv'].T).chunk(2, dim=-1)
            k = rms(k.view(-1, self.kv_heads, self.head_dim), self.weights[p + 'self_attn.k_norm.weight'])
            q = rotate(q, positions, self.cos, self.sin)
            rotate_and_cache(k, v.view(-1, self.kv_heads, self.head_dim), positions, self.cos, self.sin, caches[layer], pages)
            attended = attention.run(q, caches[layer]).reshape(batch, steps, self.heads * self.head_dim)
            projected = self._linear(attended, p + 'self_attn.o_proj.weight')
            hidden = residual + self._finish_conv(projected, coefficients, p + 'attention_conv')
            residual = hidden
            x = rms(hidden, self.weights[p + 'post_attention_layernorm.weight'])
            x, coefficients = self._prepare_conv(x, p + 'mlp_conv')
            gate, up = self._linear(x, p + 'mlp.gate_up').chunk(2, dim=-1)
            x = self._linear(F.silu(gate) * up, p + 'mlp.down_proj.weight')
            hidden = residual + self._finish_conv(x, coefficients, p + 'mlp_conv')
        return rms(hidden, self.weights['norm.weight'])

    def selector(self, hidden, anchor):
        """DFlash2's sparse conditional lattice, including predecessor effects."""
        # Keep the reference top-k until a deterministic fused kernel is gated.
        k = self.cfg['dflash_config']['selector_top_k']
        values, candidates = self.draft_head.topk(hidden.reshape(-1, self.hidden), k)
        values, candidates = values.view(*hidden.shape[:2], k), candidates.view(*hidden.shape[:2], k)
        p = 'candidate_selector.'
        gate = hidden @ self.weights[p + 'hidden_projection.weight'].T
        predecessors = torch.cat([anchor[:, None, None].expand(-1, 1, candidates.shape[-1]), candidates[:, :-1]], dim=1)
        a = self.weights[p + 'predecessor_codebook'][predecessors]
        b = self.weights[p + 'successor_codebook'][candidates]
        scores = (a * gate[:, :, None]) @ b.transpose(-1, -2)
        return candidates, scores + values[:, :, None]
