# SPDX-License-Identifier: Apache-2.0
"""Pinned Qwen DSpark/DFlash2 backbone, without Transformers or SGLang.

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


def rms(x, weight, eps=1e-6):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


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


def rotate(x, positions, cos, sin):
    # HF computes rotations with cos/sin cast to the activation dtype.
    c, s = cos[positions].to(x.dtype)[:, None], sin[positions].to(x.dtype)[:, None]
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * c - b * s, b * c + a * s], dim=-1)


def convolve(x, coefficients, base):
    """Block-local two-tap convolution; the anchor's predecessor is zero."""
    batch, steps, hidden = x.shape
    grouped = x.view(batch, steps, hidden // 16, 16)
    previous = F.pad(grouped[:, :-1], (0, 0, 0, 0, 1, 0))
    output = torch.zeros_like(grouped)
    for tap, values in enumerate((grouped, previous)):
        output = output + base[tap].view(1, 1, hidden // 16, 16) * values
        output = torch.addcmul(output, coefficients[:, :, tap, :, None], values)
    return output.flatten(-2)


class ParallelDraftModel:
    def __init__(self, snapshot, target, max_context):
        self.cfg = json.loads((snapshot / 'config.json').read_text())
        cfg = self.cfg
        self.kind = cfg['architectures'][0]
        if self.kind not in ('DSparkDraftModel', 'DFlash2DraftModel'):
            raise ValueError('unsupported draft architecture')
        if (cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim']) != (5120, 32, 8, 128):
            raise ValueError('unvalidated draft dimensions')
        self.target, self.device = target, target.device
        self.features = tuple(cfg['dflash_config']['target_layer_ids'])
        self.window = cfg['sliding_window'] or max_context
        self.cos, self.sin = frequencies(cfg, max_context + 32, self.device)
        with safe_open(str(snapshot / 'model.safetensors'), framework='pt', device='cpu') as source:
            self.weights = {key: source.get_tensor(key).to(self.device) for key in source.keys()}
        # Fuse the equal-input projections once; original storage is released.
        for layer in range(cfg['num_hidden_layers']):
            prefix = f'layers.{layer}.'
            for output, members in [('self_attn.kv', ('self_attn.k_proj.weight', 'self_attn.v_proj.weight')),
                                    ('mlp.gate_up', ('mlp.gate_proj.weight', 'mlp.up_proj.weight'))]:
                self.weights[prefix + output] = torch.cat([self.weights.pop(prefix + name) for name in members])

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (*self.weights.values(), self.cos, self.sin))

    def project_features(self, features):
        joined = torch.cat([features[i] for i in self.features], dim=-1)
        return rms(joined @ self.weights['fc.weight'].T, self.weights['hidden_norm.weight'])

    def context_kv(self, hidden, positions, caches, pages):
        for layer, (keys, values) in enumerate(caches):
            p = f'layers.{layer}.self_attn.'
            k, v = (hidden @ self.weights[p + 'kv'].T).chunk(2, dim=-1)
            k = rms(k.view(-1, 8, 128), self.weights[p + 'k_norm.weight'])
            keys[pages, 0] = rotate(k, positions, self.cos, self.sin)
            values[pages, 0] = v.view(-1, 8, 128)

    def _prepare_conv(self, hidden, name):
        if name + '.base_kernel' not in self.weights:
            return hidden, None
        coefficients = (hidden @ self.weights[name + '.kernel_projection.weight'].T).view(*hidden.shape[:-1], 2, 2, 320)
        return convolve(hidden, coefficients[:, :, 0], self.weights[name + '.base_kernel'][0]), coefficients[:, :, 1]

    def _finish_conv(self, hidden, coefficients, name):
        return convolve(hidden, coefficients, self.weights[name + '.base_kernel'][1]) if coefficients is not None else hidden

    def forward(self, tokens, positions, pages, caches, attention):
        """Static noise-block forward. attention is a planned FlashInfer wrapper."""
        batch, steps = tokens.shape
        hidden = F.embedding(tokens, self.target.embed)
        for layer in range(self.cfg['num_hidden_layers']):
            p = f'layers.{layer}.'
            residual = hidden
            x = rms(hidden, self.weights[p + 'input_layernorm.weight'])
            x, coefficients = self._prepare_conv(x, p + 'attention_conv')
            flat = x.view(-1, 5120)
            q = rms((flat @ self.weights[p + 'self_attn.q_proj.weight'].T).view(-1, 32, 128), self.weights[p + 'self_attn.q_norm.weight'])
            k, v = (flat @ self.weights[p + 'self_attn.kv'].T).chunk(2, dim=-1)
            k = rms(k.view(-1, 8, 128), self.weights[p + 'self_attn.k_norm.weight'])
            q, k = rotate(q, positions, self.cos, self.sin), rotate(k, positions, self.cos, self.sin)
            caches[layer][0][pages, 0] = k
            caches[layer][1][pages, 0] = v.view(-1, 8, 128)
            attended = attention.run(q, caches[layer]).reshape(batch, steps, 4096)
            projected = attended @ self.weights[p + 'self_attn.o_proj.weight'].T
            hidden = residual + self._finish_conv(projected, coefficients, p + 'attention_conv')
            residual = hidden
            x = rms(hidden, self.weights[p + 'post_attention_layernorm.weight'])
            x, coefficients = self._prepare_conv(x, p + 'mlp_conv')
            gate, up = (x @ self.weights[p + 'mlp.gate_up'].T).chunk(2, dim=-1)
            x = (F.silu(gate) * up) @ self.weights[p + 'mlp.down_proj.weight'].T
            hidden = residual + self._finish_conv(x, coefficients, p + 'mlp_conv')
        return rms(hidden, self.weights['norm.weight'])

    def selector(self, hidden, anchor):
        """DFlash2's sparse conditional lattice, including predecessor effects."""
        logits = self.target.logits(hidden.reshape(-1, 5120)).view(*hidden.shape[:2], -1)
        # Keep the reference top-k until a deterministic fused kernel is gated.
        values, candidates = logits.topk(self.cfg['dflash_config']['selector_top_k'], dim=-1)
        p = 'candidate_selector.'
        gate = hidden @ self.weights[p + 'hidden_projection.weight'].T
        predecessors = torch.cat([anchor[:, None, None].expand(-1, 1, candidates.shape[-1]), candidates[:, :-1]], dim=1)
        a = self.weights[p + 'predecessor_codebook'][predecessors]
        b = self.weights[p + 'successor_codebook'][candidates]
        scores = (a * gate[:, :, None]) @ b.transpose(-1, -2)
        return candidates, scores + values[:, :, None]

    def markov(self, logits, previous, hidden):
        latent = self.weights['markov_head.markov_w1.weight'][previous]
        corrected = logits + latent @ self.weights['markov_head.markov_w2.weight'].T
        confidence_input = torch.cat([hidden, latent], dim=-1)
        confidence = torch.sigmoid(confidence_input @ self.weights['confidence_head.proj.weight'].T + self.weights['confidence_head.proj.bias'])
        return corrected, confidence.squeeze(-1)
