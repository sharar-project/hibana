# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LiLiCorr's packed DFlash backbone and learned candidate-lattice scorer.

Equations follow the pinned vLLM implementation in THIRD_PARTY_NOTICES.md.
The parallel adapter owns context rings, graphs and conditional proposal laws;
the target owns verification. Checkpoint NVFP4 matrices remain weight-only.
"""
import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

from hibana.model.parallel_draft import ParallelDraftModel, frequencies, rms
from hibana.model.bolt.linear import PackedLinear, PlanRows, graph_rows, preparing_on_first_use
from hibana.proposers.draft_head import TargetHead
from hibana.model.lilicorr_config import validate_config


def convolve(x, coefficients, base):
    """Grouped two-tap convolution with FP32 coefficient sums and accumulation."""
    b, s, h = x.shape
    groups = x.float().view(b, s, h // 16, 16)
    previous = F.pad(groups[:, :-1], (0, 0, 0, 0, 1, 0))
    c = base.float().view(1, 1, 2, h // 16, 16) + coefficients.float().unsqueeze(-1)
    return (c[:, :, 0] * groups + c[:, :, 1] * previous).flatten(-2).to(x.dtype)


class LiLiCorrDraftModel(ParallelDraftModel):
    uses_context_anchor = True

    def __init__(self, snapshot, target, max_context, *, precision='bf16'):
        if precision != 'bf16':
            raise ValueError('LiLiCorr uses its published W4A16/BF16 weights; draft precision must be bf16')
        self.cfg = json.loads((snapshot / 'config.json').read_text())
        cfg = self.cfg
        self.max_depth = validate_config(cfg, getattr(target, 'target', None))
        self.kind, self.precision, self.source = 'LiLiCorrDraftModel', precision, str(snapshot)
        self.hidden, self.heads, self.kv_heads, self.head_dim = (
            cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'])
        self.layers, self.window = cfg['num_hidden_layers'], cfg['sliding_window']
        self.target, self.device = target, target.device
        if target.embed.shape != (cfg['vocab_size'], self.hidden):
            raise ValueError('LiLiCorr target embedding shape differs from the checkpoint')
        self.features = tuple(cfg['dflash_config']['target_layer_ids'])
        self.mask_token = cfg['dflash_config']['mask_token_id']
        self.draft_head = TargetHead(lambda x: self.target.head_logits(x))
        self.weights, self.packed, self.fp8_weights = {}, {}, {}
        with safe_open(str(snapshot / 'model.safetensors'), framework='pt', device='cpu') as source:
            names = set(source.keys())
            packed = {name.removesuffix('.weight') for name in names
                      if name.endswith('.weight') and source.get_slice(name).get_dtype() == 'U8'}
            for name in sorted(names):
                if any(name == p + suffix for p in packed for suffix in ('.weight', '.weight_scale', '.weight_scale_2')):
                    continue
                self.weights[name] = source.get_tensor(name).clone().to(self.device)
            for p in sorted(packed):
                self.packed[p + '.weight'] = PackedLinear(
                    source.get_tensor(p + '.weight').clone().to(self.device), 'NVFP4',
                    source.get_tensor(p + '.weight_scale').clone().to(self.device),
                    source.get_tensor(p + '.weight_scale_2').clone().to(self.device).float().reshape(()))
        from hibana.engine.config import current
        rows = PlanRows(exact=graph_rows(current()))
        with preparing_on_first_use():
            for lin in self.packed.values():  # b12x compiles on this (the loading) thread, not the engine's
                lin.prepare(rows)
        for layer in range(self.layers):
            p = f'layers.{layer}.self_attn.'
            self.weights[p + 'kv'] = torch.cat([self.weights.pop(p + name + '_proj.weight') for name in ('k', 'v')])
            self.weights[p + 'attention_sink_bias'] = self.weights[p + 'attention_sink_bias'].float()
        rope_cfg = {**cfg, 'rope_parameters': {'rope_type': 'default', 'rope_theta': cfg['rope_theta']}}
        self.cos, self.sin = frequencies(rope_cfg, max_context + 32, self.device)
        d = cfg['dflash_config']
        self.topk, self.correlator_hidden = d['lilicorr_candidate_topk'], d['lilicorr_hidden_size']
        self.correlator_heads, self.factor_dim = d['lilicorr_num_heads'], d['lilicorr_factor_dim']
        self.eps, self.vector_eps = cfg['rms_norm_eps'], d['lilicorr_vector_eps']
        slots = torch.arange(self.max_depth, device=self.device).repeat_interleave(self.topk)
        relative = slots[:, None] - slots[None, :]
        self.attention_bias = (self.weights['lilicorr.relative_slot_bias'][:, relative + cfg['block_size'] - 1]
                               + (relative == 0) * self.weights['lilicorr.same_slot_bias'][:, None, None])

    @property
    def bytes(self):
        return super().bytes + sum(w.bytes for w in self.packed.values()) + self.attention_bias.numel() * self.attention_bias.element_size()

    @property
    def feature_width(self):
        return self.packed['fc.weight'].in_features // len(self.features)

    def _linear(self, x, name):
        if name.endswith('mlp.gate_up'):
            p = name.removesuffix('gate_up')
            return torch.cat([self._linear(x, p + side + '_proj.weight') for side in ('gate', 'up')], -1)
        if name in self.packed:
            y = self.packed[name](x.reshape(-1, x.shape[-1]).contiguous()).view(*x.shape[:-1], -1)
        else:
            y = F.linear(x, self.weights[name])
        bias = self.weights.get(name.removesuffix('.weight') + '.bias')
        return y if bias is None else y + bias

    def project_features(self, features):
        x = self._linear(torch.cat([features[i] for i in self.features], -1), 'fc.weight')
        return rms(x, self.weights['hidden_norm.weight'], self.eps)

    def _prepare_conv(self, hidden, name):
        c = self._linear(hidden, name + '.kernel_projection.weight').view(*hidden.shape[:-1], 2, 2, self.hidden // 16)
        return convolve(hidden, c[:, :, 0], self.weights[name + '.base_kernel'][0]), c[:, :, 1]

    def _finish_conv(self, hidden, coefficients, name):
        return convolve(hidden, coefficients, self.weights[name + '.base_kernel'][1])

    def attend(self, query, cache, attention, layer):
        from hibana.kernels.draft_attention import with_sink
        output, lse = attention.run(query, cache, return_lse=True)
        return with_sink(output, lse, self.weights[f'layers.{layer}.self_attn.attention_sink_bias'])

    def correlate(self, embeddings, log_probs, hidden, anchor_features):
        b, slots, topk = log_probs.shape
        if not 1 <= slots <= self.max_depth or topk != self.topk:
            raise ValueError('draft length exceeds LiLiCorr trained slots')
        rank = torch.arange(topk, device=hidden.device, dtype=torch.float32) / max(topk - 1, 1)
        features = torch.stack([log_probs, log_probs.exp(), log_probs - log_probs.amax(-1, keepdim=True),
                                rank.expand_as(log_probs), (rank == 0).float().expand_as(log_probs)], -1)
        p = 'lilicorr.'
        features = F.layer_norm(features.to(hidden.dtype), (5,), self.weights[p + 'feature_norm.weight'],
                                self.weights[p + 'feature_norm.bias'])
        feature_hidden = self._linear(F.silu(self._linear(features, p + 'feature_mlp.up_proj.weight')),
                                      p + 'feature_mlp.down_proj.weight')
        x = self._linear(embeddings, p + 'token_proj.weight') + self._linear(hidden, p + 'pass_hidden_proj.weight').unsqueeze(-2)
        x = x + feature_hidden
        x = x + self.weights[p + 'slot_embedding'][:, 0, :slots]
        x = x + self.weights[p + 'rank_embedding'][:, 0]
        x = x.flatten(1, 2)
        anchor = self._linear(anchor_features, p + 'context_proj.weight')
        for layer in range(self.cfg['dflash_config']['lilicorr_num_layers']):
            q = p + f'layers.{layer}.'
            normalized = rms(x, self.weights[q + 'attn_norm.weight'], self.eps)
            qkv = F.linear(normalized, self.weights[q + 'attn.in_proj_weight'], self.weights[q + 'attn.in_proj_bias'])
            shape = (b, slots * topk, self.correlator_heads, self.correlator_hidden // self.correlator_heads)
            query, key, value = [part.view(shape).transpose(1, 2) for part in qkv.chunk(3, -1)]
            attended = F.scaled_dot_product_attention(query, key, value,
                attn_mask=self.attention_bias[None, :, :slots * topk, :slots * topk]).transpose(1, 2).reshape(b, slots * topk, -1)
            x = x + self._linear(attended, q + 'attn.out_proj.weight')
            normalized = rms(x, self.weights[q + 'mlp_norm.weight'], self.eps)
            x = x + self._linear(F.silu(self._linear(normalized, q + 'mlp.up_proj.weight')), q + 'mlp.down_proj.weight')
        x = rms(x, self.weights[p + 'output_norm.weight'], self.eps).view(b, slots, topk, -1)
        anchor = rms(anchor, self.weights[p + 'anchor_norm.weight'], self.eps)
        row = anchor[:, None, None].expand_as(x)
        factors = F.silu(self._linear(torch.cat([x, row, x * row], -1), p + 'factor_input_proj.weight'))
        outgoing = F.normalize(self._linear(factors, p + 'out_head.weight'), dim=-1, eps=self.vector_eps)
        incoming = F.normalize(self._linear(factors, p + 'in_head.weight'), dim=-1, eps=self.vector_eps)
        anchor_out = F.normalize(self._linear(anchor, p + 'anchor_out_head.weight'), dim=-1, eps=self.vector_eps)
        start = (anchor_out[:, None] * incoming[:, 0]).sum(-1).float()
        pairs = (outgoing[:, :-1] @ incoming[:, 1:].transpose(-1, -2)).float()
        return torch.cat([start[:, None, None].expand(-1, 1, topk, -1), pairs], 1) * self.cfg['dflash_config']['lilicorr_logit_scale']

    def selector(self, hidden, anchor, anchor_features):
        # The correlator consumes raw log-softmax features, before request sampling transforms.
        logits = self.draft_head.logits(hidden.reshape(-1, self.hidden)).float()
        values, ids = logits.log_softmax(-1).topk(self.topk, -1)
        ids, values = ids.view(*hidden.shape[:2], self.topk), values.view(*hidden.shape[:2], self.topk)
        return ids, self.correlate(F.embedding(ids, self.target.embed), values, hidden, anchor_features)
