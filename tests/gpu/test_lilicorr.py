"""Published LiLiCorr weights against an independent, unpacked torch reference."""
import gc
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sharar.model.lilicorr import LiLiCorrDraftModel
from sharar.proposers.factory import draft_snapshot
from sharar.target import NEMOTRON_BOLT


@pytest.fixture(scope='module')
def draft():
    snap = draft_snapshot('dflash2', None, NEMOTRON_BOLT)
    if not (snap / 'model.safetensors').is_file():
        pytest.skip('pinned Bolt LiLiCorr snapshot is not cached')
    target = SimpleNamespace(device=torch.device('cuda'), target=NEMOTRON_BOLT,
                              embed=torch.zeros(131072, 4096, dtype=torch.bfloat16, device='cuda'),
                              head_logits=lambda x: None)
    model = LiLiCorrDraftModel(snap, target, 4096)
    yield model
    del model, target
    gc.collect()
    torch.cuda.empty_cache()


def reference_linear(model):
    # Decode E2M1 independently; apply the checkpoint global gain after FP32 accumulation.
    unpacked = {}
    values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device='cuda')
    def linear(x, name):
        if name in model.packed:
            w = model.packed[name]
            if name not in unpacked:
                codes = torch.stack([w.weight & 15, w.weight >> 4], -1).flatten(-2).long()
                unpacked[name] = (values[codes] * w.scales.float().repeat_interleave(16, -1)).bfloat16()
            out = (torch.mm(x.reshape(-1, x.shape[-1]), unpacked[name].T, out_dtype=torch.float32)
                   * w.gain).bfloat16().view(*x.shape[:-1], -1)
        else:
            out = F.linear(x, model.weights[name])
        bias = model.weights.get(name.removesuffix('.weight') + '.bias')
        return out if bias is None else out + bias
    return linear


def norm(x, w):
    return F.rms_norm(x, (x.shape[-1],), w, 1e-6)


def reference_head(m, linear, embeddings, log_probs, hidden, anchor):
    b, slots, topk = log_probs.shape
    rank = torch.arange(topk, device='cuda').float() / (topk - 1)
    f = torch.stack([log_probs, log_probs.exp(), log_probs - log_probs.max(-1, keepdim=True).values,
                     rank.expand_as(log_probs), (rank == 0).float().expand_as(log_probs)], -1)
    w, p = m.weights, 'lilicorr.'
    f = F.layer_norm(f.bfloat16(), (5,), w[p + 'feature_norm.weight'], w[p + 'feature_norm.bias'])
    x = linear(embeddings, p + 'token_proj.weight') + linear(hidden, p + 'pass_hidden_proj.weight').unsqueeze(-2)
    x = x + linear(F.silu(linear(f, p + 'feature_mlp.up_proj.weight')), p + 'feature_mlp.down_proj.weight')
    x = x + w[p + 'slot_embedding'][:, 0, :slots] + w[p + 'rank_embedding'][:, 0]
    x = x.flatten(1, 2)
    pos = torch.arange(slots, device='cuda').repeat_interleave(topk)
    relative = pos[:, None] - pos[None, :]
    bias = w[p + 'relative_slot_bias'][:, relative + m.cfg['block_size'] - 1]
    bias = bias + (relative == 0) * w[p + 'same_slot_bias'][:, None, None]
    for layer in range(2):
        q = p + f'layers.{layer}.'
        normalized = norm(x, w[q + 'attn_norm.weight'])
        # Explicit FP32 attention, independent of the serving SDPA implementation.
        qkv = F.linear(normalized, w[q + 'attn.in_proj_weight'], w[q + 'attn.in_proj_bias'])
        query, key, value = [v.view(b, slots * topk, 8, 128).transpose(1, 2) for v in qkv.chunk(3, -1)]
        scores = query.float() @ key.float().transpose(-1, -2) / 128**.5 + bias.float()
        attended = (scores.softmax(-1) @ value.float()).bfloat16().transpose(1, 2).reshape(b, slots * topk, -1)
        x = x + linear(attended, q + 'attn.out_proj.weight')
        x = x + linear(F.silu(linear(norm(x, w[q + 'mlp_norm.weight']), q + 'mlp.up_proj.weight')), q + 'mlp.down_proj.weight')
    x = norm(x, w[p + 'output_norm.weight']).view(b, slots, topk, -1)
    a = norm(linear(anchor, p + 'context_proj.weight'), w[p + 'anchor_norm.weight'])
    arow = a[:, None, None].expand_as(x)
    f = F.silu(linear(torch.cat([x, arow, x * arow], -1), p + 'factor_input_proj.weight'))
    incoming = F.normalize(linear(f, p + 'in_head.weight'), dim=-1, eps=m.vector_eps)
    outgoing = F.normalize(linear(f, p + 'out_head.weight'), dim=-1, eps=m.vector_eps)
    a = F.normalize(linear(a, p + 'anchor_out_head.weight'), dim=-1, eps=m.vector_eps)
    start = (a[:, None] * incoming[:, 0]).sum(-1)
    edges = outgoing[:, :-1] @ incoming[:, 1:].transpose(-1, -2)
    return torch.cat([start[:, None, None].expand(-1, 1, topk, -1), edges], 1).float() * 8


@torch.inference_mode()
@pytest.mark.parametrize('slots', [1, 3, 7])
def test_correlator_matches_unpacked_reference_and_graphs(draft, slots):
    torch.manual_seed(17)
    b = 2
    embeddings = torch.randn(b, slots, 8, 4096, device='cuda', dtype=torch.bfloat16)
    log_probs = torch.randn(b, slots, 8, device='cuda').log_softmax(-1) - 1
    hidden = torch.randn(b, slots, 4096, device='cuda', dtype=torch.bfloat16)
    anchor = torch.randn(b, 4096, device='cuda', dtype=torch.bfloat16)
    actual = draft.correlate(embeddings, log_probs, hidden, anchor)
    expected = reference_head(draft, reference_linear(draft), embeddings, log_probs, hidden, anchor)
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    assert cosine > .9998, cosine
    torch.testing.assert_close(actual, expected, rtol=.06, atol=.055)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = draft.correlate(embeddings, log_probs, hidden, anchor)
    anchor.add_(.25)
    g.replay()
    torch.testing.assert_close(captured, draft.correlate(embeddings, log_probs, hidden, anchor), rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize('length', [11, 1100])
def test_backbone_attention_keeps_sinks(draft, length):
    import flashinfer
    m = draft
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 2**20, dtype=torch.uint8, device='cuda'), 'NHD')
    q = torch.randn(4, 32, 128, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(length, 1, 2, 128, dtype=torch.bfloat16, device='cuda')
    v = torch.randn_like(k)
    wrapper.plan(torch.tensor([0, 4], dtype=torch.int32), torch.tensor([0, length], dtype=torch.int32),
                 torch.arange(length, dtype=torch.int32), torch.ones(1, dtype=torch.int32),
                 32, 2, 128, 1, causal=False, window_left=1023, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    actual = m.attend(q, (k, v), wrapper, 0)
    keys, values = k[:, 0].repeat_interleave(16, 1), v[:, 0].repeat_interleave(16, 1)
    scores = torch.einsum('qhd,khd->hqk', q.float(), keys.float()) / 128**.5
    allowed = torch.arange(length, device='cuda')[None, :] >= (length - 4 + torch.arange(4, device='cuda')[:, None]) - 1023
    scores.masked_fill_(~allowed, -float('inf'))
    sink = m.weights['layers.0.self_attn.attention_sink_bias'][:, None, None].expand(-1, 4, 1)
    probs = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
    expected = torch.einsum('hqk,khd->qhd', probs, values.float()).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.006)


@torch.inference_mode()
def test_backbone_matches_reference_with_projected_context(draft):
    import flashinfer
    m, prefix, steps = draft, 11, 8
    torch.manual_seed(23)
    m.target.embed[990] = torch.randn(4096, device='cuda', dtype=torch.bfloat16)
    m.target.embed[1001] = torch.randn(4096, device='cuda', dtype=torch.bfloat16)
    tokens = torch.full((1, steps), 990, device='cuda', dtype=torch.int64)
    tokens[:, 0] = 1001
    positions = torch.arange(prefix, prefix + steps, device='cuda')
    context = torch.randn(prefix, 4096, device='cuda', dtype=torch.bfloat16)
    caches = [(torch.empty(prefix + steps, 1, 2, 128, device='cuda', dtype=torch.bfloat16),
               torch.empty(prefix + steps, 1, 2, 128, device='cuda', dtype=torch.bfloat16)) for _ in range(5)]
    m.context_kv(context, torch.arange(prefix, device='cuda'), caches, torch.arange(prefix, device='cuda'))
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 * 2**20, dtype=torch.uint8, device='cuda'), 'NHD')
    wrapper.plan(torch.tensor([0, steps], dtype=torch.int32), torch.tensor([0, prefix + steps], dtype=torch.int32),
                 torch.arange(prefix + steps, dtype=torch.int32), torch.ones(1, dtype=torch.int32),
                 32, 2, 128, 1, causal=False, window_left=1023, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    actual = m.forward(tokens, positions, positions, caches, wrapper)
    linear, w = reference_linear(m), m.weights
    def rotation(x, positions):
        c, s = m.cos[positions].bfloat16()[:, None], m.sin[positions].bfloat16()[:, None]
        left, right = x.chunk(2, -1)
        return torch.cat([left * c - right * s, right * c + left * s], -1)
    def conv(x, p, side, coefficients=None):
        if coefficients is None:
            coefficients = linear(x, p + '.kernel_projection.weight').view(1, steps, 2, 2, 256)
        result = torch.zeros_like(x, dtype=torch.float32)
        for t in range(steps):
            for tap in range(2):
                if t >= tap:
                    coefficient = w[p + '.base_kernel'][side, tap].float() + coefficients[0, t, side, tap].float().repeat_interleave(16)
                    result[0, t] += coefficient * x[0, t - tap].float()
        return result.bfloat16(), coefficients
    hidden = F.embedding(tokens, m.target.embed)
    for layer in range(5):
        p = f'layers.{layer}.'
        x, coefficients = conv(norm(hidden, w[p + 'input_layernorm.weight']), p + 'attention_conv', 0)
        q = norm(linear(x, p + 'self_attn.q_proj.weight').view(steps, 32, 128), w[p + 'self_attn.q_norm.weight'])
        k, v = F.linear(x, w[p + 'self_attn.kv']).chunk(2, -1)
        k = norm(k.view(steps, 2, 128), w[p + 'self_attn.k_norm.weight'])
        ck, cv = F.linear(context, w[p + 'self_attn.kv']).chunk(2, -1)
        ck = norm(ck.view(prefix, 2, 128), w[p + 'self_attn.k_norm.weight'])
        keys = torch.cat([rotation(ck, torch.arange(prefix, device='cuda')), rotation(k, positions)]).repeat_interleave(16, 1)
        values = torch.cat([cv.view(prefix, 2, 128), v.view(steps, 2, 128)]).repeat_interleave(16, 1)
        scores = torch.einsum('qhd,khd->hqk', rotation(q, positions).float(), keys.float()) / 128**.5
        sink = w[p + 'self_attn.attention_sink_bias'][:, None, None].expand(-1, steps, 1)
        probs = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
        attended = torch.einsum('hqk,khd->qhd', probs, values.float()).bfloat16().reshape(1, steps, 4096)
        x, _ = conv(linear(attended, p + 'self_attn.o_proj.weight'), p + 'attention_conv', 1, coefficients)
        hidden = hidden + x
        x, coefficients = conv(norm(hidden, w[p + 'post_attention_layernorm.weight']), p + 'mlp_conv', 0)
        x = linear(F.silu(linear(x, p + 'mlp.gate_proj.weight')) * linear(x, p + 'mlp.up_proj.weight'), p + 'mlp.down_proj.weight')
        x, _ = conv(x, p + 'mlp_conv', 1, coefficients)
        hidden = hidden + x
    expected = norm(hidden, w['norm.weight'])
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()
    assert cosine > .999, cosine
