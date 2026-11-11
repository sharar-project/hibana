"""Locate cold/split/cache drift at an identical processed token prefix.

Opt-in diagnostic: hooks clone only the last row of each layer. Measurements
from these forwards are correctness evidence, never performance samples.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch

from hibana import config
from hibana.engine.sampler import SamplingParams


def difference(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    delta = (a - b).abs()
    return {'equal': torch.equal(a, b), 'max_abs': delta.max().item(),
            'rms': delta.square().mean().sqrt().item(),
            'relative_rms': (delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)).item(),
            'cosine': torch.nn.functional.cosine_similarity(a, b, dim=0).item()}


@torch.inference_mode()
def first_projection_probe(model, ids):
    """Hold quantized operands fixed; compare both shapes to FP32 matmul.

    This dequantizes only the first QKV projection, not the full checkpoint.
    It diagnoses FP8 GEMV versus GEMM and does not change serving dispatch.
    """
    from hibana.kernels import ops
    layer = model.layers[0]
    hidden = torch.nn.functional.embedding(torch.tensor(ids, device=model.device), model.embed)
    _, xq, _ = layer.input_norm.fused_add_quant(hidden, None, layer.mixer.input_scale, inplace=False)
    lin = layer.mixer.in_qkv
    wide = ops.fp8_linear(None, lin, x_q=xq)[-1:]
    narrow = ops.fp8_linear(None, lin, x_q=xq[-1:].contiguous())
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        truth = ((xq[-1:].float() * lin.input_scale) @ (lin.weight.float() * lin.weight_scale).T)
    finally:
        torch.set_float32_matmul_precision(precision)
    return {'wide_m': len(ids), 'narrow_m': 1, 'operands': 'identical FP8 last input row and weights/scales',
            'wide_vs_narrow': difference(wide, narrow),
            'wide_vs_fp32': difference(wide, truth), 'narrow_vs_fp32': difference(narrow, truth),
            'wide_vs_rounded_fp32': difference(wide, truth.bfloat16()),
            'narrow_vs_rounded_fp32': difference(narrow, truth.bfloat16())}


@contextmanager
def last_token_trace(model, position):
    records, originals = {}, []
    active = [None]
    from hibana.kernels import ops
    from hibana.weights.modelopt import Fp8Linear, Nvfp4Linear
    # Stages are keyed by projection, not call order: a prefill wide enough for the fused MLP never runs gate_up alone.
    labels = {id(v): f'{part}.{attr}' for layer in model.layers for part in ('mixer', 'mlp')
              for attr, v in vars(getattr(layer, part)).items() if isinstance(v, (Fp8Linear, Nvfp4Linear))}
    # Preserve the normal kernels and shapes; observe their last output row.
    for name in ('fp8_linear', 'nvfp4_linear'):
        original = getattr(ops, name)
        originals.append((ops, name, original))
        def projection(x, lin, *args, original=original, name=name, **kwargs):
            value = original(x, lin, *args, **kwargs)
            if active[0] is not None:
                stages = records[active[0]]['stages']
                stages[labels.get(id(lin), f'{len(stages)}:{name}')] = value[-1].clone().cpu()
            return value
        setattr(ops, name, projection)
    for layer in model.layers:
        original = layer.forward
        originals.append((layer, 'forward', original))
        def wrapped(hidden, residual, ctx, original=original, index=layer.index):
            # The diagnostic uses one sequence and absolute ordered positions.
            selected = int(ctx.positions[-1]) == position
            if selected:
                incoming = hidden[-1].clone()
                if residual is not None:
                    incoming += residual[-1]
                records[index] = {'input': incoming.cpu(), 'stages': {}}
                active[0] = index
            h, r = original(hidden, residual, ctx)
            active[0] = None
            if selected:
                records[index]['output'] = (h[-1] + r[-1]).cpu()
            return h, r
        layer.forward = wrapped
        if layer.kind == 'linear_attention':
            original = layer.mixer._project
            originals.append((layer.mixer, '_project', original))
            def project(x, *args, original=original, **kwargs):
                values = original(x, *args, **kwargs)
                if active[0] is not None:
                    stages = records[active[0]]['stages']
                    stages['gdn_normalized_input'] = x[-1].clone().cpu()
                    for name, value in zip(('qkv', 'z', 'b', 'a'), values):
                        stages['gdn_' + name] = value[-1].clone().cpu()
                return values
            layer.mixer._project = project
            original = layer.mixer._finish
            originals.append((layer.mixer, '_finish', original))
            def finish(core, z, t, original=original):
                if active[0] is not None:
                    records[active[0]]['stages']['gdn_core'] = core.reshape(t, -1)[-1].clone().cpu()
                return original(core, z, t)
            layer.mixer._finish = finish
    try:
        yield records
    finally:
        for obj, name, original in originals:
            setattr(obj, name, original)


def run(ex, ids, output: Path):
    """Compare copies and rounding separately from kernel-shape differences."""
    assert ex.spec is None and ex.cache is not None
    if ex.state_pool.window is not None:
        raise ValueError("the cache trace compares FP32 states; run it with HIBANA_SSM_STATE=fp32")
    params = SamplingParams(max_new_tokens=1)
    traces, results, owned = {}, {}, []
    def request(tokens):
        req = ex.admit(tokens, params)
        owned.append(req)
        return req
    def forward(name, req, limit=None):
        with last_token_trace(ex.model, len(ids) - 1) as trace:
            ex.prefill(req, limit)
        if trace:
            traces[name] = trace
            results[name] = {'first_token': req.output_ids[-1], 'cache_prefix': req.prefix_len,
                             'committed_len': req.committed_len}
    try:
        ex.flush_cache()
        # Every run but the last starts cold: all are admitted before the cold run's checkpoint puts its snapshot in
        # the tree, and none of the others is cached itself. The final prompt snapshot cannot serve its own last token:
        # the boundary is the prompt's last chunk end, its last 64-token grid point before the end.
        cold = request(ids)
        boundary = max(p for p, spare in ex._checkpoint_positions(cold, 0, len(ids)) if not spare)
        live, rounded, restored = request(ids), request(ids), request(ids)
        for req in (live, rounded, restored):
            req.cache_enabled = False

        ex.prefill(live, boundary)
        prefix_ssm = ex.state_pool.ssm_all[:, live.slot].clone()
        prefix_conv = ex.state_pool.conv_all[:, live.slot].clone()
        forward('live_split', live)

        ex.prefill(rounded, boundary)
        ex.state_pool.ssm_all[:, rounded.slot].copy_(prefix_ssm.to(torch.bfloat16).float())
        ex.state_pool.conv_all[:, rounded.slot].copy_(prefix_conv)
        forward('rounded_live_split', rounded)

        forward('cold', cold)
        node = ex.cache.node_at(ids[:boundary])
        assert node is not None and node.snapshot is not None and cold.lock is not None and cold.lock.node is node, \
            "the cold run's last chunk end is in the tree, locked by it"
        checkpoint = node.snapshot.slot
        cached_ssm = ex.state_pool.ssm_all[:, checkpoint].clone()
        cached_conv = ex.state_pool.conv_all[:, checkpoint].clone()
        state_rows = []
        for i, layer in enumerate(ex.state_pool.layer_ids):
            state_rows.append({'layer': layer, 'checkpoint_vs_live_ssm': difference(cached_ssm[i], prefix_ssm[i]),
                               'checkpoint_vs_live_conv': difference(cached_conv[i], prefix_conv[i])})

        # A manual checkpoint restore uses the cold run's exact KV and state,
        # without matching, locking or ownership transfer through the radix.
        for lid in ex.kv_pool.layer_ids:
            ex.kv_pool.k[lid][restored.pages[:boundary]] = ex.kv_pool.k[lid][cold.pages[:boundary]]
            ex.kv_pool.v[lid][restored.pages[:boundary]] = ex.kv_pool.v[lid][cold.pages[:boundary]]
        for rows in ex.kv_pool.page_aux.values():  # a page's other rows (NVFP4 scales, index keys) move with it
            rows[restored.pages[:boundary]] = rows[cold.pages[:boundary]]
        ex.state_pool.ssm_all[:, restored.slot].copy_(cached_ssm)
        ex.state_pool.conv_all[:, restored.slot].copy_(cached_conv)
        ex.state.commit_prefill(restored, boundary)
        forward('manual_restore', restored)
        for req in (live, rounded, restored):
            req.cacheable = False
            ex.release(req)
        ex.release(cold)
        hit = request(ids)
        assert hit.prefix_len == boundary
        assert torch.equal(ex.state_pool.ssm_all[:, hit.slot], cached_ssm)
        assert torch.equal(ex.state_pool.conv_all[:, hit.slot], cached_conv)
        forward('radix_restore', hit)
        comparisons = {}
        for name in traces:
            if name == 'cold':
                continue
            comparisons['cold_vs_' + name] = [
                {'layer': i, 'kind': ex.model.layers[i].kind,
                 'input': difference(traces['cold'][i]['input'], traces[name][i]['input']),
                 'output': difference(traces['cold'][i]['output'], traces[name][i]['output']),
                 'stages': {s: difference(v, traces[name][i]['stages'][s])
                            for s, v in traces['cold'][i]['stages'].items() if s in traces[name][i]['stages']}}
                for i in traces['cold']]
        comparisons['manual_vs_radix'] = [
            {'layer': i, 'output': difference(traces['manual_restore'][i]['output'], traces['radix_restore'][i]['output'])}
            for i in traces['cold']]
        report = {'input_ids': ids, 'input_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                  'boundary': boundary, 'runs': results, 'states': state_rows, 'comparisons': comparisons,
                  'first_projection': first_projection_probe(ex.model, ids)}
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'boundary': boundary, 'runs': results}), flush=True)
        return report
    finally:
        for req in owned:
            req.cacheable = False
            ex.release(req)
        ex.flush_cache()


def main():
    from hibana.engine.executor import Executor
    from hibana.model.qwen38 import Qwen38
    from hibana.tokenizer import SnapshotTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config.check()
    model = Qwen38(config.snapshot_dir())
    torch.cuda.empty_cache()
    ex = Executor(model, num_pages=35000, num_slots=16, log=print)
    ids = SnapshotTokenizer(config.snapshot_dir()).encode('hello ' * 128)
    run(ex, ids, args.output)


if __name__ == '__main__':
    main()
