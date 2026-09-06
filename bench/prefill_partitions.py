"""Trace fixed-token prefill partitions without changing production dispatch.

Observer copies synchronize: these are numerical diagnostics, never timings.
Raw tensors and JSON belong under SHARAR_RESULTS, outside the repository.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.kernels import ops
from sharar.model.qwen38 import Qwen38
from sharar.parity.cache_trace import difference


@contextmanager
def observe(model, rows):
    records, originals, active = {}, [], [None]
    for name in ('fp8_linear', 'nvfp4_linear'):
        original = getattr(ops, name)
        originals.append((ops, name, original))
        def projection(*args, original=original, name=name, **kwargs):
            out = original(*args, **kwargs)
            if active[0] is not None:
                stages = records[active[0]]
                stages[f'{len(stages)}:{name}'] = out[rows].clone().cpu()
            return out
        setattr(ops, name, projection)
    for layer in model.layers:
        original = layer.forward
        originals.append((layer, 'forward', original))
        def forward(hidden, residual, ctx, original=original, index=layer.index):
            active[0] = index
            records[index] = {}
            h, r = original(hidden, residual, ctx)
            active[0] = None
            records[index]['output'] = (h[rows] + r[rows]).cpu()
            return h, r
        layer.forward = forward
    try:
        yield records
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


@torch.inference_mode()
def projection_probe(model, prompts):
    """Isolate GEMM shape with byte-identical FP8 operands and FP32 reference."""
    layer, records = model.layers[0], {}
    ids = torch.tensor(sum(prompts, []), device=model.device)
    hidden = torch.nn.functional.embedding(ids, model.embed)
    _, xq, _ = layer.input_norm.fused_add_quant(hidden, None, layer.mixer.input_scale, inplace=False)
    lin = layer.mixer.in_qkv
    wide = ops.fp8_linear(None, lin, x_q=xq)
    # b/a use a separate BF16 GEMM; keep its full operands fixed as well.
    normalized, _, _ = layer.input_norm.fused_add_quant(hidden.clone(), None, layer.mixer.input_scale, inplace=False)
    ba_wide = normalized @ layer.mixer.in_ba.T
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        offset = 0
        for i, ids in enumerate(prompts):
            chunk = xq[offset:offset + len(ids)].contiguous()
            narrow = ops.fp8_linear(None, lin, x_q=chunk)
            ba_chunk = normalized[offset:offset + len(ids)].contiguous()
            ba_split = ba_chunk @ layer.mixer.in_ba.T
            ba_truth = ba_chunk.float() @ layer.mixer.in_ba.float().T
            truth = (chunk[-1:].float() * lin.input_scale) @ (lin.weight.float() * lin.weight_scale).T
            a, b = wide[offset + len(ids) - 1:offset + len(ids)], narrow[-1:]
            records[i] = {'wide_m': len(xq), 'split_m': len(ids),
                          'wide_vs_split': difference(a, b),
                          'full_qkv_wide_vs_split': difference(wide[offset:offset + len(ids)], narrow),
                          'ba_wide_vs_split': difference(ba_wide[offset:offset + len(ids)], ba_split),
                          'ba_wide_vs_fp32': difference(ba_wide[offset:offset + len(ids)], ba_truth),
                          'ba_split_vs_fp32': difference(ba_split, ba_truth),
                          'wide_vs_fp32': difference(a, truth), 'split_vs_fp32': difference(b, truth)}
            offset += len(ids)
    finally:
        torch.set_float32_matmul_precision(precision)
    return records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only-paged', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.inputs.read_text())
    prompts = [c['input_ids'] for c in next(c for c in frozen['fixed'] if c['id'] == 'fixed-128-b4')['cases']]
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
    ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
    summary = {'projection_probe': projection_probe(ex.model, prompts), 'runs': []}
    baseline = None
    for paged_single in ((True,) if args.only_paged else (False, True)):
        for groups in ((4,), (1, 3), (2, 2), (3, 1), (1, 1, 1, 1)):
            for repeat in range(2):
                ex.flush_cache()
                reqs = [ex.admit(ids, SamplingParams(max_new_tokens=1, ignore_eos=True)) for ids in prompts]
                traces, offset = [], 0
                for count in groups:
                    selected = reqs[offset:offset + count]
                    batch = ex._prepare_prefill([(r, len(r.prompt_ids)) for r in selected])
                    ctx = batch.context
                    if paged_single and count == 1:
                        ex._plan_paged_prefill(selected[0].pages, len(batch.ids))
                        ctx.paged_prefill_wrapper = ex._paged_prefill
                        # Attention dispatch checks ctx.batched/prefix_len,
                        # not the presence of a planned wrapper alone.
                        ctx.cu_seqlens = torch.tensor([0, len(batch.ids)], dtype=torch.int32, device=ex.device)
                        ctx.seq_slots = torch.tensor([selected[0].slot], dtype=torch.int32, device=ex.device)
                        ctx.seq_prefix_lens, ctx.seq_lens = [0], [len(batch.ids)]
                        ctx.checkpoint_seqs = [0] * len(ctx.checkpoint_positions)
                    rows, end = [], 0
                    for r in selected:
                        end += len(r.prompt_ids)
                        rows.append(end - 1)
                    with observe(ex.model, rows) as stages:
                        out = ex.model.forward(batch.ids, ctx)
                    logits = ex.model.logits(out['norm'][rows]).cpu()
                    # Common M=1 head isolates target hidden-state drift from LM-head shape.
                    common = torch.cat([ex.model.logits(out['norm'][row:row + 1]).cpu() for row in rows])
                    for i, r in enumerate(selected):
                        traces.append({'layers': {lid: {name: v[i] for name, v in vals.items()} for lid, vals in stages.items()},
                                       'logits': logits[i], 'common_head_logits': common[i],
                                       'first_ssm': ex.state_pool.ssm_all[0, r.slot].cpu().clone(),
                                       'first_conv': ex.state_pool.conv_all[0, r.slot].cpu().clone(),
                                       'features': {lid: v[rows[i]].cpu() for lid, v in out['features'].items()}})
                    ex._finish_prefill(batch, out)
                    offset += count
                if baseline is None:
                    baseline = traces
                results = []
                for ref, trace in zip(baseline, traces):
                    layers = {lid: {name: difference(ref['layers'][lid][name], value) for name, value in vals.items()}
                              for lid, vals in trace['layers'].items()}
                    values, ids = torch.topk(trace['logits'].float(), 5)
                    results.append({'top_ids': ids.tolist(), 'top_logits': values.tolist(), 'margin': float(values[0] - values[1]),
                                    'common_head_top_id': int(trace['common_head_logits'].argmax()),
                                    'first_different_layer': next((lid for lid, vals in layers.items() if any(not d['equal'] for d in vals.values())), None),
                                    'layers': layers, 'logits': difference(ref['logits'], trace['logits']),
                                    'first_ssm': difference(ref['first_ssm'], trace['first_ssm']),
                                    'first_conv': difference(ref['first_conv'], trace['first_conv']),
                                    'features': {lid: difference(ref['features'][lid], value) for lid, value in trace['features'].items()}})
                name = f"{'paged' if paged_single else 'native'}-{'-'.join(map(str, groups))}-{repeat}"
                torch.save(traces, args.output / f'{name}.pt')
                summary['runs'].append({'name': name, 'groups': groups, 'paged_single_control': paged_single, 'repeat': repeat,
                                        'requests': results, 'first_ids': [r.output_ids[0] for r in reqs]})
                for r in reqs:
                    r.cacheable = False
                    ex.release(r)
                (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                print(name, summary['runs'][-1]['first_ids'], flush=True)
    ex.flush_cache()
    assert ex.state_pool.free_count == 11 and ex.kv_pool.free_count == 34999


if __name__ == '__main__':
    main()
