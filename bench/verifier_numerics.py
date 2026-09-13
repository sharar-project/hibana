"""Numerical controls for fixed eight-token chains at physical batch one/two.

Hooks and forced row shapes are diagnostic only. Observer copies synchronize;
none of these forwards measure serving performance. Raw tensors stay in results.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch


def unswizzle(scales, rows, groups):
    """Decode the documented 128-row/four-column NVFP4 scale layout."""
    mp, kp = (rows + 127) // 128 * 128, (groups + 3) // 4 * 4
    if scales.numel() != mp * kp:
        raise ValueError('scale extent does not match padded matrix')
    return scales.reshape(1, mp // 128, kp // 4, 32, 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(mp, kp)[:rows, :groups]


def unpack_fp4(packed, scales):
    """FP32 arithmetic oracle for low-nibble-first E2M1 and linear block scales."""
    if packed.ndim != 2 or packed.shape[1] % 8 or scales.shape != (packed.shape[0], packed.shape[1] // 8):
        raise ValueError('expected packed rows and one scale per sixteen values')
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1).long()
    magnitudes = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    return magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1, 1) * scales.float().repeat_interleave(16, -1)


def emission(proposal, targets):
    if len(targets) != len(proposal) + 1:
        raise ValueError('verify needs one correction/bonus row')
    n = next((i for i, token in enumerate(proposal) if token != targets[i]), len(proposal))
    return proposal[:n] + [targets[n]]


def compare_traces(reference, candidate):
    from sharar.parity.cache_trace import difference
    if reference.keys() != candidate.keys():
        raise ValueError('trace stages differ')
    stages = {key: difference(value, candidate[key]) for key, value in reference.items()}
    first = next((key for key, value in stages.items() if not value['equal']), None)
    return {'first_difference': first, 'different_stages': sum(not v['equal'] for v in stages.values()), 'stages': stages}


@contextmanager
def observe(model, mode='native', attention_wrapper=None):
    """Clone common rows before in-place residual writes; restore every hook."""
    from sharar.kernels import ops
    from sharar.kernels.nvfp4_quant import nvfp4_quantize
    records, originals, linears = {}, [], {}
    def put(key, value):
        records[key] = value[:8].detach().clone().cpu()
    def patch(obj, name, value):
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    for layer in model.layers:
        prefix = f'{layer.index:02d}'
        for name in ('in_qkv', 'in_z', 'out', 'qkv', 'o'):
            if hasattr(layer.mixer, name):
                linears[id(getattr(layer.mixer, name))] = prefix + '/' + name
        for name in ('gate_up', 'down'):
            linears[id(getattr(layer.mlp, name))] = prefix + '/' + name
    for kind in ('fp8_linear', 'nvfp4_linear'):
        native = getattr(ops, kind)
        def project(x, lin, *args, native=native, kind=kind, **kwargs):
            name = linears[id(lin)]
            if kind == 'fp8_linear':
                q = kwargs['x_q']
                packed = None
                put(name + '.input', q)
            else:
                packed = kwargs.get('prequantized')
                if packed is None:
                    packed = nvfp4_quantize(x, lin.input_scale_inv)
                    # Keep the original call and quantization path unchanged.
                    put(name + '.input', x)
                put(name + '.packed', packed[0])
                put(name + '.scales', unswizzle(packed[1], packed[0].shape[0], lin.in_features // 16))
            result = native(x, lin, *args, **kwargs)
            split = mode in ('all_linear', 'all_linear_attention') or mode == kind.split('_')[0]
            if split and result.shape[0] > 8:
                head = (native(None, lin, x_q=q[:8].contiguous()) if kind == 'fp8_linear' else
                        native(None, lin, prequantized=(packed[0][:8].contiguous(), packed[1])))
                result = torch.cat((head, result[8:]))
            put(name + '.output', result)
            return result
        patch(ops, kind, project)
    for layer in model.layers:
        prefix, mixer = f'{layer.index:02d}', layer.mixer
        native = layer.forward
        def forward(hidden, residual, ctx, native=native, prefix=prefix):
            put(prefix + '/hidden_in', hidden)
            if residual is not None:
                put(prefix + '/residual_in', residual)
            h, r = native(hidden, residual, ctx)
            put(prefix + '/hidden_out', h)
            put(prefix + '/residual_out', r)
            return h, r
        patch(layer, 'forward', forward)
        if layer.kind == 'linear_attention':
            native = mixer._project_ba
            def ba(x, native=native, mixer=mixer, prefix=prefix):
                put(prefix + '/ba.input', x)
                b, a = native(x)
                if mode in ('ba', 'all_linear', 'all_linear_attention') and len(x) > 8:
                    hb, ha = native(x[:8].contiguous())
                    b, a = torch.cat((hb, b[8:])), torch.cat((ha, a[8:]))
                put(prefix + '/ba.output', torch.cat((b, a), -1))
                return b, a
            patch(mixer, '_project_ba', ba)
            native = mixer._verify_core
            def core(qkv, b, a, ctx, native=native, prefix=prefix, lid=layer.index):
                out = native(qkv, b, a, ctx)
                put(prefix + '/core', out.reshape(qkv.shape[0], -1))
                records[prefix + '/conv_windows'] = ctx.verify_conv_windows[lid][0].clone().cpu()
                for name, tensor in ctx.verify_replay[lid].items():
                    if isinstance(tensor, torch.Tensor):
                        records[prefix + '/' + name] = tensor[0].clone().cpu()
                return out
            patch(mixer, '_verify_core', core)
        else:
            native = mixer._prefill_attention
            def attention(q, k, v, ctx, native=native, prefix=prefix):
                put(prefix + '/attention.q', q)
                put(prefix + '/attention.k', k)
                put(prefix + '/attention.v', v)
                out = native(q, k, v, ctx)
                if mode in ('attention', 'all_linear_attention') and len(q) > 8:
                    head = attention_wrapper.run(q[:8].contiguous(), ctx.kv_pool.cache(int(prefix))).reshape(8, -1)
                    out = torch.cat((head, out[8:]))
                put(prefix + '/attention.output', out)
                return out
            patch(mixer, '_prefill_attention', attention)
    try:
        yield records
    finally:
        for obj, name, native in reversed(originals):
            setattr(obj, name, native)


@torch.inference_mode()
def fixed_probes(model, trace, layer_id):
    """Same quantized operands, M=8/16, and bounded FP32 decoded-weight oracle."""
    from sharar.kernels import ops
    from sharar.parity.cache_trace import difference
    layer = model.layers[layer_id]
    prefix, result = f'{layer_id:02d}', {}
    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        linears = {name: getattr(layer.mixer, name) for name in ('in_qkv', 'in_z', 'out', 'qkv', 'o') if hasattr(layer.mixer, name)}
        for name, lin in linears.items():
            x = trace[prefix + '/' + name + '.input'].to(model.device)
            a = ops.fp8_linear(None, lin, x_q=x)
            b = ops.fp8_linear(None, lin, x_q=torch.cat((x, x)))[:8]
            truth = (x.float() @ lin.weight.float().T) * lin.alpha
            result[name] = {'shape_difference': difference(a, b), 'm8_reference': difference(a, truth),
                            'm16_reference': difference(b, truth), 'm8_rounded_reference': difference(a, truth.bfloat16()),
                            'm16_rounded_reference': difference(b, truth.bfloat16())}
        if layer.kind == 'linear_attention':
            x = trace[prefix + '/ba.input'].to(model.device)
            weight = layer.mixer.in_ba
            a, b = x @ weight.T, (torch.cat((x, x)) @ weight.T)[:8]
            truth = x.float() @ weight.float().T
            # Independent FP64 accumulation on the CPU, bounded to the small b/a projection.
            truth64 = (x.cpu().double() @ weight.cpu().double().T).float()
            result['ba'] = {'shape_difference': difference(a, b), 'm8_reference': difference(a, truth),
                            'm16_reference': difference(b, truth), 'fp32_vs_fp64': difference(truth.cpu(), truth64),
                            'm8_rounded_reference': difference(a.cpu(), truth64.bfloat16()),
                            'm16_rounded_reference': difference(b.cpu(), truth64.bfloat16())}
        for name in ('gate_up', 'down'):
            lin = getattr(layer.mlp, name)
            packed = trace[prefix + '/' + name + '.packed'].to(model.device)
            scales = trace[prefix + '/' + name + '.scales'].to(model.device)
            from sharar.weights.modelopt import swizzle_block_scales
            a = ops.nvfp4_linear(None, lin, prequantized=(packed, swizzle_block_scales(scales)))
            b = ops.nvfp4_linear(None, lin, prequantized=(torch.cat((packed, packed)), swizzle_block_scales(torch.cat((scales, scales)))))[:8]
            x = unpack_fp4(packed, scales)
            ws = unswizzle(lin.weight_scale_swizzled, lin.out_features, lin.in_features // 16)
            truth = torch.cat([(x @ unpack_fp4(lin.weight[i:i + 256], ws[i:i + 256]).T) * lin.alpha
                               for i in range(0, lin.out_features, 256)], -1)
            result[name] = {'shape_difference': difference(a, b), 'm8_reference': difference(a, truth),
                            'm16_reference': difference(b, truth), 'm8_rounded_reference': difference(a, truth.bfloat16()),
                            'm16_rounded_reference': difference(b, truth.bfloat16())}
    finally:
        torch.set_float32_matmul_precision(before)
    return result


def output_record(model, norm, proposal):
    native = model.logits(norm).float().cpu()
    common = model.logits(norm[:8].contiguous()).float().cpu()
    values, ids = native[:8].topk(5, dim=-1)
    return {'native_logits': native[:8], 'common_head_logits': common,
            'argmax': native[:8].argmax(-1).tolist(), 'common_argmax': common.argmax(-1).tolist(),
            'emission': emission(proposal, native[:8].argmax(-1).tolist()),
            'top_ids': ids.tolist(), 'top_logits': values.tolist(), 'norm': norm[:8].clone().cpu()}


@torch.inference_mode()
def verify_case(ex, cell, expected, output, window):
    from bench.two_path_study import capture_lattice, history_hashes, prepare
    req = prepare(ex, cell)
    extra, extra_ids = ex.kv_pool.alloc_with_ids(8)
    try:
        current = {}
        with capture_lattice(ex, current):
            batch = ex.spec.prepare([req])
        assert current == expected['paths'], 'frozen proposal changed'
        inputs = {1: (batch.tokens, batch.positions, batch.pages, batch.slots, batch.kv_pages),
                  2: (torch.cat((batch.tokens, torch.tensor([req.output_ids[-1]] + current['alternate'], device=ex.device))),
                      batch.positions.repeat(2), torch.cat((batch.pages, extra)), batch.slots.repeat(2),
                      [batch.kv_pages[0], torch.cat((req.pages, extra))])}
        assert set(extra_ids).isdisjoint(req.pages.tolist() + req.reservation.ids)
        before = history_hashes(ex, req)
        graph, traces, rows = {}, {}, []
        for b in (1, 2):
            norm, targets, _, _, _ = ex.spec.verify_graphs.run(*inputs[b])
            graph[b] = {'norm': norm[:8].clone().cpu(), 'argmax': targets[:8].cpu().tolist()}
            assert graph[b]['argmax'] == expected['target_ids'][str(b)][:8]
        wrapper = ex.spec.verify_graphs._st[1]['wrapper']
        for repeat in range(2):
            modes = ('native', 'ba', 'fp8', 'nvfp4', 'attention', 'all_linear', 'all_linear_attention')
            for mode in modes:
                for b in ((1, 2) if (repeat + window) % 2 else (2, 1)):
                    ex.spec.verify_graphs.run(*inputs[b])
                    st = ex.spec.verify_graphs._st[b]
                    with observe(ex.model, mode, wrapper) as trace:
                        out = ex.model.forward(st['tokens'], st['ctx'])
                    result = output_record(ex.model, out['norm'], current['base'])
                    torch.save({'trace': trace, 'result': result}, output / f'{cell["context"]}-{mode}-b{b}-r{repeat}.pt')
                    if mode == 'native':
                        assert torch.equal(result['norm'], graph[b]['norm']), 'observer/eager changed graph norm'
                        assert result['argmax'] == graph[b]['argmax']
                    if repeat == 0:
                        traces[(mode, b)] = trace
                    else:
                        assert compare_traces(traces[(mode, b)], trace)['first_difference'] is None
                    row = {'mode': mode, 'batch': b, 'repeat': repeat, **{k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}}
                    rows.append(row)
                    print(json.dumps({'context': cell['context'], **row}), flush=True)
        comparisons = {mode: compare_traces(traces[(mode, 1)], traces[(mode, 2)]) for mode in modes}
        for mode in modes:
            assert compare_traces(traces[('native', 1)], traces[(mode, 1)])['first_difference'] is None, 'M8 self-control changed'
        first = comparisons['native']['first_difference']
        lid = int(first.split('/')[0]) if first is not None else 0
        probes = {i: fixed_probes(ex.model, traces[('native', 1)], i) for i in sorted({0, lid})}
        assert before == history_hashes(ex, req), 'verification changed committed history'
        return {'context': cell['context'], 'rows': rows, 'comparisons': comparisons,
                'fixed_operand_probes': probes, 'history_unchanged': True, 'immutable_history_hashes': before}
    finally:
        ex.kv_pool.free_ids(extra_ids)
        ex.release(req)
        ex.flush_cache()
        assert ex.kv_pool.free_count == 34999 and ex.state_pool.free_count == 11


@contextmanager
def prefill_ba(model, captured, offset, fixed):
    originals = []
    for layer in model.layers:
        if layer.kind != 'linear_attention':
            continue
        mixer, lid = layer.mixer, layer.index
        native = mixer._project_ba
        originals.append((mixer, native))
        def project(x, native=native, lid=lid):
            b, a = native(x)
            if fixed is not None:
                ref = fixed[lid][offset:offset + len(x)].to(x.device)
                b, a = ref.chunk(2, -1)
                b, a = b.contiguous(), a.contiguous()
            captured.setdefault(lid, []).append(torch.cat((b, a), -1).clone().cpu())
            return b, a
        mixer._project_ba = project
    try:
        yield
    finally:
        for mixer, native in originals:
            mixer._project_ba = native


@torch.inference_mode()
def prefill_controls(ex, prompts, output):
    from bench.prefill_partitions import observe as observe_prefill, projection_probe
    from bench.two_path_study import history_hashes
    from sharar.engine.sampler import SamplingParams
    groups_all = ((4,), (1, 3), (2, 2), (3, 1), (1, 1, 1, 1))
    fixed, rows = None, []
    probe = projection_probe(ex.model, prompts)
    for mode in ('native', 'fixed_ba'):
        for groups in groups_all:
            for repeat in range(2):
                ex.flush_cache()
                reqs = [ex.admit(ids, SamplingParams(max_new_tokens=1, ignore_eos=True)) for ids in prompts]
                captured, traces, norms, states = {}, [], [], []
                offset, token_offset = 0, 0
                try:
                    for count in groups:
                        selected = reqs[offset:offset + count]
                        batch = ex._prepare_prefill([(r, len(r.prompt_ids)) for r in selected])
                        ctx = batch.context
                        if count == 1:
                            ex._plan_paged_prefill(selected[0].pages, len(batch.ids))
                            ctx.paged_prefill_wrapper = ex._paged_prefill
                            ctx.cu_seqlens = torch.tensor([0, len(batch.ids)], dtype=torch.int32, device=ex.device)
                            ctx.seq_slots = torch.tensor([selected[0].slot], dtype=torch.int32, device=ex.device)
                            ctx.seq_prefix_lens, ctx.seq_lens = [0], [len(batch.ids)]
                            ctx.checkpoint_seqs = [0] * len(ctx.checkpoint_positions)
                        ends = [sum(len(r.prompt_ids) for r in selected[:i + 1]) - 1 for i in range(count)]
                        with prefill_ba(ex.model, captured, token_offset, fixed if mode == 'fixed_ba' else None):
                            with observe_prefill(ex.model, ends) as stages:
                                out = ex.model.forward(batch.ids, ctx)
                        for i, r in enumerate(selected):
                            traces.append({f'{lid:02d}/{name}': values[i].clone() for lid, layer in stages.items() for name, values in layer.items()})
                            norms.append(out['norm'][ends[i]].clone().cpu())
                        ex._finish_prefill(batch, out)
                        states.extend(history_hashes(ex, r) for r in selected)
                        offset += count
                        token_offset += len(batch.ids)
                    gates = {lid: torch.cat(values) for lid, values in captured.items()}
                    if fixed is None:
                        fixed = gates
                    if mode == 'fixed_ba':
                        assert all(torch.equal(gates[lid], value) for lid, value in fixed.items())
                    common = torch.cat([ex.model.logits(n[None].to(ex.device)).cpu() for n in norms])
                    record = {'mode': mode, 'groups': groups, 'repeat': repeat, 'traces': traces,
                              'norms': torch.stack(norms), 'common_logits': common, 'states': states, 'gates': gates}
                    filename = f'prefill-{mode}-{len(rows)}.pt'
                    torch.save(record, output / filename)
                    row = {'mode': mode, 'groups': groups, 'repeat': repeat, 'file': filename,
                           'common_argmax': common.argmax(-1).tolist(), 'ba_fixed': mode == 'fixed_ba'}
                    rows.append(row)
                    print(json.dumps({'prefill': row}), flush=True)
                finally:
                    for req in reqs:
                        ex.release(req)
                    ex.flush_cache()
                    assert ex.kv_pool.free_count == 34999 and ex.state_pool.free_count == 11
    return {'projection_probe': probe, 'runs': rows}


def main():
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--window', required=True, type=int, choices=(1, 2))
    args = parser.parse_args()
    frozen = json.loads(args.inputs.read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    report = {'window': args.window, 'verification': [], 'prefill': None, 'completed': False}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    with torch.inference_mode():
        ex = Executor(Qwen38(config.snapshot_dir(), log=print), num_pages=35000, num_slots=12)
        ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        for cell in frozen['verify_cells'][::1 if args.window == 1 else -1]:
            expected = next(c for c in frozen['expected_cost'] if c['context'] == cell['context'])
            report['verification'].append(verify_case(ex, cell, expected, args.output, args.window))
            save()
        report['prefill'] = prefill_controls(ex, frozen['prefill_prompts'], args.output)
        report['completed'] = True
        save()


if __name__ == '__main__':
    main()
