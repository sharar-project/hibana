"""Current target-matrix capture, backend screen and Nsight counter replay.

Captures are cloned inside the existing B1 16-row verify graph. Full output
IDs must match an uninstrumented reference. Screen timings are isolated costs,
not generation TPS. Raw tensor artifacts belong outside the repository.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import flashinfer

from bench.nvfp4_cost import capture_graph, time_graph
from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.kernels import ops
from sharar.model.qwen38 import Qwen38
from sharar.weights.modelopt import Fp8Linear, Nvfp4Linear


def linears(model):
    result = {}
    for i, layer in enumerate(model.layers):
        for part in ('mixer', 'mlp'):
            for name, value in vars(getattr(layer, part)).items():
                if isinstance(value, (Fp8Linear, Nvfp4Linear)):
                    result[f'l{i}.{part}.{name}'] = value
    result['vocab'] = model.lm_head
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


@contextmanager
def capture_operands(weights, rows):
    names = {id(w): n for n, w in weights.items()}
    fp4, fp8 = ops.nvfp4_linear, ops.fp8_linear
    def record4(x, lin, prequantized=None):
        m = len(x) if prequantized is None else len(prequantized[0])
        if m != 16 or not torch.cuda.is_current_stream_capturing():
            return fp4(x, lin, prequantized)
        q, sf = prequantized if prequantized is not None else ops.nvfp4_quantize(x, lin.input_scale_inv)
        out = fp4(None, lin, (q, sf))
        rows.append({'name': names[id(lin)], 'format': 'nvfp4', 'q': q.clone(),
                     'sf': sf.clone(), 'reference': out.clone()})
        return out
    def record8(x, lin, x_q=None):
        m = len(x) if x_q is None else len(x_q)
        if m != 16 or not torch.cuda.is_current_stream_capturing():
            return fp8(x, lin, x_q)
        q = x_q if x_q is not None else ops.quant_fp8(x.reshape(-1, x.shape[-1]), lin.input_scale)
        out = fp8(None, lin, q)
        rows.append({'name': names[id(lin)], 'format': 'fp8', 'q': q.clone(), 'reference': out.clone()})
        return out
    ops.nvfp4_linear, ops.fp8_linear = record4, record8
    try:
        yield
    finally:
        ops.nvfp4_linear, ops.fp8_linear = fp4, fp8


def capture(args):
    frozen = json.loads(args.inputs.read_text())
    gold = json.loads(args.reference.read_text())
    model = Qwen38(config.snapshot_dir(), log=print)
    ex = Executor(model, num_pages=60000, num_slots=24, max_batch=4, log=print)
    ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8', log=print)
    rows = []
    with capture_operands(linears(model), rows):
        ex.capture_graphs(log=print)
    ex.warmup(log=print)
    assert len(rows) == 305, (len(rows), [r['name'] for r in rows])
    assert sum(r['format'] == 'fp8' for r in rows) == 176
    report = {'completed': False, 'input_sha256': digest(args.inputs),
              'source_sha256': digest(Path(__file__)), 'target_revision': config.MODEL_REVISION,
              'reference_sha256': digest(args.reference), 'matrices_per_round': len(rows), 'cells': []}
    save(args.output / 'report.json', report)
    for case in ('code', 'design', 'reasoning'):
        ex.flush_cache()
        ids = frozen[case][0] if isinstance(frozen[case][0], list) else frozen[case]
        req = ex.admit(ids, SamplingParams(max_new_tokens=1000, ignore_eos=True))
        while req.prefill_pos < req.prefill_len:
            ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
        for position in (32, 384, 768):
            while len(req.output_ids) < position:
                ex.decode_step([req])
            offset = len(req.output_ids)
            ex.decode_step([req])
            torch.cuda.synchronize()
            captured = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in row.items()} for row in rows]
            name = f'{case}-p{position}.pt'
            torch.save(captured, args.output / name)
            report['cells'].append({'case': case, 'requested_position': position, 'actual_position': offset,
                                    'file': name, 'sha256': digest(args.output / name)})
        while not req.finished:
            ex.decode_step([req])
        matches = req.output_ids == gold[case]
        report.setdefault('outputs', {})[case] = {'ids': req.output_ids[:], 'matches_reference': matches}
        save(args.output / 'report.json', report)
        assert matches, f'instrumented graph changed {case} output IDs'
        req.cacheable = False
        ex.release(req)
        ex.flush_cache()
        assert ex.kv_pool.free_count == 59999 and ex.state_pool.free_count == 23
        print(json.dumps({'captured_case': case, 'output_matches': matches}), flush=True)
    report['completed'] = True
    save(args.output / 'report.json', report)


def apply(row, lin, backend):
    if row['format'] == 'nvfp4':
        return ops._mm_fp4(row['q'], lin.weight.T, row['sf'], lin.weight_scale_swizzled.T,
                           lin.alpha, torch.bfloat16, backend=backend)
    if backend in ('torch', 'torch_fast'):
        return torch._scaled_mm(row['q'], lin.weight.T, scale_a=lin.input_scale.reshape(1, 1),
                                scale_b=lin.weight_scale.reshape(1, 1), out_dtype=torch.bfloat16,
                                use_fast_accum=backend == 'torch_fast')
    return flashinfer.bmm_fp8(row['q'][None], lin.weight.T[None], lin.input_scale, lin.weight_scale,
                              torch.bfloat16, backend=backend)[0]


def error(actual, expected):
    delta = actual.float() - expected.float()
    return {'exact': torch.equal(actual, expected), 'finite': bool(torch.isfinite(actual).all()),
            'nrmse': float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
            'max_abs': float(delta.abs().max()),
            'argmax_equal': torch.equal(actual.argmax(-1), expected.argmax(-1))}


def load_capture(folder, name):
    report = json.loads((folder / 'report.json').read_text())
    assert report['completed']
    expected = next(r['sha256'] for r in report['cells'] if r['file'] == name)
    assert digest(folder / name) == expected
    return torch.load(folder / name, map_location='cuda', weights_only=True)


def pilot_rows(rows, weights):
    chosen, shapes = [], set()
    for row in rows:
        key = (row['format'], tuple(weights[row['name']].weight.shape))
        if key not in shapes or row['name'].startswith(('l32.', 'l63.')):
            chosen.append(row)
            shapes.add(key)
    return chosen


def screen(args):
    weights = linears(Qwen38(config.snapshot_dir(), log=print))
    rows = load_capture(args.captures, 'code-p32.pt')
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    report = {'completed': False, 'source_sha256': digest(Path(__file__)), 'cells': [],
              'protocol': {'input': 'code-p32.pt', 'm': 16, 'warmups': 2, 'abba_blocks': 3,
                           'samples_per_entry': 3, 'flush_bytes': len(flush),
                           'screen': 'Exact output required before a full-family replay; full generation is the promotion gate.'}}
    save(args.output / 'report.json', report)
    for row in pilot_rows(rows, weights):
        lin = weights[row['name']]
        baseline = 'cutlass' if row['format'] == 'nvfp4' else 'torch'
        backends = ('cutlass', 'b12x', 'cudnn') if row['format'] == 'nvfp4' else ('torch', 'torch_fast', 'cublas', 'cudnn', 'cutlass')
        cell = {'name': row['name'], 'format': row['format'], 'weight_shape': list(lin.weight.shape),
                'baseline': baseline, 'backends': {}}
        report['cells'].append(cell)
        graphs = {}
        for backend in backends:
            try:
                fn = lambda backend=backend: apply(row, lin, backend)
                graph, output = capture_graph(fn)
                graph.replay()
                torch.cuda.synchronize()
                cell['backends'][backend] = {'error': error(output, row['reference']), 'timings': []}
                graphs[backend] = graph
            except (RuntimeError, ValueError, AssertionError, NotImplementedError) as exc:
                cell['backends'][backend] = {'unavailable': str(exc)}
                print(row['name'], backend, type(exc).__name__, str(exc)[:160], flush=True)
        for backend, graph in graphs.items():
            if backend == baseline:
                continue
            for block in range(3):
                for mode in (baseline, backend, backend, baseline):
                    timing = time_graph(graphs[mode], flush, samples=3)
                    cell['backends'][mode]['timings'].append({'against': backend, 'block': block, **timing})
        for backend, result in cell['backends'].items():
            if result.get('timings'):
                result['median_ms'] = statistics.median(t['median_ms'] for t in result['timings'])
        save(args.output / 'report.json', report)
        print(json.dumps({'name': row['name'], 'results': {b: {k: v for k, v in r.items() if k != 'timings'}
                                                         for b, r in cell['backends'].items()}}), flush=True)
    report['completed'] = True
    save(args.output / 'report.json', report)


def profile(args):
    weights = linears(Qwen38(config.snapshot_dir(), log=print))
    rows = load_capture(args.captures, 'code-p32.pt')
    selected = [r for r in rows if r['name'] in ('l32.mixer.in_qkv', 'l32.mixer.in_z', 'l32.mixer.out',
                                                'l63.mixer.qkv', 'l32.mlp.gate_up', 'l32.mlp.down', 'vocab')]
    calls = [(r, 'cutlass' if r['format'] == 'nvfp4' else 'torch') for r in selected]
    calls += [(r, 'cublas') for r in selected if r['name'] == 'l63.mixer.qkv']
    for row, backend in calls:
        apply(row, weights[row['name']], backend)
    torch.cuda.synchronize()
    save(args.output / 'calls.json', [{'name': r['name'], 'format': r['format'], 'backend': backend,
          'weight_shape': list(weights[r['name']].weight.shape)} for r, backend in calls])
    torch.cuda.cudart().cudaProfilerStart()
    for row, backend in calls:
        torch.cuda.nvtx.range_push('matrix16')
        torch.cuda.nvtx.range_push(row['name'] + '/' + backend)
        apply(row, weights[row['name']], backend)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def attention_qkv(row, lin):
    return row['format'] == 'fp8' and row['q'].shape == (16, 5120) and lin.weight.shape == (14336, 5120)


def family(args):
    """Validate held-out operands, then time all target FP8 calls in order."""
    weights = linears(Qwen38(config.snapshot_dir(), log=print))
    report = {'completed': False, 'source_sha256': digest(Path(__file__)), 'checks': [], 'timings': []}
    save(args.output / 'report.json', report)
    captures = json.loads((args.captures / 'report.json').read_text())
    for cell in captures['cells']:
        rows = load_capture(args.captures, cell['file'])
        for row in rows:
            lin = weights[row['name']]
            if attention_qkv(row, lin):
                check = {'capture': cell['file'], 'name': row['name'],
                         **error(apply(row, lin, 'cublas'), row['reference'])}
                report['checks'].append(check)
                save(args.output / 'report.json', report)
                assert check['exact'], check
        del rows
    assert len(report['checks']) == 144
    rows = [r for r in load_capture(args.captures, 'code-p32.pt') if r['format'] == 'fp8']
    assert len(rows) == 176
    graphs = {}
    for mode in ('baseline', 'candidate'):
        def run(mode=mode):
            for row in rows:
                lin = weights[row['name']]
                backend = 'cublas' if mode == 'candidate' and attention_qkv(row, lin) else 'torch'
                out = apply(row, lin, backend)
            return out
        graphs[mode], _ = capture_graph(run)
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    for block in range(5):
        for mode in ('baseline', 'candidate', 'candidate', 'baseline'):
            report['timings'].append({'block': block, 'mode': mode,
                                      **time_graph(graphs[mode], flush, samples=5)})
    report['median_ms'] = {mode: statistics.median(r['median_ms'] for r in report['timings'] if r['mode'] == mode)
                           for mode in graphs}
    report['completed'] = True
    save(args.output / 'report.json', report)
    print(json.dumps({'validated_matrices': len(report['checks']), 'family_ms': report['median_ms']}), flush=True)


@contextmanager
def qkv_candidate():
    native = ops.fp8_linear
    def linear(x, lin, x_q=None):
        m = len(x) if x_q is None else len(x_q)
        if m != 16 or lin.weight.shape != (14336, 5120):
            return native(x, lin, x_q)
        q = x_q if x_q is not None else ops.quant_fp8(x.reshape(-1, x.shape[-1]), lin.input_scale)
        return apply({'format': 'fp8', 'q': q}, lin, 'cublas').view(16, lin.out_features)
    ops.fp8_linear = linear
    try:
        yield
    finally:
        ops.fp8_linear = native


@contextmanager
def fp4_mlp_candidate():
    from bench.gb10_fp4_pipeline import Pipeline
    pipeline = Pipeline({'tile_k': 512})
    native = ops.nvfp4_linear
    def linear(x, lin, prequantized=None):
        m = len(x) if prequantized is None else len(prequantized[0])
        if m != 16 or tuple(lin.weight.shape) not in ((34816, 2560), (5120, 8704)):
            return native(x, lin, prequantized)
        q, sf = prequantized if prequantized is not None else ops.nvfp4_quantize(x, lin.input_scale_inv)
        return pipeline({'q': q, 'sf': sf}, lin)
    ops.nvfp4_linear = linear
    try:
        yield
    finally:
        ops.nvfp4_linear = native


def native(args):
    """Alternate two graph sets on one model, preserving complete output IDs."""
    from bench.gb10_optimize import prepare, release
    from sharar.engine.spec_graphs import VerifyGraphs
    frozen = json.loads(args.inputs.read_text())
    reference = json.loads(args.reference.read_text())
    ex = Executor(Qwen38(config.snapshot_dir(), log=print), num_pages=60000, num_slots=24, max_batch=4, log=print)
    ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8', log=print)
    ex.capture_graphs(log=print)
    graphs = {'baseline': ex.spec.verify_graphs}
    with ExitStack() as stack:
        stack.enter_context(qkv_candidate())
        if args.fp4_mlp:
            stack.enter_context(fp4_mlp_candidate())
        graphs['candidate'] = VerifyGraphs(ex.spec)
        graphs['candidate'].capture(log=print)
    ex.warmup(log=print)
    report = {'completed': False, 'source_sha256': digest(Path(__file__)), 'input_sha256': digest(args.inputs),
              'reference_sha256': digest(args.reference), 'fp4_mlp': args.fp4_mlp, 'cells': [],
              'protocol': 'One model, baseline/candidate CUDA graphs, cold prefix, 10k/1k, ABBA order; wall timing.'}
    save(args.output / 'report.json', report)
    controls = {}
    for case, batch in [('code', 1), ('design', 1), ('reasoning', 1), ('code', 4)]:
        if case not in args.cases or batch not in args.batches:
            continue
        # Warm the complete shape and both graph sets before alternating samples.
        for rep, mode in enumerate(('baseline', 'candidate', 'baseline', 'candidate', 'candidate', 'baseline')):
            ex.spec.verify_graphs = graphs[mode]
            ex.spec.width_graphs[15] = graphs[mode]
            ex.spec.reset_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            reqs = prepare(ex, frozen[case][:batch])
            prefill = time.perf_counter() - start
            start = time.perf_counter()
            emitted = []
            while active := [r for r in reqs if not r.finished]:
                before = [len(r.output_ids) for r in active]
                ex.decode_step(active)
                emitted.append([len(r.output_ids) - n for r, n in zip(active, before)])
            torch.cuda.synchronize()
            decode = time.perf_counter() - start
            ids = [r.output_ids[:] for r in reqs]
            key = (case, batch)
            controls.setdefault(key, (ids, emitted))
            matches = (ids, emitted) == controls[key] and (batch != 1 or ids[0] == reference[case])
            cell = {'case': case, 'batch': batch, 'rep': rep, 'warmup': rep < 2, 'mode': mode,
                    'prefill_s': prefill, 'decode_s': decode, 'total_s': prefill + decode,
                    'decode_tps': batch * 999 / decode, 'e2e_tps': batch * 1000 / (prefill + decode),
                    'output_ids': ids, 'emitted_per_round': emitted, 'matches_reference': matches,
                    'stats': ex.spec.stats()}
            report['cells'].append(cell)
            save(args.output / 'report.json', report)
            release(ex, reqs)
            print(json.dumps({k: v for k, v in cell.items() if k not in ('output_ids', 'emitted_per_round', 'stats')}), flush=True)
            assert matches, f'candidate changed outputs or accepted work: {case}, batch {batch}'
    report['completed'] = True
    save(args.output / 'report.json', report)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('capture', 'screen', 'profile', 'family', 'native'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inputs', type=Path)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--captures', type=Path)
    ap.add_argument('--fp4-mlp', action='store_true', help='Native mode: add the K512 FP4 MLP candidate')
    ap.add_argument('--cases', nargs='+', choices=('code', 'design', 'reasoning'), default=['code', 'design', 'reasoning'])
    ap.add_argument('--batches', nargs='+', type=int, choices=(1, 4), default=[1, 4])
    args = ap.parse_args()
    if args.mode in ('capture', 'native') and (not args.inputs or not args.reference):
        ap.error('capture/native requires --inputs and --reference')
    if args.mode not in ('capture', 'native') and not args.captures:
        ap.error('screen/profile/family require --captures')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    {'capture': capture, 'screen': screen, 'profile': profile, 'family': family, 'native': native}[args.mode](args)


if __name__ == '__main__':
    main()
