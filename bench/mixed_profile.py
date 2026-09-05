"""Projection-sharing feasibility on GB10; this does not implement mixed serving.

Observe frozen HTTP arrivals, then replay real per-layer activations through
separate and packed dense passes. All 64 layers participate in each graph:
the baseline visits all prefill weights before revisiting them for verification.
No physical DRAM counters or end-to-end candidate speedup are inferred.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time


def shape_grid():
    """One prefilling request leaves at most three verification slots."""
    return [{'prefill_rows': p, 'verify_rows': v, 'separate_verify_rows': 8 if v == 8 else 32,
             'budget': budget, 'layout': layout}
            for v in (8, 24) for budget in (128, 512, 2048)
            for layout, p in (('append', budget), ('fill_budget', budget - v))] + [
                {'prefill_rows': 124, 'verify_rows': 24, 'separate_verify_rows': 32,
                 'budget': None, 'layout': 'observed_session_tail'}]


def eligible_requests(requests, chunks, ready):
    prefilling = {id(req) for req, _ in chunks}
    return [r for r in requests if id(r) not in prefilling and not r.finished
            and not r.cancel_requested and not r.released and r.prefill_done and ready(r)]


def summarize_samples(samples):
    separate = statistics.median(s['ms'] for s in samples if s['mode'] == 'separate')
    packed = statistics.median(s['ms'] for s in samples if s['mode'] == 'packed')
    return {'separate_ms': separate, 'packed_ms': packed,
            'saved_ms': separate - packed, 'reduction_fraction': 1 - packed / separate}


BRIEF = {
    'name': 'mixed-projection-v2', 'runtime_change': False,
    'hypothesis': 'One packed target projection pass can amortize weight reads across independent prefill and verification rows on GB10.',
    'shapes': shape_grid(),
    'timing': 'Three reversed-order ABBA repetitions, five graph replays per sample; all 64 layers, separate full prefill then verify sweeps versus packed per-layer sweeps. Includes torch.cat and output split views; no state/attention work.',
    'numerical_gate': 'Every operation and each prefill/verify partition finite with relative RMS error <= 0.01 against separate execution. Diagnostic only: full target oracle, EOS/cache/seeded sampling and quality gates still required.',
    'prototype_gate': 'At least 10% reduction of a named dense family at a prefill size <= 512, passing the numerical gate. Independent confirmation required; this is not an HTTP speed claim.',
    'promotion_gate': 'local-v1 selected completion >= 10% better, protected regression <= 5%, full correctness and quality, equal memory and independent repetitions. No default change from this profile.',
    'limitations': 'Captured activation replay excludes cross-layer dependencies, attention, GDN state, draft generation, logits and commits. Logical weight bytes are not measured DRAM traffic. Three repetitions are a pilot; no tail percentiles or statistical claim.',
    'v2_amendment': 'Account for current B=3 graph padding to B=4; separate verification executes 32 rows while packed uses 24 active rows. Add the observed 124-row session tail at prefix 2048. Gates unchanged. HTTP text removes the same terminal IDs as local-v1 before grading.',
}


@contextmanager
def observe(ex, records):
    """Engine-thread event pairs; resolve only after HTTP has become idle."""
    import torch
    owned, originals = [], []
    original_admit = ex.admit
    def admit(*args, **kwargs):
        req = original_admit(*args, **kwargs)
        if req not in owned:
            owned.append(req)
        return req
    originals.append((ex, 'admit', original_admit))
    ex.admit = admit
    for obj, name in ((ex, 'prefill_batch'), (ex, 'decode_step'),
                      (ex.spec.verify_graphs, 'run')):
        original = getattr(obj, name)
        originals.append((obj, name, original))
        def wrapped(*args, original=original, name=name, obj=obj, **kwargs):
            row = {'phase': name, 'start_s': time.perf_counter()}
            if name == 'prefill_batch':
                row['chunks'] = [{'rid': r.request_id, 'rows': n, 'prefix': r.prefill_pos}
                                 for r, n in args[0]]
                row['eligible_verify_rids'] = [r.request_id for r in eligible_requests(owned, args[0], ex.spec.ready)]
            elif name == 'decode_step':
                row['rids'] = [r.request_id for r in args[0]]
            else:
                row['verify_rows'] = args[0].numel()
                row['executed_verify_rows'] = obj.pick(args[3].numel()) * obj.s_len
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = original(*args, **kwargs)
            end.record()
            row['wall_ms'] = (time.perf_counter() - row['start_s']) * 1000
            records.append((row, begin, end))
            return result
        setattr(obj, name, wrapped)
    try:
        yield
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def mixed_http(ex, tok, frozen, output):
    import torch
    from bench import local_workloads
    from sharar import config
    # Existing standalone benchmark module imports its sibling directly.
    from phase7c import http_server
    cases = {c['id']: c for c in frozen['cases']}
    results = []
    with http_server(ex, tok, config.snapshot_dir(),
                     {'profile': BRIEF['name'], 'graphs': True, 'proposer': 'dflash2', 'k': 7}) as url:
        for repetition in range(4):
            local_workloads.await_idle(url)
            local_workloads.json_request(url + '/flush_cache', {})
            events = []
            with observe(ex, events), ThreadPoolExecutor(max_workers=4) as pool:
                start = time.perf_counter() + .1
                futures = [(cases[item['case']], pool.submit(local_workloads.stream_request, url,
                            cases[item['case']], f'mixed-profile-{repetition}-{item["case"]}',
                            deadline=start + item['delay_s'])) for item in frozen['manifest']['mixed_arrivals']]
                rows = []
                for case, future in futures:
                    row = future.result()
                    row['text'] = tok.decode([t for t in row['output_ids'] if t not in (248046, 248044)])
                    row['quality'] = local_workloads.quality(case, row['text'])
                    rows.append(row)
                local_workloads.await_idle(url)
            torch.cuda.synchronize()
            phases = [{**row, 'start_s': row['start_s'] - start, 'event_ms': begin.elapsed_time(end)}
                      for row, begin, end in events]
            results.append({'repetition': repetition, 'warmup': repetition == 0,
                            'requests': rows, 'phases': sorted(phases, key=lambda row: row['start_s'])})
            (output / 'http.json').write_text(json.dumps(results, indent=2) + '\n')
            if any(r['error'] for r in rows):
                raise RuntimeError('mixed HTTP request failed; see raw results')
            print(f'mixed HTTP repetition {repetition} complete', flush=True)
    return results


def operations(model):
    """Common dense operations only; no sequence-specific state or attention."""
    from sharar.kernels import ops
    result = []
    for layer in model.layers:
        result.append({'name': f'{layer.index}.mlp', 'family': 'mlp', 'owner': layer,
                       'fn': layer.mlp, 'weights': (layer.mlp.gate_up, layer.mlp.down)})
        names = ('qkv', 'o') if layer.kind == 'full_attention' else ('in_qkv', 'in_z', 'out')
        for name in names:
            weight = getattr(layer.mixer, name)
            result.append({'name': f'{layer.index}.{name}', 'family': 'fp8', 'weight': weight,
                           'fn': lambda x, weight=weight: ops.fp8_linear(None, weight, x_q=x),
                           'weights': (weight,)})
    return result


@contextmanager
def capture_inputs(items, captured):
    from sharar.kernels import ops
    fp8 = {id(item['weight']): item['name'] for item in items if item['family'] == 'fp8'}
    original_fp8 = ops.fp8_linear
    def record_fp8(x, lin, x_q=None):
        if id(lin) in fp8:
            if x_q is None:
                raise RuntimeError('target projection unexpectedly quantizes its input internally')
            captured[fp8[id(lin)]] = x_q.clone()
        return original_fp8(x, lin, x_q=x_q)
    ops.fp8_linear = record_fp8
    originals = []
    for item in items:
        if item['family'] == 'mlp':
            layer, original = item['owner'], item['fn']
            originals.append((layer, original))
            def record_mlp(x, original=original, name=item['name']):
                captured[name] = x.clone()
                return original(x)
            layer.mlp = record_mlp
    try:
        yield
    finally:
        ops.fp8_linear = original_fp8
        for layer, original in originals:
            layer.mlp = original


def capture(ex, frozen, items):
    from sharar.engine.sampler import SamplingParams
    cases = {c['id']: c for c in frozen['cases']}
    owned, prefill, verify, tail = [], {}, {}, {}
    cache, graphs = ex.cache, ex.use_graphs
    ex.flush_cache()
    ex.cache, ex.use_graphs = None, False
    try:
        req = ex.admit(cases['session']['input_ids'], SamplingParams(max_new_tokens=256, ignore_eos=True))
        owned.append(req)
        with capture_inputs(items, prefill):
            ex.prefill(req, max_tokens=2048)
        with capture_inputs(items, tail):
            ex.prefill(req)
        req.cacheable = False
        ex.release(req)
        reqs = []
        for name in ('coding', 'chat', 'json'):
            req = ex.admit(cases[name]['input_ids'], SamplingParams(max_new_tokens=256, ignore_eos=True))
            owned.append(req)
            ex.prefill(req)
            reqs.append(req)
        # Replay the graph's exact padded inputs eagerly, before acceptance
        # changes any committed state. Capture includes the actual dummy row.
        # This diagnostic recomputes verify scratch and is never timed.
        ex.use_graphs = True
        graph = ex.spec.verify_graphs
        original = graph.run
        def record_verify(*args, **kwargs):
            result = original(*args, **kwargs)
            st = graph._st[graph.pick(args[3].numel())]
            with capture_inputs(items, verify):
                ex.model.forward(st['tokens'], st['ctx'])
            return result
        graph.run = record_verify
        try:
            ex.decode_step(reqs)
        finally:
            graph.run = original
        if any(set(source) != {i['name'] for i in items} for source in (prefill, verify, tail)):
            raise RuntimeError('incomplete target activation capture')
        if any(x.shape[0] != n for source, n in ((prefill, 2048), (verify, 32), (tail, 124)) for x in source.values()):
            raise RuntimeError('unexpected capture shapes')
        return prefill, verify, tail
    finally:
        for req in owned:
            req.cacheable = False
            ex.release(req)
        ex.cache, ex.use_graphs = cache, graphs


def numerical_check(items, prefill, verify, p, v, separate_v=None):
    import torch
    rows = []
    for item in items:
        xp, xv = prefill[item['name']][:p], verify[item['name']][:v]
        separate = (item['fn'](xp), item['fn'](verify[item['name']][:separate_v or v])[:v])
        packed = item['fn'](torch.cat((xp, xv))).split((p, v))
        for part, reference, actual in zip(('prefill', 'verify'), separate, packed):
            ref, value = reference.float(), actual.float()
            error = value - ref
            relative = (error.square().sum() / ref.square().sum().clamp_min(1e-30)).sqrt().item()
            finite = bool(torch.isfinite(value).all() and torch.isfinite(ref).all())
            rows.append({'operation': item['name'], 'part': part, 'relative_rms': relative,
                         'max_abs': error.abs().max().item(), 'unequal': (actual != reference).sum().item(),
                         'elements': actual.numel(), 'finite': finite})
    return {'passed': all(r['finite'] and r['relative_rms'] <= .01 for r in rows),
            'max_relative_rms': max(r['relative_rms'] for r in rows),
            'unequal_fraction': sum(r['unequal'] for r in rows) / sum(r['elements'] for r in rows),
            'operations': rows}


def paired_graphs(items, prefill, verify, p, v, separate_v):
    import torch
    def separate():
        for source, rows in ((prefill, p), (verify, separate_v)):
            for item in items:
                result = item['fn'](source[item['name']][:rows])
        return result
    def packed():
        for item in items:
            x = torch.cat((prefill[item['name']][:p], verify[item['name']][:v]))
            result = item['fn'](x).split((p, v))
        return result
    graphs, outputs = {}, {}
    for name, fn in (('separate', separate), ('packed', packed)):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = fn()
        graphs[name] = graph
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for repetition in range(3):
        for name in ('separate', 'packed', 'packed', 'separate'):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(5):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples.append({'repetition': repetition, 'mode': name, 'ms': begin.elapsed_time(end) / 5})
    return {'samples': samples, **summarize_samples(samples)}


def main():
    import torch
    from bench import local_workloads
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.tokenizer import SnapshotTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'brief.json').write_text(json.dumps(BRIEF, indent=2) + '\n')
    root = Path(config.__file__).resolve().parents[1]
    frozen = local_workloads.materialize(local_workloads.DEFAULT_MANIFEST, config.snapshot_dir())
    (args.output / 'workloads.json').write_text(json.dumps(frozen, indent=2) + '\n')
    report = {'target': config.MODEL_REVISION, 'brief': BRIEF, 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(), 'sources': {
                  str(p.relative_to(root)): local_workloads.digest(p.read_bytes())
                  for folder in ('sharar', 'bench', 'tests') for p in (root / folder).rglob('*.py')},
              'cells': [], 'physical_dram_bytes': None}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with torch.inference_mode():
        ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
        ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        tok = SnapshotTokenizer(config.snapshot_dir())
        report['http'] = mixed_http(ex, tok, frozen, args.output)
        save()
        items = operations(ex.model)
        prefill, verify, tail = capture(ex, frozen, items)
        report['capture'] = {'prefill': 'First 2048 rows of cold local-v1 session; smaller shapes use prefix slices.',
                             'verify': 'Eager replay of exact graph inputs for k=7 B=3 padded to B=4, before acceptance; first 24 rows active, final eight dummy. B=1 probes use the first eight rows.',
                             'tail': 'Final 124 session rows after the first 2048 are committed; paged prefix attention.',
                             'bytes': sum(x.numel() * x.element_size() for source in (prefill, verify, tail) for x in source.values())}
        report['families'] = {}
        for family in ('mlp', 'fp8'):
            selected = [item for item in items if item['family'] == family]
            weights = {id(t): t for item in selected for weight in item['weights']
                       for t in vars(weight).values() if isinstance(t, torch.Tensor)}
            report['families'][family] = {'operations': len(selected),
                                         'logical_weight_bytes': sum(t.numel() * t.element_size() for t in weights.values())}
            for shape in shape_grid():
                p, v = shape['prefill_rows'], shape['verify_rows']
                source = tail if shape['layout'] == 'observed_session_tail' else prefill
                check = numerical_check(selected, source, verify, p, v, shape['separate_verify_rows'])
                timing = paired_graphs(selected, source, verify, p, v, shape['separate_verify_rows'])
                row = {**shape, 'family': family, 'numerical': check, 'timing': timing}
                report['cells'].append(row)
                save()
                print(json.dumps({**shape, 'family': family, 'numerical_passed': check['passed'],
                                  **{k: value for k, value in timing.items() if k != 'samples'}}), flush=True)
                ex.ledger.check_floor()
        report['completed'] = True
        save()


if __name__ == '__main__':
    main()
