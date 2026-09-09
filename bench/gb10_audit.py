"""GB10 native-generation audit: exact inputs, full continuations, separate traces.

Run inside the pinned GPU image, alone on the GPU with the host memory guard.
This is a diagnostic, not a serving benchmark or an answer-quality evaluation.
Raw evidence belongs outside the repository. No acceptance counts are forced.
"""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from bench.benchmark_text_server import FILLER_TEXT
from bench.gb10_profile import trace_summary
from bench.verify_cost import phases, summarize_phases
from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.tokenizer import SnapshotTokenizer
from sharar.weights.modelopt import Fp8Linear, Nvfp4Linear


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inputs(tok, length, batch):
    source = (Path(__file__).parent / 'workloads/request_state_7aebbe6.txt').read_text()
    cases = {}
    for name in ('filler', 'code'):
        rows = []
        for row in range(batch):
            if name == 'filler':
                prefix = tok.encode(f'Unique benchmark request audit-{row}. Continue processing text: ')
                body, suffix = tok.encode(FILLER_TEXT), []
            else:
                prefix = tok.encode(f'<|im_start|>user\nReview task {row}. Here is historical inference-engine source:\n')
                body = tok.encode(source)
                suffix = tok.encode('\nExplain the ownership and rollback invariants in this code. '
                                    'Then write a complete Python implementation of a transactional '
                                    'range allocator with reserve, commit, rollback and release, '
                                    'with a detailed explanation and tests.\n<|im_end|>\n'
                                    '<|im_start|>assistant\n<think>\n\n</think>\n\n')
            size = length - len(prefix) - len(suffix)
            if size <= 0:
                raise ValueError('input length is too short for the task envelope')
            rows.append(prefix + (body * ((size + len(body) - 1) // len(body)))[:size] + suffix)
        cases[name] = rows
    return cases


def inventory(model):
    """Logical unique operand bytes per target traversal, NOT measured traffic."""
    rows = []
    def linear(name, value):
        weight = value.weight
        scale = value.weight_scale_swizzled if isinstance(value, Nvfp4Linear) else value.weight_scale
        rows.append({'name': name, 'format': 'nvfp4' if isinstance(value, Nvfp4Linear) else 'fp8',
                     'shape': [value.out_features, value.in_features if isinstance(value, Nvfp4Linear) else weight.shape[1]],
                     'weight_bytes': weight.numel() * weight.element_size(),
                     'scale_bytes': scale.numel() * scale.element_size()})
    for i, layer in enumerate(model.layers):
        for name, value in vars(layer.mixer).items():
            if isinstance(value, (Fp8Linear, Nvfp4Linear)):
                linear(f'layers.{i}.mixer.{name}', value)
        for name in ('gate_up', 'down'):
            linear(f'layers.{i}.mlp.{name}', getattr(layer.mlp, name))
    linear('lm_head', model.lm_head)
    totals = {fmt: sum(r['weight_bytes'] + r['scale_bytes'] for r in rows if r['format'] == fmt)
              for fmt in ('fp8', 'nvfp4')}
    return {'projections': rows, 'projection_bytes': totals,
            'embedding_allocation_bytes_not_streamed': model.embed.numel() * model.embed.element_size(),
            'limitation': 'Unique matrix and scale operands, excluding small BF16 tensors, activations, KV, state, draft and rereads. Not hardware DRAM counters.'}


def export_trace(prof, path, *, decode):
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    totals, counts = Counter(), Counter()
    for e in trace['traceEvents']:
        if e.get('cat') == 'kernel' and e.get('ph') == 'X':
            totals[e['name']] += e['dur'] / 1000
            counts[e['name']] += 1
    result = {'kernels': [{'name': n, 'ms': ms, 'calls': counts[n]} for n, ms in totals.most_common()]}
    if decode:
        result.update(trace_summary(trace))
    with gzip.open(str(path) + '.gz', 'wt') as out:
        json.dump(trace, out)
    path.unlink()
    return result


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--length', type=int, default=10000)
    ap.add_argument('--output-tokens', type=int, default=1000)
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--max-batch', type=int, default=4)
    ap.add_argument('--depth', type=int, default=15)
    ap.add_argument('--cases', nargs='+', choices=['filler', 'code'], default=['filler', 'code'])
    args = ap.parse_args()
    if args.output_tokens < 512 or any(b not in (1, 2, 3, 4) for b in args.batches):
        ap.error('use at least 512 output tokens and batches in 1..4')
    if args.max_batch not in (1, 2, 3, 4) or args.max_batch < max(args.batches):
        ap.error('max-batch must cover selected batches and be in 1..4')
    args.output.mkdir(parents=True, exist_ok=False)
    tok = SnapshotTokenizer(config.snapshot_dir())
    frozen = inputs(tok, args.length, max(args.batches))
    (args.output / 'inputs.json').write_text(json.dumps(frozen))
    report = {'completed': False, 'inputs_sha256': sha(frozen), 'cells': [],
              'protocol': {**vars(args), 'output': str(args.output), 'chunk_size': 2048,
                           'full_generation_repetitions': 1, 'profile_location': 'after >=256 native output tokens on a fresh continuation',
                           'ignore_eos': True, 'temperature': 0, 'physical_dram_counters': None},
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ('sharar', 'bench') for p in (Path(__file__).parents[1] / folder).rglob('*.py')},
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'target_revision': config.MODEL_REVISION}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    prop = torch.cuda.get_device_properties(0)
    report['hardware'] = {k: getattr(prop, k) for k in ('name', 'major', 'minor', 'multi_processor_count',
        'total_memory', 'L2_cache_size', 'shared_memory_per_block_optin', 'shared_memory_per_multiprocessor',
        'max_threads_per_multi_processor', 'regs_per_multiprocessor', 'memory_bus_width', 'clock_rate')}
    save()
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=max(60000, (args.length + args.output_tokens + 32) * max(args.batches) + 1),
                  num_slots=24, max_batch=args.max_batch, log=print)
    ex.enable_spec(args.depth, proposer='dflash2', state_strategy='reconstruct', log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    report['weights'] = inventory(ex.model)
    report['draft_weight_bytes'] = ex.spec.proposer.model.bytes
    save()

    def prepare(ids):
        ex.flush_cache()
        reqs = [ex.admit(row, SamplingParams(max_new_tokens=args.output_tokens, ignore_eos=True)) for row in ids]
        # Deliberately no cross-request scheduler interleave in the component profile.
        # Each request uses the production single-request prefill partition.
        for req in reqs:
            while req.prefill_pos < req.prefill_len:
                ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
        torch.cuda.synchronize()
        return reqs

    def release(reqs):
        for req in reqs:
            req.cacheable = False
            ex.release(req)
        ex.flush_cache()
        assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
        assert ex.state_pool.free_count == 23

    for case in args.cases:
        for batch in args.batches:
            ids = frozen[case][:batch]
            torch.cuda.synchronize()
            start = time.perf_counter()
            reqs = prepare(ids)
            row = {'case': case, 'batch': batch, 'prefill_all_requests_s': time.perf_counter() - start,
                   'rounds': [], 'cached_tokens': [r.prefix_len for r in reqs]}
            start = time.perf_counter()
            while active := [r for r in reqs if not r.finished]:
                before = [len(r.output_ids) for r in active]
                t = time.perf_counter()
                ex.decode_step(active)
                row['rounds'].append({'host_wall_ms': (time.perf_counter() - t) * 1000,
                                      'batch': len(active), 'positions': before,
                                      'emitted': [len(r.output_ids) - n for r, n in zip(active, before)]})
            torch.cuda.synchronize()
            row['decode_s'] = time.perf_counter() - start
            row['output_ids'] = [r.output_ids[:] for r in reqs]
            row['decode_tps'] = sum(len(r.output_ids) - 1 for r in reqs) / row['decode_s']
            row['output_text'] = [tok.decode(r.output_ids) for r in reqs]
            assert all(len(r.output_ids) == args.output_tokens for r in reqs)
            release(reqs)
            report['cells'].append(row)
            save()
            print(json.dumps({k: row[k] for k in ('case', 'batch', 'prefill_all_requests_s', 'decode_s', 'decode_tps')}), flush=True)

            reqs = prepare(ids)
            while min(len(r.output_ids) for r in reqs) < min(256, args.output_tokens // 2):
                ex.decode_step([r for r in reqs if not r.finished])
            records = defaultdict(list)
            row['phase_round_batches'] = []
            with phases(ex, records):
                for _ in range(8):
                    active = [r for r in reqs if not r.finished]
                    row['phase_round_batches'].append(len(active))
                    ex.decode_step(active)
            torch.cuda.synchronize()
            row['phases'] = summarize_phases(records)
            row['trace_round_batches'] = []
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(4):
                    with torch.profiler.record_function('sharar.decode_step'):
                        active = [r for r in reqs if not r.finished]
                        row['trace_round_batches'].append(len(active))
                        ex.decode_step(active)
                        torch.cuda.synchronize()
            row['trace'] = export_trace(prof, args.output / f'{case}-b{batch}-decode.json', decode=True)
            row['allocated_bytes'] = torch.cuda.memory_allocated()
            row['reserved_bytes'] = torch.cuda.memory_reserved()
            release(reqs)
            save()
    # Prefill is profiled independently; its timings are never used as throughput.
    ex.flush_cache()
    req = ex.admit(frozen['code'][0], SamplingParams(max_new_tokens=1, ignore_eos=True))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        while req.prefill_pos < req.prefill_len:
            with torch.profiler.record_function('sharar.prefill_chunk'):
                ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
        torch.cuda.synchronize()
    report['prefill_trace'] = export_trace(prof, args.output / 'code-prefill.json', decode=False)
    release([req])
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
