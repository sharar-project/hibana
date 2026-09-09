"""Frozen 10k/1k optimization trials with explicit arithmetic controls.

Run inside the exclusive, memory-guarded GPU window. Startup/graph capture is
outside timing. Native timings and HTTP timings are separate measurements.
Profiles change draft arithmetic; --target-residual-sum selects target math
independently. The default uses the current serving arithmetic.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
from functools import partial
import json
from pathlib import Path
import sys
import time

import torch

from bench.benchmark_text_server import flush_cache, run_quality_canaries, stream_generate
from sharar import config
from sharar.engine.sampler import SamplingParams
from sharar.model import parallel_draft
from sharar.tokenizer import SnapshotTokenizer

PROFILES = {'control': ('bf16', False, False), 'fused': ('bf16', True, False),
            'selector': ('bf16', True, True), 'fp8': ('fp8', True, False),
            'fp8_selector': ('fp8', True, True), 'fp8_rowwise': ('fp8', True, False),
            'bf16_f32_math': ('bf16', True, False), 'fp8_f32_math': ('fp8', True, False)}


def configure_target(residual_sum):
    """Process-local reference arithmetic experiment, never a serving default.

    The FP32 variant failed a strict JSON quality canary; it is retained for
    reproducibility and further parity work, not as a promoted optimization.
    Call before constructing the model or capturing graphs.
    """
    if residual_sum == 'native':
        return
    if residual_sum not in ('bf16', 'fp32'):
        raise ValueError('unsupported residual normalization experiment')
    from sharar.model import norm
    from sharar.kernels.fused_norm import gemma_rmsnorm
    from sharar.kernels import fused_norm_nvfp4
    def experimental_norm(x, weight, eps, residual=None, fp8_scale=None, *, verify=False):
        return gemma_rmsnorm(x, weight, eps, residual=residual, fp8_scale=fp8_scale,
                             fp32_residual_sum=residual_sum == 'fp32')
    norm.gemma_rmsnorm = experimental_norm
    fused_norm_nvfp4.gemma_rmsnorm_nvfp4 = partial(
        fused_norm_nvfp4.gemma_rmsnorm_nvfp4, fp32_residual_sum=residual_sum == 'fp32')


def rms_f32(x, weight, eps=1e-6):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def convolve_f32(x, coefficients, base):
    """Higher-precision equation screen; not a claim of SGLang kernel parity."""
    b, s, h = x.shape
    grouped = x.float().view(b, s, h // 16, 16)
    previous = torch.nn.functional.pad(grouped[:, :-1], (0, 0, 0, 0, 1, 0))
    out = (base[0].float().view(1, 1, h // 16, 16) + coefficients[:, :, 0, :, None].float()) * grouped
    out += (base[1].float().view(1, 1, h // 16, 16) + coefficients[:, :, 1, :, None].float()) * previous
    return out.flatten(-2).to(x.dtype)


class RowwiseDraftFp8Linear:
    """Exploratory per-output-channel weights; no serving configuration hook."""
    def __init__(self, weight, scale):
        self.weight, self.scale = weight, scale

    @classmethod
    def from_bf16(cls, weight):
        scale = (weight.float().abs().amax(1, keepdim=True) / 448).clamp_min(1e-12)
        packed = (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return cls(packed, scale.T.contiguous())

    @property
    def bytes(self):
        return self.weight.numel() + self.scale.numel() * self.scale.element_size()

    def __call__(self, x):
        from sharar.kernels.draft_fp8 import quant_dynamic
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        packed, scale = quant_dynamic(flat)
        out = torch._scaled_mm(packed, self.weight.T,
                               scale_a=scale.expand(len(flat), 1).contiguous(), scale_b=self.scale,
                               out_dtype=torch.bfloat16)
        return out.reshape(*x.shape[:-1], self.weight.shape[0])


def freeze(tok, source, output):
    old = json.loads(source.read_text())
    data = {name: old[name] for name in ('code', 'filler')}
    paragraphs = {
        'design': '\n'.join(f'Note {i}: Requests own mutable state; cached prefixes are immutable. '
                            'A cancelled request must release its private reservations. '
                            'Cache eviction must wait for active readers. Optional caches may be reclaimed '
                            'under memory pressure. Streaming clients need bounded queues and orderly shutdown.'
                            for i in range(400)),
        'reasoning': '\n'.join(f'Experiment {i}: workload A has 400 ms setup and 12 ms per operation. '
                               'Workload B has 700 ms setup and 8 ms per operation. Workers share a '
                               'single memory controller; parallel workers do not multiply its bandwidth. '
                               'A cached result avoids setup only when all input identifiers match.'
                               for i in range(400)),
    }
    questions = {
        'design': 'Write a detailed design for a local inference service with cancellation, prefix reuse, '
                  'memory admission and streaming backpressure. Explain failure cases, concrete state '
                  'transitions and tests. Include pseudocode for the ownership protocol.',
        'reasoning': 'Derive the break-even operation count for A and B, compute their costs for 50, 100 '
                     'and 1000 operations, and develop a detailed model that includes shared bandwidth, '
                     'cache hits and queueing. Explain the assumptions and give worked examples.',
    }
    for name, body in paragraphs.items():
        rows = []
        content = tok.encode(body)
        for row in range(4):
            prefix = tok.encode(f'<|im_start|>user\nIndependent review {row}. Reference notes:\n')
            suffix = tok.encode('\n' + questions[name] + '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            rows.append(prefix + content[:10000 - len(prefix) - len(suffix)] + suffix)
        data[name] = rows
    assert all(len(row) == 10000 for rows in data.values() for row in rows)
    output.write_text(json.dumps(data))


def legacy_selector(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ParallelDraftModel')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'selector')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'torch': torch}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['selector']


def configure(profile, baseline_source):
    precision, fused, selector32 = PROFILES[profile]
    if profile == 'fp8_rowwise':
        from sharar.model import draft_linear
        draft_linear.DraftFp8Linear = RowwiseDraftFp8Linear
    if not fused:
        parallel_draft.convolve = parallel_draft.convolve_reference
    if profile.endswith('_f32_math'):
        parallel_draft.rms = rms_f32
        parallel_draft.convolve = convolve_f32
    parallel_draft.ParallelDraftModel.selector = selector_fp32 if selector32 else legacy_selector(baseline_source)
    return precision


def selector_fp32(self, hidden, anchor):
    """Reference-style final score precision, retained as a measured ablation."""
    logits = self.target.logits(hidden.reshape(-1, 5120)).view(*hidden.shape[:2], -1)
    values, candidates = logits.topk(self.cfg['dflash_config']['selector_top_k'], dim=-1)
    p = 'candidate_selector.'
    gate = hidden @ self.weights[p + 'hidden_projection.weight'].T
    predecessors = torch.cat([anchor[:, None, None].expand(-1, 1, candidates.shape[-1]), candidates[:, :-1]], dim=1)
    a = self.weights[p + 'predecessor_codebook'][predecessors]
    b = self.weights[p + 'successor_codebook'][candidates]
    scores = (a * gate[:, :, None]) @ b.transpose(-1, -2)
    return candidates, scores.float() + values[:, :, None].float()


def release(ex, reqs):
    for r in reqs:
        r.cacheable = False
        ex.release(r)
    ex.flush_cache()
    assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
    assert ex.state_pool.free_count == ex.state_pool.num_slots - 1
    assert len(ex.spec.proposer.free_slots) == ex.max_batch


def prepare(ex, ids, budget=1000):
    ex.flush_cache()
    reqs = [ex.admit(row, SamplingParams(max_new_tokens=budget, ignore_eos=True)) for row in ids]
    for r in reqs:
        while r.prefill_pos < r.prefill_len:
            ex.prefill_batch([(r, min(2048, r.prefill_len - r.prefill_pos))])
    torch.cuda.synchronize()
    assert all(r.prefix_len == 0 for r in reqs)
    return reqs


@torch.inference_mode()
def main():
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, default=Path('/proof/frozen/inputs.json'))
    ap.add_argument('--baseline-source', type=Path, default=Path('/proof/frozen/parallel_draft_baseline.py'))
    ap.add_argument('--profile', choices=PROFILES, required=True)
    ap.add_argument('--target-residual-sum', choices=['native', 'bf16', 'fp32'], default='native',
                    help='native uses current serving math; bf16/fp32 are historical norm ablations')
    ap.add_argument('--mode', choices=['native', 'http'], default='native')
    ap.add_argument('--depth', type=int, default=15)
    ap.add_argument('--cases', nargs='+', default=['code'])
    ap.add_argument('--batches', nargs='+', type=int, default=[1])
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--warmups', type=int, default=0)
    ap.add_argument('--quality', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tok = SnapshotTokenizer(config.snapshot_dir())
    if not args.inputs.exists():
        freeze(tok, args.inputs.parent / 'audit-inputs.json', args.inputs)
    raw = args.inputs.read_bytes()
    inputs = json.loads(raw)
    precision = configure(args.profile, args.baseline_source)
    configure_target(args.target_residual_sum)
    report = {'completed': False, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'inputs_sha256': hashlib.sha256(raw).hexdigest(), 'cells': [],
              'source_sha256': {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__).parents[1] / 'sharar').rglob('*.py')}}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=60000, num_slots=24, max_batch=4, log=print)
    ex.enable_spec(args.depth, proposer='dflash2', state_strategy='reconstruct', draft_precision=precision, log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    report['draft_model_bytes'] = ex.spec.proposer.model.bytes
    report['allocated_bytes_after_warmup'] = torch.cuda.memory_allocated()
    if args.mode == 'native':
        for case in args.cases:
            for batch in args.batches:
                for rep in range(-args.warmups, args.repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    reqs = prepare(ex, inputs[case][:batch])
                    prefill = time.perf_counter() - start
                    start = time.perf_counter()
                    rounds = []
                    while active := [r for r in reqs if not r.finished]:
                        before = [len(r.output_ids) for r in active]
                        ex.decode_step(active)
                        rounds.append([len(r.output_ids) - n for r, n in zip(active, before)])
                    torch.cuda.synchronize()
                    decode = time.perf_counter() - start
                    cell = {'case': case, 'batch': batch, 'repetition': rep, 'warmup': rep < 0,
                            'prefill_s': prefill, 'decode_s': decode, 'complete_s': prefill + decode,
                            'decode_tps': batch * 999 / decode, 'rounds': rounds,
                            'output_ids': [r.output_ids[:] for r in reqs],
                            'output_text': [tok.decode(r.output_ids) for r in reqs]}
                    assert all(len(r.output_ids) == 1000 for r in reqs)
                    release(ex, reqs)
                    report['cells'].append(cell)
                    save()
                    print(json.dumps({k: cell[k] for k in ('case', 'batch', 'repetition', 'complete_s', 'decode_tps')}), flush=True)
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from bench.phase7c import http_server
        with http_server(ex, tok, config.snapshot_dir(), {'engine': 'sharar', 'trial': args.profile,
                                                        'draft_precision': precision, 'depth': args.depth}) as url:
            for case in args.cases:
                for batch in args.batches:
                    # B4 measures the main code case; held-out task diversity is B1.
                    if batch > 1 and case != 'code':
                        continue
                    for rep in range(-args.warmups, args.repeats):
                        flush_cache(url)
                        start = time.perf_counter()
                        with ThreadPoolExecutor(max_workers=batch) as pool:
                            futures = [pool.submit(stream_generate, url, ids, 1000, 600, f'{case}-{row}')
                                       for row, ids in enumerate(inputs[case][:batch])]
                            requests = [f.result() for f in futures]
                        wall = time.perf_counter() - start
                        cell = {'case': case, 'batch': batch, 'repetition': rep, 'warmup': rep < 0,
                                'complete_s': wall, 'aggregate_tps': batch * 1000 / wall, 'requests': requests}
                        report['cells'].append(cell)
                        save()
                        assert all(r['success'] and r['completion_tokens'] == 1000 and r['cached_tokens'] == 0 for r in requests), requests
                        print(json.dumps({k: cell[k] for k in ('case', 'batch', 'repetition', 'complete_s', 'aggregate_tps')}), flush=True)
            if args.quality:
                report['quality'] = run_quality_canaries(url, config.MODEL_ID, timeout=120)
                save()
                assert all(r['passed'] for r in report['quality']), report['quality']
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
