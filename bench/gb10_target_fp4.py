"""Layerwise NVFP4 calibration/cost screen of currently-FP8 target projections.

Calibrate on code/design prefill and decode; validate on held-out reasoning.
Weights are reconstructed from the current checkpoint, not an original BF16
model. This does not establish model quality or enable a serving profile.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from bench.nvfp4_cost import capture_graph, time_graph
from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.kernels import ops
from sharar.kernels.nvfp4_quant import nvfp4_quantize
from sharar.model.qwen38 import Qwen38
from sharar.weights.modelopt import Fp8Linear, Nvfp4Linear


def error(actual, expected):
    a, b = actual.float(), expected.float()
    return {'nrmse': ((a - b).square().sum() / b.square().sum().clamp_min(1e-30)).sqrt().item(),
            'max_abs': (a - b).abs().max().item(), 'finite': bool(torch.isfinite(a).all())}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, default=Path('/proof/frozen/inputs.json'))
    ap.add_argument('--draft-precision', choices=['bf16', 'fp8'], default='fp8')
    ap.add_argument('--calibration-cases', nargs='+', default=['code', 'design'])
    ap.add_argument('--validation-cases', nargs='+', default=['reasoning'])
    ap.add_argument('--decode-positions', nargs='+', type=int, default=[32, 384, 768])
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    if set(args.calibration_cases) & set(args.validation_cases):
        ap.error('calibration and validation cases must be disjoint')
    if any(p < 1 or p >= 984 for p in args.decode_positions):
        ap.error('decode positions must be in [1, 983]')
    report = {'completed': False, 'protocol': {'calibration_rows_per_prefill_chunk': 16,
              'calibration_cases': args.calibration_cases, 'validation_cases': args.validation_cases,
              'decode_positions': args.decode_positions, 'draft_precision': args.draft_precision,
              'input_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              'smooth_alpha': [None, .5], 'weight_scale_factors': [1., 2.],
              'screen_nrmse_limit': .05, 'target_quality_validated': False,
              'target_revision': config.MODEL_REVISION,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, 'cells': []}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=60000, num_slots=24, max_batch=4)
    torch.manual_seed(42)
    ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision=args.draft_precision)
    linears = {f'l{i}.{name}': value for i in (0, 3, 32, 63)
               for name, value in vars(ex.model.layers[i].mixer).items() if isinstance(value, Fp8Linear)}
    names = {id(w): name for name, w in linears.items()}
    captured = {name: {'calibration': [], 'validation': []} for name in linears}
    phase = None
    original = ops.fp8_linear
    def record(x, lin, x_q=None):
        result = original(x, lin, x_q=x_q)
        name = names.get(id(lin))
        if name is not None and phase is not None:
            q = x_q if x_q is not None else ops.quant_fp8(x.reshape(-1, x.shape[-1]), lin.input_scale)
            # Spread prefill rows through each chunk; retain every verify row.
            selected = q[::max(1, len(q) // 16)][:16].clone()
            captured[name][phase].append(selected)
        return result
    inputs = json.loads(args.inputs.read_text())
    ops.fp8_linear = record
    try:
        for group, cases in [('calibration', args.calibration_cases), ('validation', args.validation_cases)]:
            for case in cases:
                ids = inputs[case][0] if isinstance(inputs[case][0], list) else inputs[case]
                req = ex.admit(ids, SamplingParams(max_new_tokens=1000, ignore_eos=True))
                phase = group if group == 'calibration' else None
                while req.prefill_pos < req.prefill_len:
                    ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
                phase = None
                for position in sorted(set(args.decode_positions)):
                    while not req.finished and len(req.output_ids) < position:
                        ex.decode_step([req])
                    if req.finished:
                        break
                    phase = group
                    ex.decode_step([req])
                    phase = None
                req.cacheable = False
                ex.release(req)
                ex.flush_cache()
                print(json.dumps({'captured_case': case, 'group': group}), flush=True)
    finally:
        ops.fp8_linear = original
    torch.cuda.synchronize()
    assert ex.kv_pool.free_count == 59999 and ex.state_pool.free_count == 23
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    for name, old in linears.items():
        qcal = torch.cat(captured[name]['calibration'])
        qval = torch.cat(captured[name]['validation'])
        cal = (qcal.float() * old.input_scale).bfloat16()
        val = (qval.float() * old.input_scale).bfloat16()
        expected_cal = original(None, old, qcal)
        # Preserve the serving verify shape when assessing held-out numerics.
        expected_blocks = [original(None, old, block) for block in qval.split(16)]
        expected_val = torch.cat(expected_blocks)
        weight = old.weight.float() * old.weight_scale
        choices = []
        best = None
        for alpha in (None, .5):
            smooth = torch.ones(weight.shape[1], device='cuda') if alpha is None else (
                cal.float().abs().amax(0).clamp_min(1e-6).pow(alpha)
                / weight.abs().amax(0).clamp_min(1e-6).pow(1 - alpha)).clamp(.01, 100)
            w = (weight * smooth).bfloat16()
            # Bound every activation representable by the current static FP8 input.
            input_scale = (448 * old.input_scale / smooth.min() / (448 * 6)).reshape(())
            for factor in (1., 2.):
                weight_scale = (w.float().abs().amax() / (448 * 6) * factor).reshape(())
                packed, sf = nvfp4_quantize(w, weight_scale.reciprocal())
                lin = Nvfp4Linear(packed, sf.clone(), input_scale * weight_scale,
                                  input_scale.reciprocal(), w.shape[0], w.shape[1])
                output = ops.nvfp4_linear((cal.float() / smooth).bfloat16(), lin)
                metric = error(output, expected_cal)
                choices.append({'smooth_alpha': alpha, 'weight_scale_factor': factor, 'calibration': metric})
                if metric['finite'] and (best is None or metric['nrmse'] < best[0]):
                    best = (metric['nrmse'], lin, smooth, choices[-1])
        _, lin, smooth, choice = best
        # Validate all held-out positions; time the real B1 16-row shape.
        actual_blocks = [ops.nvfp4_linear((block.float() / smooth).bfloat16(), lin)
                         for block in val.split(16)]
        validation = error(torch.cat(actual_blocks), expected_val)
        blocks = [error(actual, expected) for actual, expected in zip(actual_blocks, expected_blocks)]
        timed_val, timed_qval = val[:16], qval[:16]
        new_fn = lambda: ops.nvfp4_linear((timed_val.float() / smooth).bfloat16(), lin)
        old_fn = lambda: original(timed_val, old)
        new_graph, actual = capture_graph(new_fn)
        old_graph, control = capture_graph(old_fn)
        # Capture reserves output storage; it does not execute the graph.
        new_graph.replay()
        old_graph.replay()
        torch.cuda.synchronize()
        row = {'name': name, 'shape': list(weight.shape), 'calibration_grid': choices, 'selected': choice,
               'calibration_rows': len(cal), 'validation_rows': len(val), 'timing_rows': len(timed_val),
               'validation': validation, 'validation_blocks': blocks,
               'regenerated_fp8_input_error': error(control, original(None, old, timed_qval)),
               'fp8_ms': time_graph(old_graph, flush, samples=7),
               'nvfp4_ms_including_smoothing_pack': time_graph(new_graph, flush, samples=7),
               'old_weight_bytes': old.weight.numel(),
               'new_weight_bytes': lin.weight.numel() + lin.weight_scale_swizzled.numel(),
               'passes_layer_screen': validation['finite'] and all(b['finite'] and b['nrmse'] <= .05 for b in blocks)}
        report['cells'].append(row)
        save()
        print(json.dumps({k: row[k] for k in ('name', 'validation', 'passes_layer_screen')}), flush=True)
        del new_graph, old_graph
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
