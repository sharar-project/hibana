"""Layerwise NVFP4 calibration/cost screen of currently-FP8 target projections.

Calibrate on 64 real prefill rows; validate on a later verification block.
Weights are reconstructed from the current checkpoint, not an original BF16
model. This does not establish model quality or enable a serving profile.
"""
import argparse
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
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    report = {'completed': False, 'protocol': {'calibration_rows': 64, 'validation_rows': 16,
              'smooth_alpha': [None, .5], 'weight_scale_factors': [1., 2.],
              'screen_nrmse_limit': .05, 'target_quality_validated': False}, 'cells': []}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=60000, num_slots=24, max_batch=4)
    ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct')
    linears = {f'l{i}.{name}': value for i in (0, 3, 32, 63)
               for name, value in vars(ex.model.layers[i].mixer).items() if isinstance(value, Fp8Linear)}
    names = {id(w): name for name, w in linears.items()}
    captured = {name: {} for name in linears}
    phase = None
    original = ops.fp8_linear
    def record(x, lin, x_q=None):
        result = original(x, lin, x_q=x_q)
        name = names.get(id(lin))
        if name is not None and phase is not None:
            q = x_q if x_q is not None else ops.quant_fp8(x.reshape(-1, x.shape[-1]), lin.input_scale)
            captured[name][phase] = q[:64].clone()
        return result
    ids = json.loads(args.inputs.read_text())['code'][0]
    req = ex.admit(ids, SamplingParams(max_new_tokens=32, ignore_eos=True))
    ops.fp8_linear = record
    try:
        while req.prefill_pos < req.prefill_len:
            count = min(2048, req.prefill_len - req.prefill_pos)
            phase = 'calibration' if req.prefill_pos + count == req.prefill_len else None
            ex.prefill_batch([(req, count)])
        phase = 'validation'
        ex.decode_step([req])
    finally:
        ops.fp8_linear = original
    torch.cuda.synchronize()
    req.cacheable = False
    ex.release(req)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 59999 and ex.state_pool.free_count == 23
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    for name, old in linears.items():
        qcal, qval = captured[name]['calibration'], captured[name]['validation']
        cal = (qcal.float() * old.input_scale).bfloat16()
        val = (qval.float() * old.input_scale).bfloat16()
        expected_cal, expected_val = original(None, old, qcal), original(None, old, qval)
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
        new_fn = lambda: ops.nvfp4_linear((val.float() / smooth).bfloat16(), lin)
        old_fn = lambda: original(val, old)
        new_graph, actual = capture_graph(new_fn)
        old_graph, control = capture_graph(old_fn)
        # Capture reserves output storage; it does not execute the graph.
        new_graph.replay()
        old_graph.replay()
        torch.cuda.synchronize()
        validation = error(actual, expected_val)
        row = {'name': name, 'shape': list(weight.shape), 'calibration_grid': choices, 'selected': choice,
               'validation': validation, 'regenerated_fp8_input_error': error(control, expected_val),
               'fp8_ms': time_graph(old_graph, flush, samples=7),
               'nvfp4_ms_including_smoothing_pack': time_graph(new_graph, flush, samples=7),
               'old_weight_bytes': old.weight.numel(),
               'new_weight_bytes': lin.weight.numel() + lin.weight_scale_swizzled.numel(),
               'passes_layer_screen': validation['finite'] and validation['nrmse'] <= .05}
        report['cells'].append(row)
        save()
        print(json.dumps({k: row[k] for k in ('name', 'validation', 'passes_layer_screen')}), flush=True)
        del new_graph, old_graph
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
