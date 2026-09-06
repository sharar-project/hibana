"""Bounded narrow-M NVFP4 experiment using the installed SM120 CuTe kernel.

This intentionally uses private FlashInfer interfaces, outside serving. The
direct variant tries a narrower live MMA tile with the existing scale layout.
The transposed variant uses the upstream narrow-N operand-swap path; padding
and the return to contiguous row-major output are included in every timing.
No checkpoint bytes are repacked or expanded. See docs/nvfp4-cost.md for gates.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import cutlass
from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _compile_block_scaled_gemm
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import Sm120B12xBlockScaledDenseGemmKernel
from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm120_b12x as kernel_module

from bench.nvfp4_cost import capture_graph, gemm, tensor_hash, time_graph
from sharar import config
from sharar.weights.modelopt import load_nvfp4_linear
from sharar.weights.safetensors import SnapshotWeights


@contextmanager
def full_scale_block(enabled):
    """Benchmark-only compile patch; restore the private helper on every exit."""
    native = kernel_module.sm120_make_smem_layout_sfa
    def layout(tiled_mma, tile, sf_vec_size, num_stages, **kwargs):
        return native(tiled_mma, (max(64, tile[0]), tile[1], tile[2]),
                      sf_vec_size, num_stages, **kwargs)
    try:
        if enabled:
            kernel_module.sm120_make_smem_layout_sfa = layout
        yield
    finally:
        kernel_module.sm120_make_smem_layout_sfa = native


class NarrowTiles:
    def __init__(self, variant, retain_full_scale_block=False):
        assert variant in ('direct', 'transposed')
        self.variant = variant
        self.retain_full_scale_block = retain_full_scale_block
        self.cache = {}

    def __call__(self, q, sf, lin):
        rows = len(q)
        assert 1 <= rows <= 32
        width = 16 if rows <= 16 else 32
        if self.variant == 'direct':
            a, b, sa, sb = q, lin.weight, sf, lin.weight_scale_swizzled
            tile, swap = (width, 128), False
        else:
            # A row-major BF16 TMA output requires a 16-byte row stride.
            padded = (rows + 7) // 8 * 8
            if padded != rows:
                q = torch.nn.functional.pad(q, (0, 0, 0, padded - rows))
            a, b, sa, sb = lin.weight, q, lin.weight_scale_swizzled, sf
            tile, swap = (128, width), True
        m, n, k = len(a), len(b), a.shape[1] * 2
        sf_m, sf_n, sf_k = (m + 127) // 128, (n + 127) // 128, (k // 16 + 3) // 4
        out = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
        key = (self.variant, self.retain_full_scale_block, tile, sf_m, sf_n, sf_k)
        with full_scale_block(self.retain_full_scale_block):
            compiled, _ = _compile_block_scaled_gemm(
                self.cache, key,
                lambda: Sm120B12xBlockScaledDenseGemmKernel(
                    16, tile, (1, 1), swap_ab=swap, enable_pdl=True),
                ab_cutlass_dtype=cutlass.Uint8, sf_dtype=cutlass.Float8E4M3FN,
                c_cutlass_dtype=cutlass.BFloat16, ab_assumed_align=32,
                cluster_shape_mn=(1, 1), swap_ab=False,
                sf_m=sf_m, sf_n=sf_n, sf_k=sf_k, batch_size=1)
        compiled(a, b, out, sf_m, sf_n, sf_k, sa.data_ptr(), sb.data_ptr(), lin.alpha.reshape(1))
        return out if self.variant == 'direct' else out.T[:rows].contiguous()


def load_linears():
    weights, snapshot = {}, SnapshotWeights(config.snapshot_dir())
    for i in range(64):
        p = f'model.language_model.layers.{i}.mlp'
        weights[f'l{i}.gate_up'] = load_nvfp4_linear(snapshot, [p + '.gate_proj', p + '.up_proj'], 'cuda')
        weights[f'l{i}.down'] = load_nvfp4_linear(snapshot, [p + '.down_proj'], 'cuda')
        snapshot.close()
    weights['vocab'] = load_nvfp4_linear(snapshot, ['lm_head'], 'cuda')
    snapshot.close()
    return weights


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('direct', 'transposed'), required=True)
    parser.add_argument('--retain-full-scale-block', action='store_true',
                        help='refine direct narrow-M compilation while preserving the full scale block')
    parser.add_argument('--cell', action='append', help='optional bounded pilot cell')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.retain_full_scale_block and args.variant != 'direct':
        parser.error('--retain-full-scale-block applies only to --variant direct')
    report = {'variant': args.variant, 'retain_full_scale_block': args.retain_full_scale_block,
              'target_revision': config.MODEL_REVISION,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cells': []}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    weights = load_linears()
    candidate = NarrowTiles(args.variant, args.retain_full_scale_block)
    baseline_report = json.loads((args.inputs / 'report.json').read_text())
    for source in baseline_report['cells']:
        label = f"t{source['context']}-b{source['batch']}"
        if args.cell and label not in args.cell:
            continue
        rows = torch.load(args.inputs / (label + '-packed.pt'), map_location='cuda', weights_only=True)
        assert len(rows) == len(source['calls']) == 130
        row = {'cell': label, 'checks': [], 'timings': {}}
        report['cells'].append(row)
        for (name, q, sf, reference), recorded in zip(rows, source['calls']):
            assert name == recorded['name'] and len(q) == recorded['m']
            assert tensor_hash(q) == recorded['packed_sha256']
            assert tensor_hash(sf) == recorded['scales_sha256']
            assert tensor_hash(reference) == recorded['output_sha256']
            output = candidate(q, sf, weights[name])
            torch.cuda.synchronize()
            delta = (output.float() - reference.float()).abs()
            item = {'name': name, 'm': len(q), 'exact': torch.equal(output, reference),
                    'finite': torch.isfinite(output).all().item(), 'max_abs_error': delta.max().item(),
                    'relative_l2_error': (delta.norm() / reference.float().norm().clamp_min(1e-30)).item(),
                    'scaled_max_error': (delta.max() / reference.float().abs().max().clamp_min(1e-30)).item(),
                    'argmax_equal': torch.equal(output.argmax(-1), reference.argmax(-1))}
            row['checks'].append(item)
        row['numerical_screen_passed'] = all(
            r['finite'] and r['relative_l2_error'] <= .001 and r['scaled_max_error'] <= .01
            and (r['name'] != 'vocab' or r['argmax_equal']) for r in row['checks'])
        save()
        groups = defaultdict(list)
        for p in rows:
            groups[p[0].split('.')[-1]].append(p)
        groups['all'] = rows
        for family, values in groups.items():
            graphs, outputs = {}, {}
            for mode, fn in [('baseline', gemm), ('candidate', candidate)]:
                def run(fn=fn):
                    for name, q, sf, _ in values:
                        output = fn(q, sf, weights[name])
                    return output
                graphs[mode], outputs[mode] = capture_graph(run)
            results = []
            for block in range(3):
                for mode in ('baseline', 'candidate', 'candidate', 'baseline'):
                    results.append({'block': block, 'mode': mode,
                                    **time_graph(graphs[mode], samples=5, repeats=5)})
            medians = {mode: statistics.median(r['median_ms'] for r in results if r['mode'] == mode)
                       for mode in graphs}
            row['timings'][family] = {'samples': results, 'medians_ms': medians,
                                     'improvement': 1 - medians['candidate'] / medians['baseline']}
            save()
            print(label, family, medians, 'numeric', row['numerical_screen_passed'], flush=True)
        del graphs, outputs, groups, rows
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
