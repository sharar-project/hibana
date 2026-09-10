"""Counter-guided SM121 FP4 tile/pipeline screen on captured 16-row operands.

Keep the live M tile at 64: this tests N tiling, K stages and scheduling rather
than repeating the failed narrow-M experiment. Private CuTe APIs stay here.
Measured component costs do not establish generation speedups.
"""
import argparse
import json
from pathlib import Path
import statistics

import cutlass
import torch
from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _compile_block_scaled_gemm
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import DenseGemmKernel

from bench.gb10_matrix import apply, digest, error, linears, load_capture, save
from bench.support.cuda import capture_graph, time_graph
from sharar import config
from sharar.model.qwen38 import Qwen38


VARIANTS = {
    'b12x-default': {},
    'n64': {'n': 64},
    'n64-k256': {'n': 64, 'tile_k': 256},
    'k64': {'tile_k': 64},
    'k256': {'tile_k': 256},
    'k512': {'tile_k': 512},
    'nonpersistent': {'nonpersistent': True},
    'n64-nonpersistent': {'n': 64, 'nonpersistent': True},
}


class OneTileKernel(DenseGemmKernel):
    """Pair upstream's one-tile loop with a grid that covers every N tile.

    The installed private helper ignores direct_one_m_tile_scheduler when
    split_k_slices == 1. Enabling its flag alone would leave output unwritten.
    """
    @staticmethod
    def _compute_grid(c, tile_shape_mnk, max_active_clusters,
                      direct_one_m_tile_scheduler, split_k_slices):
        assert split_k_slices == 1
        params, _ = DenseGemmKernel._compute_grid(c, tile_shape_mnk,
                                                max_active_clusters, False, 1)
        return params, (1, 1, params.problem_shape_ntile_mnl[1])


class Pipeline:
    def __init__(self, options):
        options = dict(options)
        self.tile = (64, options.pop('n', 128))
        self.kernel = OneTileKernel if options.pop('nonpersistent', False) else DenseGemmKernel
        if self.kernel is OneTileKernel:
            options.update(single_work_tile_per_cta=True, direct_one_m_tile_scheduler=True)
        self.options, self.cache = options, {}

    def __call__(self, row, lin):
        q, sf = row['q'], row['sf']
        m, n, k = len(q), len(lin.weight), q.shape[1] * 2
        assert m == 16
        sf_m, sf_n, sf_k = (m + 127) // 128, (n + 127) // 128, (k // 16 + 3) // 4
        out = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
        key = (self.tile, tuple(sorted(self.options.items())), sf_m, sf_n, sf_k)
        compiled, _ = _compile_block_scaled_gemm(
            self.cache, key, lambda: self.kernel(16, self.tile, (1, 1), **self.options),
            ab_cutlass_dtype=cutlass.Uint8, sf_dtype=cutlass.Float8E4M3FN,
            c_cutlass_dtype=cutlass.BFloat16, ab_assumed_align=32,
            cluster_shape_mn=(1, 1), swap_ab=False,
            sf_m=sf_m, sf_n=sf_n, sf_k=sf_k, batch_size=1)
        compiled(q, lin.weight, out, sf_m, sf_n, sf_k, sf.data_ptr(),
                 lin.weight_scale_swizzled.data_ptr(), lin.alpha.reshape(1))
        return out


def family(args, weights):
    candidate = Pipeline({'tile_k': 512})
    report = {'completed': False, 'source_sha256': digest(Path(__file__)),
              'variant': {'tile_k': 512}, 'checks': [], 'timings': []}
    save(args.output / 'report.json', report)
    captures = __import__('json').loads((args.captures / 'report.json').read_text())
    for cell in captures['cells']:
        rows = load_capture(args.captures, cell['file'])
        for row in rows:
            if row['format'] != 'nvfp4' or row['name'] == 'vocab':
                continue
            check = {'capture': cell['file'], 'name': row['name'],
                     **error(candidate(row, weights[row['name']]), row['reference'])}
            report['checks'].append(check)
            if not check['exact']:
                save(args.output / 'report.json', report)
                raise AssertionError(check)
        del rows
        save(args.output / 'report.json', report)
    assert len(report['checks']) == 1152
    rows = [r for r in load_capture(args.captures, 'code-p32.pt')
            if r['format'] == 'nvfp4' and r['name'] != 'vocab']
    graphs = {}
    for mode in ('baseline', 'candidate'):
        def run(mode=mode):
            for row in rows:
                lin = weights[row['name']]
                out = candidate(row, lin) if mode == 'candidate' else apply(row, lin, 'cutlass')
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
    print({'validated_matrices': len(report['checks']), 'family_ms': report['median_ms']}, flush=True)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--family', action='store_true', help='Validate the K512 pilot winner and time all MLP matrices')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    weights = linears(Qwen38(config.snapshot_dir(), log=print))
    if args.family:
        return family(args, weights)
    rows = load_capture(args.captures, 'code-p32.pt')
    chosen = [r for r in rows if r['name'] in ('l32.mlp.gate_up', 'l32.mlp.down', 'vocab')]
    assert len(chosen) == 3
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    candidates = {name: Pipeline(options) for name, options in VARIANTS.items()}
    report = {'completed': False, 'source_sha256': digest(Path(__file__)), 'variants': VARIANTS,
              'kernel_source_sha256': digest(Path(__import__(DenseGemmKernel.__module__, fromlist=['']).__file__)),
              'protocol': 'code-p32 operands; M tile 64; exact output screen; 3 ABBA blocks, 5 flushed replays per entry.',
              'cells': []}
    save(args.output / 'report.json', report)
    for row in chosen:
        lin = weights[row['name']]
        cell = {'name': row['name'], 'weight_shape': list(lin.weight.shape), 'variants': {}}
        report['cells'].append(cell)
        baseline, _ = capture_graph(lambda: apply(row, lin, 'cutlass'))
        for name, fn in candidates.items():
            result = {}
            cell['variants'][name] = result
            try:
                graph, output = capture_graph(lambda: fn(row, lin))
                graph.replay()
                torch.cuda.synchronize()
                result['error'] = error(output, row['reference'])
                result['timings'] = []
                for block in range(3):
                    for mode in ('baseline', 'candidate', 'candidate', 'baseline'):
                        result['timings'].append({'block': block, 'mode': mode,
                            **time_graph(baseline if mode == 'baseline' else graph, flush, samples=5)})
                result['median_ms'] = {mode: statistics.median(r['median_ms'] for r in result['timings'] if r['mode'] == mode)
                                       for mode in ('baseline', 'candidate')}
                del graph, output
            except Exception as exc:
                result['unavailable'] = f'{type(exc).__name__}: {exc}'
                # A failed device operation invalidates subsequent measurements.
                torch.cuda.synchronize()
            save(args.output / 'report.json', report)
            print(json.dumps({'name': row['name'], 'variant': name,
                              **{k: v for k, v in result.items() if k != 'timings'}}), flush=True)
    report['completed'] = True
    save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
