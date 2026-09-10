"""Validate the integrated exact bundle and diagnose FP8 using real operands.

Run inside an exclusive GB10 window. All timing uses CUDA graphs of the full
distinct-weight projection sequence, with ABBA ordering and cache flushing.
It is diagnostic kernel timing, separate from native or HTTP generation TPS.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import torch
import triton

from bench.gb10_fp8_m16 import _fp8_m16_kernel, fp8_gemm_m16
from bench.gb10_matrix import apply, digest, error, linears, load_capture
from bench.nvfp4_cost import capture_graph, time_graph
from sharar import config
from sharar.kernels import ops
from sharar.model.qwen38 import Qwen38


def shape_tuned(row, lin):
    if tuple(lin.weight.shape) != (14336, 5120):
        return fp8_gemm_m16(row['q'], lin.weight, lin.alpha)
    q = row['q']
    n, k = lin.weight.shape
    out = torch.empty((16, n), device=q.device, dtype=torch.bfloat16)
    _fp8_m16_kernel[(triton.cdiv(n, 16),)](
        q, lin.weight, out, lin.alpha, n, k, q.stride(0), lin.weight.stride(0), out.stride(0),
        BLOCK_N=16, BLOCK_K=256, M_PAD=16, num_warps=2, num_stages=3)
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    report = {'completed': False, 'source_sha256': digest(Path(__file__)),
              'protocol': 'Nine frozen captures; exact-value integration checks (torch.equal); 3 ABBA blocks with 5 cold graph replays per entry.',
              'exact_checks': [], 'fp8_checks': [], 'timings': {}}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    weights = linears(Qwen38(config.snapshot_dir()))
    manifest = json.loads((args.captures / 'report.json').read_text())
    ops.VERIFY_GEMM_ENABLED = True
    for cell in manifest['cells']:
        rows = load_capture(args.captures, cell['file'])
        for row in rows:
            lin = weights[row['name']]
            if row['format'] == 'nvfp4' and row['name'] != 'vocab':
                got = ops.nvfp4_linear(None, lin, (row['q'], row['sf']))
            elif row['format'] == 'fp8' and tuple(lin.weight.shape) == (14336, 5120):
                got = ops.fp8_linear(None, lin, row['q'])
            else:
                continue
            result = {'capture': cell['file'], 'name': row['name'], **error(got, row['reference'])}
            report['exact_checks'].append(result)
            if not result['exact']:
                save()
                raise AssertionError(result)
        print('Validated integrated dispatch:', cell['file'], flush=True)
        save()
        del rows
    assert len(report['exact_checks']) == 1296
    rows = [r for r in load_capture(args.captures, 'code-p32.pt') if r['format'] == 'fp8']
    assert len(rows) == 176
    report['inventory'] = dict(Counter(str(tuple(weights[r['name']].weight.shape)) for r in rows))
    report['fp8_weight_bytes'] = sum(weights[r['name']].weight.numel() for r in rows)
    modes = {
        'native': lambda r, w: apply(r, w, 'torch'),
        'exact_qkv': lambda r, w: ops.fp8_linear(None, w, r['q']),
        'm16_default': lambda r, w: fp8_gemm_m16(r['q'], w.weight, w.alpha),
        'm16_shape_tuned': shape_tuned,
    }
    for row in rows:
        lin = weights[row['name']]
        for mode in ('m16_default', 'm16_shape_tuned'):
            result = {'name': row['name'], 'mode': mode, **error(modes[mode](row, lin), row['reference'])}
            report['fp8_checks'].append(result)
            assert result['finite'], result
    save()
    groups = {'all': rows}
    for shape in sorted({tuple(weights[r['name']].weight.shape) for r in rows}):
        groups[str(shape)] = [r for r in rows if tuple(weights[r['name']].weight.shape) == shape]
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    for group, selected in groups.items():
        graphs = {}
        for mode, fn in modes.items():
            def forward(fn=fn):
                for row in selected:
                    out = fn(row, weights[row['name']])
                return out
            graphs[mode], _ = capture_graph(forward)
        samples = []
        for candidate in ('exact_qkv', 'm16_default', 'm16_shape_tuned'):
            for block in range(3):
                for mode in ('native', candidate, candidate, 'native'):
                    samples.append({'candidate': candidate, 'block': block, 'mode': mode,
                                    **time_graph(graphs[mode], flush, samples=5)})
        summary = {}
        for candidate in ('exact_qkv', 'm16_default', 'm16_shape_tuned'):
            pair = {mode: statistics.median(r['median_ms'] for r in samples
                                           if r['candidate'] == candidate and r['mode'] == mode)
                    for mode in ('native', candidate)}
            summary[candidate] = {**pair, 'saving_ms': pair['native'] - pair[candidate]}
        report['timings'][group] = {'calls': len(selected), 'summary': summary, 'samples': samples}
        save()
        print(group, json.dumps(summary), flush=True)
        del graphs
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
