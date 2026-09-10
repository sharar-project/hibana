"""Enumerate cuBLASLt tactics instead of accepting one backend heuristic.

Real code-p32 operands select tactics. Held-out operands and full generation
remain separate gates. Tactic indices are process-local, never serving IDs.
"""
import argparse
from pathlib import Path
import statistics

import torch
from torch.utils.cpp_extension import load

from bench.gb10_matrix import apply, digest, error, linears, load_capture, save
from bench.support.cuda import capture_graph, time_graph
from sharar import config
from sharar.model.qwen38 import Qwen38


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--expand', action='store_true', help='Also check custom options around every heuristic tile/stage')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent / 'csrc/gb10_fp8_tactics.cpp'
    extension = load(name='sharar_bench_fp8_tactics', sources=[str(source)], with_cuda=True,
                     extra_cflags=['-O3'], extra_ldflags=['-lcublasLt'], verbose=False)
    weights = linears(Qwen38(config.snapshot_dir(), log=print))
    rows = load_capture(args.captures, 'code-p32.pt')
    selected = [r for r in rows if r['name'] in ('l32.mixer.in_qkv', 'l32.mixer.in_z',
                                                'l32.mixer.out', 'l63.mixer.qkv')]
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    workspace = torch.empty(64 * 2**20, dtype=torch.uint8, device='cuda')
    report = {'completed': False, 'source_sha256': digest(Path(__file__)), 'cpp_sha256': digest(source),
              'workspace_bytes': len(workspace), 'heuristics_requested': 128,
              'expanded_options': args.expand,
              'metadata_fields': ['workspace_bytes', 'id', 'tile', 'split_k', 'reduction', 'swizzle', 'custom', 'stages'], 'cells': []}
    save(args.output / 'report.json', report)
    for row in selected:
        lin = weights[row['name']]
        tactics = extension.Tactics(16, len(lin.weight), row['q'].shape[1], 128, len(workspace))
        if args.expand:
            print({'name': row['name'], 'expanded_tactics': tactics.expand()}, flush=True)
        cell = {'name': row['name'], 'weight_shape': list(lin.weight.shape), 'tactics': []}
        report['cells'].append(cell)
        baseline, _ = capture_graph(lambda: apply(row, lin, 'torch'))
        for index, metadata in enumerate(tactics.metadata()):
            item = {'index': index, 'metadata': metadata}
            cell['tactics'].append(item)
            try:
                graph, out = capture_graph(lambda: tactics.run(row['q'], lin.weight, lin.input_scale,
                                                               lin.weight_scale, index, workspace))
                graph.replay()
                torch.cuda.synchronize()
                item['error'] = error(out, row['reference'])
                item['pilot'] = time_graph(graph, flush, samples=5)
                del graph, out
            except RuntimeError as exc:
                item['unavailable'] = str(exc)
                torch.cuda.synchronize()
        exact = [r for r in cell['tactics'] if r.get('error', {}).get('exact')]
        cell['confirmation'] = []
        for best in sorted(exact, key=lambda r: r['pilot']['median_ms'])[:3]:
            index = best['index']
            graph, _ = capture_graph(lambda: tactics.run(row['q'], lin.weight, lin.input_scale,
                                                         lin.weight_scale, index, workspace))
            confirm = {'index': index, 'metadata': best['metadata'], 'timings': []}
            cell['confirmation'].append(confirm)
            for block in range(3):
                for mode in ('baseline', 'candidate', 'candidate', 'baseline'):
                    confirm['timings'].append({'block': block, 'mode': mode,
                        **time_graph(baseline if mode == 'baseline' else graph, flush, samples=5)})
            confirm['median_ms'] = {mode: statistics.median(r['median_ms'] for r in confirm['timings'] if r['mode'] == mode)
                                     for mode in ('baseline', 'candidate')}
            del graph
        save(args.output / 'report.json', report)
        print({'name': row['name'], 'tactics': len(cell['tactics']), 'exact': len(exact),
               'best': [{k: v for k, v in c.items() if k != 'timings'} for c in cell['confirmation']]}, flush=True)
    report['completed'] = True
    save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
