"""Exact NVFP4 encoding screen, including captured MLP outputs and prefill shapes.

The bit encoder retains round-to-nearest-even and signed zero. Hardware CVT
is a separate compile probe; unsupported instructions never enter serving.
"""
import argparse
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from bench.gb10_matrix import digest, linears, load_capture, save
from bench.nvfp4_cost import capture_graph, time_graph
from sharar import config
from sharar.kernels import nvfp4_quant as native
from sharar.model.qwen38 import Qwen38


@triton.jit
def _encode(v, HARDWARE: tl.constexpr):
    if HARDWARE:
        pairs = tl.reshape(v, (v.numel // 2, 2))
        low, high = tl.split(pairs)
        return tl.inline_asm_elementwise(
            "{ .reg .b8 r; cvt.rn.satfinite.e2m1x2.f32 r, $1, $2; mov.b32 $0, {r,r,r,r}; }",
            constraints='=r,f,f', args=[high, low], dtype=tl.uint8, is_pure=True, pack=1)
    a = tl.minimum(tl.abs(v), 6.0)
    bits = a.to(tl.int32, bitcast=True)
    # Normal E2M1 numbers keep one mantissa bit and have exponent bias one.
    normal = (bits - 0x3f000000 + 0x1fffff + ((bits >> 22) & 1)) >> 22
    subnormal = tl.inline_asm_elementwise('cvt.rni.s32.f32 $0, $1;', '=r,f',
                                        [a * 2.0], dtype=tl.int32, is_pure=True, pack=1)
    code = tl.minimum(tl.where(a < 1.0, subnormal, normal), 7)
    code |= (v.to(tl.int32, bitcast=True) >> 28) & 8
    low, high = tl.split(tl.reshape(code, (v.numel // 2, 2)))
    return (low | (high << 4)).to(tl.uint8)


@triton.jit
def _quant(X, GS, Q, SF, K: tl.constexpr, KP: tl.constexpr, STRIDE: tl.constexpr,
           BLOCK: tl.constexpr, SILU: tl.constexpr, HARDWARE: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + cols, cols < K, 0).to(tl.float32)
    if SILU:
        up = tl.load(X + row * STRIDE + K + cols, cols < K, 0).to(tl.float32)
        x = (tl.div_rn(x, 1.0 + libdevice.exp(-x)) * up).to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(x, (BLOCK // 16, 16))
    peak = tl.max(tl.abs(grouped), 1)
    scale = tl.load(GS)
    sf = (scale * (peak * native.rcp_approx_ftz(tl.full((), 6.0, tl.float32)))).to(tl.float8e4nv)
    inverse = tl.where(peak != 0.0, native.rcp_approx_ftz(sf.to(tl.float32) * native.rcp_approx_ftz(scale)), 0.0)
    values = tl.reshape(grouped * inverse[:, None], (BLOCK,))
    packed = _encode(values, HARDWARE)
    dst = chunk * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(Q + row * (K // 2) + dst, packed, dst < K // 2)
    groups = chunk * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
    offset = row // 128 * (128 * KP) + (row % 32) * 16 + (row // 32 % 4) * 4 + groups // 4 * 512 + groups % 4
    tl.store(SF + offset, sf, groups < K // 16)


def quant(x, scale, silu=False, hardware=False):
    m, width = x.shape
    k = width // 2 if silu else width
    out = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    sf = native._scale_buffer(m, k, x.device)
    _quant[(m, triton.cdiv(k, 1024))](x, scale, out, sf, k, sf.shape[1], x.stride(0),
                                     1024, silu, hardware, num_warps=4)
    return out, sf


@triton.jit
def _probe(X, Y, N: tl.constexpr, HARDWARE: tl.constexpr):
    ix = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    v = tl.load(X + ix, ix < N, 0)
    result = _encode(v, HARDWARE)
    out_ix = tl.program_id(0) * 512 + tl.arange(0, 512)
    tl.store(Y + out_ix, result, out_ix < N // 2)


def encoder_check(hardware):
    # All finite BF16 values plus FP32 neighbors of each quantization midpoint.
    bf = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    values = bf[torch.isfinite(bf)]
    ties = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    extra = torch.cat([ties, torch.nextafter(ties, torch.zeros_like(ties)),
                       torch.nextafter(ties, torch.full_like(ties, float('inf')))])
    values = torch.cat([values, extra, -extra])
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    distance = (values.abs().clamp_max(6)[:, None] - levels).abs()
    minimum = distance.min(1, keepdim=True).values
    # Prefer an even code at exact ties, independently of the bit encoder.
    ranks = torch.arange(8) + torch.arange(8) % 2 * 8
    codes = torch.where(distance == minimum, ranks, 99).argmin(1)
    codes |= torch.signbit(values).long() * 8
    gold = (codes[::2] | (codes[1::2] << 4)).to(torch.uint8)
    device = values.cuda()
    out = torch.empty(len(gold), device='cuda', dtype=torch.uint8)
    _probe[(triton.cdiv(len(values), 1024),)](device, out, len(values), hardware)
    assert torch.equal(out.cpu(), gold), f'FP4 encoder mismatch: hardware={hardware}'
    return len(values)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'source_sha256': digest(Path(__file__)), 'encoder': {}, 'cells': []}
    modes = []
    for mode in ('bits', 'hardware'):
        try:
            report['encoder'][mode] = {'exact_values': encoder_check(mode == 'hardware')}
            modes.append(mode)
        except Exception as exc:
            report['encoder'][mode] = {'failed': f'{type(exc).__name__}: {exc}'}
            torch.cuda.synchronize()
        save(args.output / 'report.json', report)
    assert 'bits' in modes, report['encoder']
    weights = linears(Qwen38(config.snapshot_dir(), log=print))
    rows = load_capture(args.captures, 'code-p32.pt')
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    for layer in (0, 32, 63):
        x = next(r['reference'] for r in rows if r['name'] == f'l{layer}.mlp.gate_up')
        scale = weights[f'l{layer}.mlp.down'].input_scale_inv
        for m in (16, 2048):
            value = x if m == 16 else x.repeat(m // 16, 1)
            for silu in (False, True):
                value_input = value if silu else value[:, :5120].contiguous()
                oracle = native.silu_mul_nvfp4_quantize if silu else native.nvfp4_quantize
                q, sf = oracle(value_input, scale)
                gold_q, gold_sf = q.clone(), sf.clone()
                baseline, _ = capture_graph(lambda: oracle(value_input, scale))
                for mode in modes:
                    candidate, output = capture_graph(lambda: quant(value_input, scale, silu, mode == 'hardware'))
                    candidate.replay()
                    torch.cuda.synchronize()
                    cell = {'layer': layer, 'm': m, 'silu': silu, 'mode': mode,
                            'prefill_input': 'repeated captured M16 rows' if m > 16 else 'captured verify rows',
                            'packed_exact': torch.equal(output[0], gold_q), 'scales_exact': torch.equal(output[1], gold_sf), 'timings': []}
                    report['cells'].append(cell)
                    assert cell['packed_exact'] and cell['scales_exact'], cell
                    for block in range(3):
                        for which in ('baseline', 'candidate', 'candidate', 'baseline'):
                            cell['timings'].append({'block': block, 'which': which,
                                **time_graph(baseline if which == 'baseline' else candidate, flush, samples=5)})
                    cell['median_ms'] = {which: statistics.median(r['median_ms'] for r in cell['timings'] if r['which'] == which)
                                         for which in ('baseline', 'candidate')}
                    save(args.output / 'report.json', report)
                    print({k: v for k, v in cell.items() if k != 'timings'}, flush=True)
    report['completed'] = True
    save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
