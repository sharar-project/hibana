// Nemotron-3.5-Super's latent MoE block for decode and verify calls (at most 16 rows), SM120/SM121.
//
// Every matrix of the block streams once from DRAM per call, every distinct routed expert once for all of its
// (row, k) pairs. Four launches, chained by programmatic dependent launch (a kernel's first weight loads go out
// under the previous kernel's tail):
//   router  the router|fc1_latent GEMV [1536, 4096] into FP32 (on the CUDA cores up to 8 rows: exact products in a
//           fixed order, closer to an FP32 reference than the tensor core; else tensor-core fragments). Around it x
//           is quantized for the shared expert, each latent block's 16 features of every row for the routed experts,
//           and the last of the 32 router-logit blocks routes every row (`moe.py`'s _route_kernel, bit for bit) and
//           plans the call: its distinct experts in ascending id and their pairs
//   up      routed experts' up and the shared expert's up, relu^2 (BF16, as the reference rounds it), quantized to
//           NVFP4 in the epilogue: a warp's 16 rows are one block-scale group
//   down    routed experts' down, each pair's FP32 output weighted; the last block of each 64-feature tile sums the
//           pairs per row in selection order (deterministic)
//   out     fc2_latent (BF16) plus the shared expert's down (NVFP4) into the block's BF16 output
// (`prep` plans given ids instead: budgeted routing, or the routed experts alone for the MTP head's DraftMoe.)
//
// The matrices go through the tensor core with small-M fragments: weights in the 16-row A operand of mma.sync
// m16n8k{16,64}, a call's rows (or an expert's pairs) in the 8 columns of B, FP32 accumulation. NVFP4 is the
// checkpoint's layout (packed rows, low nibble first; E4M3 block scales per 16 in CUTLASS's SM120 128x4 interleave),
// so prefill keeps `cutlass_fused_moe` and the NVFP4 GEMM on the same tensors; the tensor core applies the block scales
// (kind::mxf4nvf4 scale_vec::4X, UE4M3).
//
// NVFP4 fragments. A lane (g = lane / 4, t = lane % 4) loads 16 contiguous bytes (32 FP4, two block-scale groups) of
// rows g and g + 8 per 128-k step; the MMA wants each lane to hold half of a group (k 8t .. 8t + 7 of the 64), the
// other half in lane t ^ 1. One exchange of two words between lanes t and t ^ 1 gives that, the step's eight physical
// groups taken by the two MMAs in the order (0, 4, 1, 5) and (2, 6, 3, 7). The weight scales follow by one byte
// permute; the activations (which these kernels quantize themselves) are stored in that order, so a lane reads its B
// fragments of a step as one 16-byte word and their scales as one 8-byte word. BF16 needs no exchange (no groups):
// both operands take the same permutation of k within a 32-k step, the natural one of a 16-byte load.
//
// Tried and dropped (block 1, real inputs, CUDA graphs, GB10): staging the weights through shared memory by cp.async
// in 256-byte row pieces (DRAM serves those ~10% faster than the fragments' 64-byte ones in a pure read) made the down
// kernel 177 GB/s against 216; the CUDA-core path for the experts and the shared expert (whole 512-byte row pieces a
// warp, E2M1 -> f16x2 dot products) ran 162-198 GB/s against 216-233.
//
// Measured (Sharar, spark-9931, 2026-10-24): block 1, real inputs (captured text, consecutive rows), CUDA graphs, dense
// weights rotated over 4 copies, 8 windows, median us per call, no other GPU process during any measurement. GB/s over
// the bytes the call must read (45.7 MB dense + distinct experts x 3.10 MB). `cutlass` / `b12x`: the block with the
// routed experts through cutlass_fused_moe / FlashInfer's b12x MoE (router|fc1 torch.mm, route_topk, NVFP4 shared
// expert, fc2 addmm around them); `these`: this file's kernels (parts: router, up, down, out).
//      M  experts    MB |  cutlass  GB/s |    b12x  GB/s |   these  GB/s  (router    up  down   out)
//      1     22.0   114 |    621.4   183 |    571.8  199 |   505.5  225   (    58   207   155    89)
//      2     42.8   178 |    889.5   200 |    826.6  216 |   780.4  228   (    58   340   303    93)
//      4     81.2   297 |   1414.5   210 |   1363.4  218 |  1304.2  228   (    58   595   554    99)
//      8    145.5   496 |   2295.7   216 |   2265.0  219 |  2174.9  228   (    59   998  1024    81)
//     16    240.6   791 |   3601.9   220 |   3591.0  220 |  3480.5  227   (    71  1628  1686    97)
//     32    338.1  1093 |   4987.1   219 |   4875.5  224 |
//     64    403.4  1295 |   5962.0   217 |   5768.5  224 |
//    128    486.1  1551 |   7286.4   213 |   6958.2  223 |
//    512    511.5  1630 |   8725.6   187 |   7685.2  212 |
//   2048    512.0  1631 |  13497.6   121 |  10324.0  158 |
// Past 16 rows b12x streams the experts at 224-226 GB/s; these kernels' run ~228 at 16 rows, so wider ones would gain
// a percent or two: calls past 16 rows go to b12x (SHARAR_NEMOTRON_EXPERTS=auto). Accuracy (blocks 1, 8, 14, real
// inputs; cosine against the reference with NVFP4 activations): these kernels >= 0.999998 (M 1-16), cutlass >= 0.99923
// (0.99936 at M 512 in block 1), b12x 0.99926-0.99996 with its routed sum mostly 0.2-1.6% smaller than cutlass's
// (block 14: 1.5% at M 1-4; block 1 at M 512: 1.5% larger); against the FP32 reference all are at the W4A4 floor (0.981-0.998).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kE = 512, kL = 1024, kI = 2688, kH = 4096, kS = 5376;  // experts, latent, expert inter, hidden, shared inter
constexpr int kRouter = kE + kL;  // router|fc1 rows
constexpr int kMaxRows = 16;      // rows per call; also pairs per expert in one pass (two n8 tiles)
constexpr int kMaxTopk = 32;      // experts per row (a warp's lanes in the routing)
constexpr int kWarps = 4;         // up / down blocks: 4 warps of 16 rows
constexpr int kBlockRows = kWarps * 16;
constexpr int kSplitWarps = 8;    // router / out blocks: 16 rows, K split over 8 warps
constexpr int kUpShared = kS / kBlockRows;  // up blocks of the shared expert, first in the grid
constexpr int kRouteCounter = 32;  // counters: [0, 16) the down kernel's tiles, [32] the router's logit blocks
constexpr int kPF = 4;             // steps in flight per warp
// plan words: D, the expert of each distinct slot, the slot starts (D + 1)
constexpr int kPlanExp = 1, kPlanStart = kPlanExp + kE, kPlanWords = kPlanStart + kE + 1;

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// two FP32 -> E2M1 codes in one byte (lo in the low nibble), round to nearest even, saturating
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
  uint32_t r;
  asm("{ .reg .b8 e; cvt.rn.satfinite.e2m1x2.f32 e, %2, %1; cvt.u32.u8 %0, e; }" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

__device__ __forceinline__ void mma_fp4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1, uint32_t sfa, uint32_t sfb) {
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Programmatic dependent launch: a kernel may start while the one before it finishes; it reads what that one wrote
// only after `grid_wait`, and lets the next one start (`grid_launch`) once all of its blocks are running.
__device__ __forceinline__ void grid_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void grid_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

// Storage of an NVFP4 activation row: per 128-k step 64 bytes of codes and 8 bytes of scales. Physical group G (0..7,
// k 16 G .. 16 G + 15 of the step), half h (k 8 h ..): word 8 (G / 4) + 4 h + G % 4; its scale at byte 2 (G % 4) + G / 4.
__device__ __forceinline__ int code_word(int G, int h) { return 8 * (G >> 2) + 4 * h + (G & 3); }
__device__ __forceinline__ int scale_byte(int G) { return 2 * (G & 3) + (G >> 2); }

// NVFP4 of 16 values with global scale gs (`kernels/nvfp4_quant.py`'s arithmetic): two code words and the E4M3 scale.
__device__ __forceinline__ uint8_t quant16(const float (&v)[16], float gs, uint32_t& lo, uint32_t& hi) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const __nv_fp8_storage_t sf = __nv_cvt_float_to_fp8(gs * (amax * rcp_approx(6.f)), __NV_SATFINITE, __NV_E4M3);
  const float sff = __half2float(__half(__nv_cvt_fp8_to_halfraw(sf, __NV_E4M3)));
  const float inv = amax != 0.f ? rcp_approx(sff * rcp_approx(gs)) : 0.f;
  lo = hi = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lo |= e2m1x2(v[2 * i] * inv, v[2 * i + 1] * inv) << (8 * i);
    hi |= e2m1x2(v[8 + 2 * i] * inv, v[8 + 2 * i + 1] * inv) << (8 * i);
  }
  return sf;
}

// Group j (k 16 j .. 16 j + 15) of an activation row (codes q, scales s), quantized from 16 values.
__device__ __forceinline__ void store_group(uint8_t* q, uint8_t* s, int j, const float (&v)[16], float gs) {
  uint32_t lo, hi;
  const uint8_t sf = quant16(v, gs, lo, hi);
  const int st = j >> 3, G = j & 7;
  uint32_t* w = reinterpret_cast<uint32_t*>(q + 64 * st);
  w[code_word(G, 0)] = lo;
  w[code_word(G, 1)] = hi;
  s[8 * st + scale_byte(G)] = sf;
}

// Group j of BF16 row `src` quantized into an activation row.
__device__ __forceinline__ void quant_group_bf16(const __nv_bfloat16* src, int j, float gs, uint8_t* q, uint8_t* s) {
  const uint4* p = reinterpret_cast<const uint4*>(src + 16 * j);
  const uint4 u0 = p[0], u1 = p[1];
  const __nv_bfloat16* b0 = reinterpret_cast<const __nv_bfloat16*>(&u0);
  const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&u1);
  float v[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = __bfloat162float(b0[i]);
    v[8 + i] = __bfloat162float(b1[i]);
  }
  store_group(q, s, j, v, gs);
}

// Byte offset of row r's scales in the 128x4 interleave (groups padded to KG, a multiple of 4); group kg adds
// (kg / 4) * 512 + kg % 4.
__device__ __forceinline__ size_t swz_row(int r, int KG) {
  return static_cast<size_t>(r >> 7) * (128 * KG) + (r & 31) * 16 + ((r >> 5) & 3) * 4;
}

// Lanes t, t ^ 1 trade two words: afterwards word j of a lane is half (t % 2) of physical group 4 (t / 2) + j.
__device__ __forceinline__ uint4 exchange(uint4 v, bool odd) {
  const uint32_t s0 = odd ? v.x : v.y, s1 = odd ? v.z : v.w;
  const uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
  return odd ? make_uint4(r0, r1, v.y, v.w) : make_uint4(v.x, v.z, r0, r1);
}

// ---- one warp's 16 weight rows against NT n8 tiles of activation rows, steps [s0, s1), kPF steps in flight

// NVFP4: rg / rg8 rows g / g + 8 of the weight (+ 16 t bytes), sr the scale row g + 8 (t % 2) (swz_row), xq / xs
// the B column g of each tile (code row + 16 t, scale row; null: a zero column).
template <int NT>
struct Fp4Op {
  const uint8_t *rg, *rg8, *sr;
  const uint8_t* xq[NT];
  const uint8_t* xs[NT];
};

template <int NT>
struct Fp4Step {
  uint4 a, b;
  uint32_t slo, shi;
  uint4 x[NT];
  uint2 xs[NT];
};

template <int NT>
__device__ __forceinline__ void load_w(Fp4Step<NT>& w, const Fp4Op<NT>& op, int s) {
  w.a = __ldcs(reinterpret_cast<const uint4*>(op.rg + 64 * s));
  w.b = __ldcs(reinterpret_cast<const uint4*>(op.rg8 + 64 * s));
  w.slo = __ldcs(reinterpret_cast<const unsigned int*>(op.sr + (2 * s) * 512));
  w.shi = __ldcs(reinterpret_cast<const unsigned int*>(op.sr + (2 * s + 1) * 512));
}

template <int NT>
__device__ __forceinline__ void load_x(Fp4Step<NT>& w, const Fp4Op<NT>& op, int s) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    w.x[nt] = op.xq[nt] ? __ldg(reinterpret_cast<const uint4*>(op.xq[nt] + 64 * s)) : make_uint4(0u, 0u, 0u, 0u);
    w.xs[nt] = op.xs[nt] ? __ldg(reinterpret_cast<const uint2*>(op.xs[nt] + 8 * s)) : make_uint2(0u, 0u);
  }
}

template <int NT>
__device__ __forceinline__ void compute(float (&acc)[NT][4], const Fp4Step<NT>& w, bool odd) {
  const uint4 a = exchange(w.a, odd), b = exchange(w.b, odd);
  const uint32_t sfa0 = prmt(w.slo, w.shi, 0x5140), sfa1 = prmt(w.slo, w.shi, 0x7362);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    mma_fp4(acc[nt], a.x, b.x, a.y, b.y, w.x[nt].x, w.x[nt].y, sfa0, w.xs[nt].x);
    mma_fp4(acc[nt], a.z, b.z, a.w, b.w, w.x[nt].z, w.x[nt].w, sfa1, w.xs[nt].y);
  }
}

// BF16: rg / rg8 rows g / g + 8 (+ 16 t bytes), x the B column g of each tile (row + 16 t bytes, or null); 32 k a step.
template <int NT>
struct Bf16Op {
  const uint8_t *rg, *rg8;
  const uint8_t* x[NT];
};

template <int NT>
struct Bf16Step {
  uint4 a, b;
  uint4 x[NT];
};

template <int NT>
__device__ __forceinline__ void load_w(Bf16Step<NT>& w, const Bf16Op<NT>& op, int s) {
  w.a = __ldcs(reinterpret_cast<const uint4*>(op.rg + 64 * s));
  w.b = __ldcs(reinterpret_cast<const uint4*>(op.rg8 + 64 * s));
}

template <int NT>
__device__ __forceinline__ void load_x(Bf16Step<NT>& w, const Bf16Op<NT>& op, int s) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
    w.x[nt] = op.x[nt] ? __ldg(reinterpret_cast<const uint4*>(op.x[nt] + 64 * s)) : make_uint4(0u, 0u, 0u, 0u);
}

template <int NT>
__device__ __forceinline__ void compute(float (&acc)[NT][4], const Bf16Step<NT>& w, bool) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    mma_bf16(acc[nt], w.a.x, w.b.x, w.a.y, w.b.y, w.x[nt].x, w.x[nt].y);
    mma_bf16(acc[nt], w.a.z, w.b.z, w.a.w, w.b.w, w.x[nt].z, w.x[nt].w);
  }
}

// Steps [s0, s1) into acc (zeroed here). `wait`: the first kPF steps' weights are requested before `grid_wait`, their
// activations (the previous kernel's output) after it.
template <int NT, class Step, class Op>
__device__ __forceinline__ void run(float (&acc)[NT][4], const Op& op, int s0, int s1, bool wait) {
  const bool odd = threadIdx.x & 1;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;
  Step ring[kPF];
#pragma unroll
  for (int j = 0; j < kPF; ++j)
    if (s0 + j < s1) load_w(ring[j], op, s0 + j);
  if (wait) {
    grid_wait();
    grid_launch();
  }
#pragma unroll
  for (int j = 0; j < kPF; ++j)
    if (s0 + j < s1) load_x(ring[j], op, s0 + j);
  for (int s = s0; s < s1; s += kPF) {
#pragma unroll
    for (int j = 0; j < kPF; ++j) {
      if (s + j < s1) {
        const Step w = ring[j];
        if (s + j + kPF < s1) {
          load_w(ring[j], op, s + j + kPF);
          load_x(ring[j], op, s + j + kPF);
        }
        compute(acc, w, odd);
      }
    }
  }
}

// ---- paired steps: a warp's 16 rows as two tiles of 8, each instruction reading 128 contiguous bytes (two steps) of
// 4 rows. Lane (g, t) loads row g / 2 of the instruction's four at step 2 s + g % 2, so the MMA's row g is that row
// at that step (rows g and g + 1 one row's two steps); the B columns pair up the same way, column 2 c + j token c's
// activation at step 2 s + j, so only the accumulator entries where the row's and the column's step agree count,
// and the two of a row add up across lanes g, g ^ 1 at the end. Four tokens an n8 tile, half the MMA's columns spent;
// the tensor core has the throughput. DRAM serves the 128-byte row pieces 221-236 GB/s where the 64-byte ones of the
// plain fragments get 201-227 (a pure read on GB10, row widths 512-8192 bytes).
template <int NT>
struct P2Fp4Op {
  const uint8_t* w[4];    // rows g/2, 4 + g/2, 8 + g/2, 12 + g/2 of the warp's 16, + 64 (g % 2) + 16 t
  const uint8_t* sr[2];   // tile 0 / 1: the scale row of the lane's SFA row (virtual row g + 8 (t % 2))
  const uint8_t* xq[NT];  // token 4 nt + g/2: code row + 64 (g % 2) + 16 t, or null
  const uint8_t* xs[NT];  // its scale row + 8 (g % 2)
  int ks, par;            // the matrix's steps; g % 2
};

template <int NT>
struct P2Fp4Step {
  uint4 w[4];
  uint32_t slo[2], shi[2];
  uint4 x[NT];
  uint2 xs[NT];
};

template <int NT>
__device__ __forceinline__ void load_w(P2Fp4Step<NT>& d, const P2Fp4Op<NT>& op, int s) {
  const int step = 2 * s + op.par;
  const bool ok = step < op.ks;  // an odd step count: the last pair's second step is zero
#pragma unroll
  for (int i = 0; i < 4; ++i) d.w[i] = ok ? __ldcs(reinterpret_cast<const uint4*>(op.w[i] + 128 * s)) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    d.slo[i] = ok ? __ldcs(reinterpret_cast<const unsigned int*>(op.sr[i] + (2 * step) * 512)) : 0u;
    d.shi[i] = ok ? __ldcs(reinterpret_cast<const unsigned int*>(op.sr[i] + (2 * step + 1) * 512)) : 0u;
  }
}

template <int NT>
__device__ __forceinline__ void load_x(P2Fp4Step<NT>& d, const P2Fp4Op<NT>& op, int s) {
  const bool ok = 2 * s + op.par < op.ks;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    d.x[nt] = ok && op.xq[nt] ? __ldg(reinterpret_cast<const uint4*>(op.xq[nt] + 128 * s)) : make_uint4(0u, 0u, 0u, 0u);
    d.xs[nt] = ok && op.xs[nt] ? __ldg(reinterpret_cast<const uint2*>(op.xs[nt] + 16 * s)) : make_uint2(0u, 0u);
  }
}

template <int NT>
__device__ __forceinline__ void compute2(float (&acc)[2][NT][4], const P2Fp4Step<NT>& d, bool odd) {
#pragma unroll
  for (int T = 0; T < 2; ++T) {
    const uint4 a = exchange(d.w[2 * T], odd), b = exchange(d.w[2 * T + 1], odd);
    const uint32_t sfa0 = prmt(d.slo[T], d.shi[T], 0x5140), sfa1 = prmt(d.slo[T], d.shi[T], 0x7362);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      mma_fp4(acc[T][nt], a.x, b.x, a.y, b.y, d.x[nt].x, d.x[nt].y, sfa0, d.xs[nt].x);
      mma_fp4(acc[T][nt], a.z, b.z, a.w, b.w, d.x[nt].z, d.x[nt].w, sfa1, d.xs[nt].y);
    }
  }
}

template <int NT>
struct P2Bf16Op {
  const uint8_t* w[4];
  const uint8_t* x[NT];
  int ks, par;
};

template <int NT>
struct P2Bf16Step {
  uint4 w[4];
  uint4 x[NT];
};

template <int NT>
__device__ __forceinline__ void load_w(P2Bf16Step<NT>& d, const P2Bf16Op<NT>& op, int s) {
  const bool ok = 2 * s + op.par < op.ks;
#pragma unroll
  for (int i = 0; i < 4; ++i) d.w[i] = ok ? __ldcs(reinterpret_cast<const uint4*>(op.w[i] + 128 * s)) : make_uint4(0u, 0u, 0u, 0u);
}

template <int NT>
__device__ __forceinline__ void load_x(P2Bf16Step<NT>& d, const P2Bf16Op<NT>& op, int s) {
  const bool ok = 2 * s + op.par < op.ks;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
    d.x[nt] = ok && op.x[nt] ? __ldg(reinterpret_cast<const uint4*>(op.x[nt] + 128 * s)) : make_uint4(0u, 0u, 0u, 0u);
}

template <int NT>
__device__ __forceinline__ void compute2(float (&acc)[2][NT][4], const P2Bf16Step<NT>& d, bool) {
#pragma unroll
  for (int T = 0; T < 2; ++T)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      mma_bf16(acc[T][nt], d.w[2 * T].x, d.w[2 * T + 1].x, d.w[2 * T].y, d.w[2 * T + 1].y, d.x[nt].x, d.x[nt].y);
      mma_bf16(acc[T][nt], d.w[2 * T].z, d.w[2 * T + 1].z, d.w[2 * T].w, d.w[2 * T + 1].w, d.x[nt].z, d.x[nt].w);
    }
}

// Paired steps [s0, s1) (two steps each) into acc (zeroed here), two pairs in flight; `wait` as in `run`.
template <int NT, class Step, class Op>
__device__ __forceinline__ void run2(float (&acc)[2][NT][4], const Op& op, int s0, int s1, bool wait) {
  constexpr int PF2 = 2;
  const bool odd = threadIdx.x & 1;
#pragma unroll
  for (int T = 0; T < 2; ++T)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[T][nt][i] = 0.f;
  Step ring[PF2];
#pragma unroll
  for (int j = 0; j < PF2; ++j)
    if (s0 + j < s1) load_w(ring[j], op, s0 + j);
  if (wait) {
    grid_wait();
    grid_launch();
  }
#pragma unroll
  for (int j = 0; j < PF2; ++j)
    if (s0 + j < s1) load_x(ring[j], op, s0 + j);
  for (int s = s0; s < s1; s += PF2) {
#pragma unroll
    for (int j = 0; j < PF2; ++j) {
      if (s + j < s1) {
        const Step d = ring[j];
        if (s + j + PF2 < s1) {
          load_w(ring[j], op, s + j + PF2);
          load_x(ring[j], op, s + j + PF2);
        }
        compute2(acc, d, odd);
      }
    }
  }
}

// The counted entries summed per row: out[row][token] for the warp's 16 rows and tokens 4 nt + t < cols.
template <int NT>
__device__ __forceinline__ void finish2(const float (&acc)[2][NT][4], float (*out)[17], int cols) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, par = g & 1;
#pragma unroll
  for (int T = 0; T < 2; ++T)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      float lo = par ? acc[T][nt][1] : acc[T][nt][0], hi = par ? acc[T][nt][3] : acc[T][nt][2];
      lo += __shfl_xor_sync(0xffffffffu, lo, 4);
      hi += __shfl_xor_sync(0xffffffffu, hi, 4);
      const int c = 4 * nt + t;
      if (!par && c < cols) {
        out[8 * T + (g >> 1)][c] = lo;
        out[8 * T + 4 + (g >> 1)][c] = hi;
      }
    }
}

// An NVFP4 [*, K] weight's rows R0 .. R0 + 15 (w: the matrix, sc: its scales) against `cols` activation rows (row(c))
// of a [*, K] activation (xq / xs), paired steps [s0, s1), into out (raw sums).
template <int NT, int K, class RowOf>
__device__ __forceinline__ void p2_fp4(const uint8_t* w, const uint8_t* sc, int R0, const uint8_t* xq, const uint8_t* xs,
                                       int cols, RowOf row, int s0, int s1, bool wait, float (*out)[17]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  P2Fp4Op<NT> op;
  op.par = g & 1;
  op.ks = K / 128;
#pragma unroll
  for (int i = 0; i < 4; ++i) op.w[i] = w + static_cast<size_t>(R0 + 4 * i + (g >> 1)) * (K / 2) + 64 * op.par + 16 * t;
#pragma unroll
  for (int T = 0; T < 2; ++T) op.sr[T] = sc + swz_row(R0 + 8 * T + 4 * (t & 1) + (g >> 1), K / 16);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int c = 4 * nt + (g >> 1);
    const int r = c < cols ? row(c) : -1;
    op.xq[nt] = r >= 0 ? xq + static_cast<size_t>(r) * (K / 2) + 64 * op.par + 16 * t : nullptr;
    op.xs[nt] = r >= 0 ? xs + static_cast<size_t>(r) * (K / 16) + 8 * op.par : nullptr;
  }
  float acc[2][NT][4];
  run2<NT, P2Fp4Step<NT>>(acc, op, s0, s1, wait);
  finish2(acc, out, cols);
}

template <int NT, class RowOf>
__device__ __forceinline__ void p2_bf16(const __nv_bfloat16* w, int K, int R0, const __nv_bfloat16* x, int cols, RowOf row,
                                        int s0, int s1, bool wait, float (*out)[17]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  P2Bf16Op<NT> op;
  op.par = g & 1;
  op.ks = K / 32;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    op.w[i] = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(R0 + 4 * i + (g >> 1)) * K) + 64 * op.par + 16 * t;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int c = 4 * nt + (g >> 1);
    const int r = c < cols ? row(c) : -1;
    op.x[nt] = r >= 0 ? reinterpret_cast<const uint8_t*>(x + static_cast<size_t>(r) * K) + 64 * op.par + 16 * t : nullptr;
  }
  float acc[2][NT][4];
  run2<NT, P2Bf16Step<NT>>(acc, op, s0, s1, wait);
  finish2(acc, out, cols);
}

// p2_fp4 with one n8 tile up to 4 columns, two up to 8, four up to 16.
template <int K, class RowOf>
__device__ __forceinline__ void p2_fp4_any(const uint8_t* w, const uint8_t* sc, int R0, const uint8_t* xq, const uint8_t* xs,
                                           int cols, RowOf row, int s0, int s1, bool wait, float (*out)[17]) {
  if (cols <= 4) p2_fp4<1, K>(w, sc, R0, xq, xs, cols, row, s0, s1, wait, out);
  else if (cols <= 8) p2_fp4<2, K>(w, sc, R0, xq, xs, cols, row, s0, s1, wait, out);
  else p2_fp4<4, K>(w, sc, R0, xq, xs, cols, row, s0, s1, wait, out);
}

struct Params {
  const __nv_bfloat16* x;    // [M, kH]
  const __nv_bfloat16* lat;  // [M, kL] (prep only)
  const int* ids;            // [M, topk]
  const float* wts;          // [M, topk]
  const __nv_bfloat16 *wr, *fc2;         // [kRouter, kH], [kH, kL]
  const uint8_t *w1, *s1, *w2, *s2;      // [E, kI, kL/2], swizzled [E, kI, kL/16]; [E, kL, kI/2], [E, kL, kI/16]
  const uint8_t *wsu, *ssu, *wsd, *ssd;  // shared up [kS, kH/2] + scales, shared down [kH, kS/2] + scales
  const float *alpha1, *alpha2;          // [E] dequantization scales (fc2's carries routed_scaling_factor)
  const float *gs1, *gs2;                // [1] routed activation global scales (1 / input_scale)
  const float *alpha_su, *alpha_sd, *gs_su, *gs_sd;  // [1] shared expert's
  float* fused;   // [M, kRouter]
  int* plan;
  int* order;     // [P] pairs by slot
  int* counters;  // zero between calls (each call's last blocks reset theirs)
  uint8_t *xq, *xs;    // latent [M, kL/2], [M, kL/16]
  uint8_t *xsq, *xss;  // x for the shared expert [M, kH/2], [M, kH/16]
  uint8_t *hq, *hs;    // routed intermediate [P, kI/2], [P, kI/16]
  uint8_t *hsq, *hss;  // shared intermediate [M, kS/2], [M, kS/16]
  float* y;            // [P, kL]
  __nv_bfloat16* routed;  // [M, kL]
  __nv_bfloat16* out;     // [M, kH]
  int M, topk;      // rows; experts per row (the ids' width)
  int up_shared;    // up blocks of the shared expert (kUpShared, or 0: the routed experts alone)
  const float* bias;  // [E] routing: the choice's score correction
  const int* slots;   // routing: [M / steps] state slots (0: a padded request), or null
  int steps;
  int route;          // the router kernel routes (ids_out, wts_out, the plan)
  int* ids_out;
  float* wts_out;
  int simt;           // the router GEMV on the CUDA cores for calls of at most 8 rows
  int p2;             // paired-step fragments (128-byte row pieces) for the experts, the shared expert and fc2
  int pad_id;         // routing: the id a padded row names (-1: row 0's top expert; 3-bit experts: 512, no expert)
  __nv_bfloat16* lat_out;  // router: the BF16 latent [M, kL] (the 3-bit experts' input), or null
};

// ---------------------------------------------------------------- plan: the call's experts and their pairs

// Distinct experts in ascending id (plan: D, the experts, the slot starts) and the pairs of each slot (order); a pair
// at weight 0 adds nothing and is left out (graph padding, route_budget's unfilled slots). id(i) / wt(i): pair i.
template <class Id, class Wt>
__device__ void make_plan(const Params& p, Id id, Wt wt, int P, int* count, int* slot, int* start, int* fill) {
  const int tid = threadIdx.x, n = blockDim.x, lane = tid & 31;
  for (int e = tid; e < kE; e += n) count[e] = fill[e] = 0;
  __syncthreads();
  for (int i = tid; i < P; i += n) {
    const int e = id(i);
    if (e >= 0 && e < kE && wt(i) != 0.f) atomicAdd(count + e, 1);
  }
  __syncthreads();
  if (tid < 32) {
    int base = 0, total = 0;
    for (int c = 0; c < kE / 32; ++c) {
      const int e = c * 32 + lane, m = count[e];
      const unsigned ballot = __ballot_sync(0xffffffffu, m > 0);
      const int d = base + __popc(ballot & ((1u << lane) - 1u));
      int inclusive = m;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inclusive, o);
        if (lane >= o) inclusive += v;
      }
      slot[e] = m > 0 ? d : -1;
      if (m > 0) {
        start[d] = total + inclusive - m;
        p.plan[kPlanExp + d] = e;
        p.plan[kPlanStart + d] = total + inclusive - m;
      }
      base += __popc(ballot);
      total += __shfl_sync(0xffffffffu, inclusive, 31);
    }
    if (lane == 0) {
      p.plan[0] = base;
      p.plan[kPlanStart + base] = total;
    }
  }
  __syncthreads();
  // the order of an expert's pairs only places them in columns: the result is the same
  for (int i = tid; i < P; i += n) {
    const int e = id(i);
    if (e >= 0 && e < kE && wt(i) != 0.f) p.order[start[slot[e]] + atomicAdd(fill + e, 1)] = i;
  }
}

// ---------------------------------------------------------------- router|fc1: 16 rows a block

// torch's FP32 sigmoid, 1 / (1 + exp(-x)) with the accurate exp and a rounded division (`moe.py`'s _sigmoid_exact)
__device__ __forceinline__ float sigmoid_exact(float x) { return __fdiv_rn(1.f, 1.f + expf(-x)); }

// The routing of `moe.py`'s _route_kernel, one warp a row: top K of sigmoid(logit) + bias (as order-preserving
// integer keys over the expert, ties to the lower expert), the plain sigmoid weights renormalized (the butterfly sum
// of Triton's warp reduction). ids / weights of row t into sid / sw [t * 32 + k].
__device__ void route_rows(const Params& p, int* sid, float* sw) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int t = warp; t < p.M; t += kSplitWarps) {
    const float* row = p.fused + static_cast<size_t>(t) * kRouter;
    long long key[kE / 32];
#pragma unroll
    for (int j = 0; j < kE / 32; ++j) {
      const int e = lane + 32 * j;
      const float c = sigmoid_exact(__ldcg(row + e)) + __ldg(p.bias + e);
      const int bits = __float_as_int(c), ord = bits ^ ((bits >> 31) & 0x7FFFFFFF);
      key[j] = (static_cast<long long>(ord) << 32) | static_cast<unsigned>(kE - 1 - e);
    }
    int mine = 0;
    for (int k = 0; k < p.topk; ++k) {
      long long b = key[0];
#pragma unroll
      for (int j = 1; j < kE / 32; ++j) b = max(b, key[j]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) b = max(b, __shfl_xor_sync(0xffffffffu, b, o));
      const int e = kE - 1 - static_cast<int>(b & 0xFFFFFFFF);
      if (lane == k) mine = e;
      if ((e & 31) == lane) {
#pragma unroll
        for (int j = 0; j < kE / 32; ++j)
          if (j == (e >> 5)) key[j] = LLONG_MIN;
      }
    }
    const float s = lane < p.topk ? sigmoid_exact(__ldcg(row + mine)) : 0.f;
    float sum = s;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    sid[t * 32 + lane] = mine;
    sw[t * 32 + lane] = __fdiv_rn(s, sum + 1e-20f);
  }
}

// Tensor-core fragments: K over the 8 warps, 16 steps each; partial sums into red[warp][row][column].
template <int NT>
__device__ __forceinline__ void router_block(const Params& p, float (*red)[16][kMaxRows + 1]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int R0 = blockIdx.x * 16;
  constexpr int KS = kH / 32;
  Bf16Op<NT> op;
  op.rg = reinterpret_cast<const uint8_t*>(p.wr + static_cast<size_t>(R0 + g) * kH) + 16 * t;
  op.rg8 = op.rg + 8 * kH * 2;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int c = nt * 8 + g;
    op.x[nt] = c < p.M ? reinterpret_cast<const uint8_t*>(p.x + static_cast<size_t>(c) * kH) + 16 * t : nullptr;
  }
  float acc[NT][4];
  run<NT, Bf16Step<NT>>(acc, op, warp * KS / kSplitWarps, (warp + 1) * KS / kSplitWarps, true);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][g + 8 * (i >> 1)][nt * 8 + 2 * t + (i & 1)] = acc[nt][i];
}

// The same 16 rows on the CUDA cores, for calls of at most MX rows: a warp's K range is 512, which its lanes read as
// two 512-byte pieces of each row (the contiguous row pieces DRAM serves fastest, where the fragments read 64 bytes of
// a row at a time), x of the range in registers, the products exact in FP32 and summed in a fixed order (closer to an
// FP32 reference than the tensor core's accumulation, which helps the top-22 ties), four rows' loads in flight.
template <int MX>
__device__ __forceinline__ void router_block_simt(const Params& p, float (*red)[16][kMaxRows + 1]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int R0 = blockIdx.x * 16, k0 = 512 * warp + 8 * lane;
  const uint8_t* wb = reinterpret_cast<const uint8_t*>(p.wr + static_cast<size_t>(R0) * kH + k0);
  constexpr int RB = 4;
  uint4 w[2][RB][2];
  auto fetch = [&](uint4 (&d)[RB][2], int b) {
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        d[r][j] = __ldcs(reinterpret_cast<const uint4*>(wb + static_cast<size_t>(b * RB + r) * kH * 2 + 512 * j));
  };
  fetch(w[0], 0);
  grid_wait();
  grid_launch();
  uint4 xr[MX][2];
#pragma unroll
  for (int m = 0; m < MX; ++m)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      xr[m][j] = m < p.M ? __ldg(reinterpret_cast<const uint4*>(p.x + static_cast<size_t>(m) * kH + k0 + 256 * j))
                         : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int b = 0; b < 16 / RB; ++b) {
    if (b + 1 < 16 / RB) fetch(w[(b + 1) & 1], b + 1);
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      float acc[MX];
#pragma unroll
      for (int m = 0; m < MX; ++m) acc[m] = 0.f;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint32_t* wv = reinterpret_cast<const uint32_t*>(&w[b & 1][r][j]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float w0 = __uint_as_float(wv[i] << 16), w1 = __uint_as_float(wv[i] & 0xFFFF0000u);
#pragma unroll
          for (int m = 0; m < MX; ++m) {
            const uint32_t xv = reinterpret_cast<const uint32_t*>(&xr[m][j])[i];
            acc[m] = fmaf(w0, __uint_as_float(xv << 16), acc[m]);
            acc[m] = fmaf(w1, __uint_as_float(xv & 0xFFFF0000u), acc[m]);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < MX; ++m) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
      }
      if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MX; ++m) red[warp][b * RB + r][m] = acc[m];
      }
    }
  }
}

// The router|fc1 GEMV into fused (FP32), and around it what needs only x and the logits: x quantized for the shared
// expert (spread over the blocks), each latent block's 16 features of every row quantized for the routed experts (one
// block-scale group: the BF16 latent the routing kernel would hand over, quantized), and (p.route) the last of the
// 32 logit blocks routes every row and plans the call.
__global__ void __launch_bounds__(kSplitWarps * 32) router_kernel(const Params p) {
  __shared__ float red[kSplitWarps][16][kMaxRows + 1];
  if (p.simt && p.M <= 4) router_block_simt<4>(p, red);
  else if (p.simt && p.M <= 8) router_block_simt<8>(p, red);
  else if (p.M <= 8) router_block<1>(p, red);
  else router_block<2>(p, red);
  const float gs_su = __ldg(p.gs_su);
  for (int i = blockIdx.x + gridDim.x * threadIdx.x; i < p.M * (kH / 16); i += gridDim.x * blockDim.x) {
    const int m = i / (kH / 16);
    quant_group_bf16(p.x + static_cast<size_t>(m) * kH, i % (kH / 16), gs_su, p.xsq + static_cast<size_t>(m) * (kH / 2),
                     p.xss + static_cast<size_t>(m) * (kH / 16));
  }
  __syncthreads();
  const int R0 = blockIdx.x * 16;
  if (R0 >= kE) {  // latent features R0 - 512 ..: one block-scale group of each row
    const int c = threadIdx.x;
    if (c < p.M) {
      float v[16];
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        float a = 0.f;
#pragma unroll
        for (int w = 0; w < kSplitWarps; ++w) a += red[w][r][c];
        p.fused[static_cast<size_t>(c) * kRouter + R0 + r] = a;
        v[r] = bf16r(a);
        if (p.lat_out) p.lat_out[static_cast<size_t>(c) * kL + R0 - kE + r] = __float2bfloat16_rn(a);
      }
      store_group(p.xq + static_cast<size_t>(c) * (kL / 2), p.xs + static_cast<size_t>(c) * (kL / 16), (R0 - kE) / 16, v,
                  __ldg(p.gs1));
    }
    return;
  }
  for (int i = threadIdx.x; i < 16 * p.M; i += kSplitWarps * 32) {
    const int r = i & 15, c = i >> 4;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < kSplitWarps; ++w) v += red[w][r][c];
    p.fused[static_cast<size_t>(c) * kRouter + R0 + r] = v;
  }
  if (!p.route) return;
  __shared__ int last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(p.counters + kRouteCounter, 1) == kE / 16 - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  __shared__ int sid[kMaxRows * 32], count[kE], slot[kE], start[kE + 1], fill[kE];
  __shared__ float sw[kMaxRows * 32];
  route_rows(p, sid, sw);
  __syncthreads();
  const int K = p.topk;
  // a padded row (a graph's dummy slot 0) names row 0's top expert at weight 0
  auto pad = [&](int t) { return p.slots && __ldg(p.slots + t / p.steps) == 0; };
  for (int i = threadIdx.x; i < p.M * K; i += blockDim.x) {
    const int t = i / K, k = i % K;
    p.ids_out[i] = pad(t) ? (p.pad_id >= 0 ? p.pad_id : sid[0]) : sid[t * 32 + k];
    p.wts_out[i] = pad(t) ? 0.f : sw[t * 32 + k];
  }
  make_plan(
      p, [&](int i) { return sid[(i / K) * 32 + i % K]; }, [&](int i) { return pad(i / K) ? 0.f : sw[(i / K) * 32 + i % K]; },
      p.M * K, count, slot, start, fill);
  if (threadIdx.x == 0) p.counters[kRouteCounter] = 0;
}

// ---------------------------------------------------------------- prep: the plan of given ids (and the latent)

__global__ void __launch_bounds__(kE) prep_kernel(const Params p) {
  grid_wait();
  grid_launch();
  if (blockIdx.x > 0) {  // the latent row, one block-scale group a thread (the routed experts alone)
    const int r = blockIdx.x - 1;
    if (threadIdx.x < kL / 16)
      quant_group_bf16(p.lat + static_cast<size_t>(r) * kL, threadIdx.x, __ldg(p.gs1), p.xq + static_cast<size_t>(r) * (kL / 2),
                       p.xs + static_cast<size_t>(r) * (kL / 16));
    return;
  }
  __shared__ int count[kE], slot[kE], start[kE + 1], fill[kE];
  make_plan(p, [&](int i) { return __ldg(p.ids + i); }, [&](int i) { return __ldg(p.wts + i); }, p.M * p.topk, count, slot,
            start, fill);
}

// ---------------------------------------------------------------- up: relu(W x)^2 -> NVFP4

// One warp's 16 rows at R0 of a [rows, K] NVFP4 weight against `cols` activation rows (row c of the B operand at
// xq + row(c) * K/2: list[c] / div, or c without a list), relu(alpha acc)^2 rounded to BF16 (the intermediate the
// reference and cutlass_fused_moe quantize), quantized with gs into hq / hs (row stride N/2, N/16) at list[c] (or c).
template <int NT, int K, int N>
__device__ __forceinline__ void up_warp(const uint8_t* w, const uint8_t* sc, int R0, const uint8_t* xq, const uint8_t* xs,
                                        const int* list, int div, int cols, float alpha, float gs, uint8_t* hq,
                                        uint8_t* hs, float (*hsm)[kMaxRows + 1], bool wait, bool p2) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if (p2) {
    p2_fp4_any<K>(w, sc, R0, xq, xs, cols, [&](int c) { return list ? __ldg(list + c) / div : c; }, 0, (K / 128 + 1) / 2,
                  wait, hsm);
  } else {
  Fp4Op<NT> op;
  op.rg = w + static_cast<size_t>(R0 + g) * (K / 2) + 16 * t;
  op.rg8 = op.rg + 8 * (K / 2);
  op.sr = sc + swz_row(R0 + g + 8 * (t & 1), K / 16);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int c = nt * 8 + g;
    const int r = c < cols ? (list ? __ldg(list + c) / div : c) : -1;
    op.xq[nt] = r >= 0 ? xq + static_cast<size_t>(r) * (K / 2) + 16 * t : nullptr;
    op.xs[nt] = r >= 0 ? xs + static_cast<size_t>(r) * (K / 16) : nullptr;
  }
  float acc[NT][4];
  run<NT, Fp4Step<NT>>(acc, op, 0, K / 128, wait);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) hsm[g + 8 * (i >> 1)][nt * 8 + 2 * t + (i & 1)] = acc[nt][i];
  }
  __syncwarp();
  // one (column, half) a lane: the warp's 16 rows are one block-scale group of the intermediate
  const int c = lane >> 1, h = lane & 1;
  if (c < cols) {
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float a = fmaxf(hsm[i][c] * alpha, 0.f);
      v[i] = bf16r(a * a);
    }
    uint32_t lo, hi;
    const uint8_t sf = quant16(v, gs, lo, hi);
    const int q = list ? __ldg(list + c) : c;
    const int st = R0 / 128, G = (R0 % 128) / 16;
    reinterpret_cast<uint32_t*>(hq + static_cast<size_t>(q) * (N / 2) + 64 * st)[code_word(G, h)] = h ? hi : lo;
    if (h == 0) hs[static_cast<size_t>(q) * (N / 16) + 8 * st + scale_byte(G)] = sf;
  }
  __syncwarp();
}

// Blocks [0, up_shared): the shared expert's 5376 rows for all of the call's rows; then 42 blocks per distinct slot.
__global__ void __launch_bounds__(kWarps * 32) up_kernel(const Params p) {
  __shared__ float hsm[kWarps][16][kMaxRows + 1];
  const int warp = threadIdx.x >> 5;
  if (blockIdx.x < p.up_shared) {
    const int R0 = blockIdx.x * kBlockRows + warp * 16;
    const float alpha = __ldg(p.alpha_su), gs = __ldg(p.gs_sd);
    if (p.M <= 8) up_warp<1, kH, kS>(p.wsu, p.ssu, R0, p.xsq, p.xss, nullptr, 1, p.M, alpha, gs, p.hsq, p.hss, hsm[warp], true, p.p2);
    else up_warp<2, kH, kS>(p.wsu, p.ssu, R0, p.xsq, p.xss, nullptr, 1, p.M, alpha, gs, p.hsq, p.hss, hsm[warp], true, p.p2);
    return;
  }
  const int b = blockIdx.x - p.up_shared, d = b / (kI / kBlockRows);
  grid_wait();  // the plan
  grid_launch();
  if (d >= __ldg(p.plan)) return;
  const int e = __ldg(p.plan + kPlanExp + d), s0 = __ldg(p.plan + kPlanStart + d);
  const int n = __ldg(p.plan + kPlanStart + d + 1) - s0;
  const int R0 = (b % (kI / kBlockRows)) * kBlockRows + warp * 16;
  const uint8_t* w = p.w1 + static_cast<size_t>(e) * kI * (kL / 2);
  const uint8_t* sc = p.s1 + static_cast<size_t>(e) * kI * (kL / 16);
  const float alpha = __ldg(p.alpha1 + e), gs = __ldg(p.gs2);
  for (int base = 0; base < n; base += kMaxRows) {  // more than 16 pairs on one expert: only with repeated ids
    const int cols = min(kMaxRows, n - base);
    const int* list = p.order + s0 + base;
    if (cols <= 8) up_warp<1, kL, kI>(w, sc, R0, p.xq, p.xs, list, p.topk, cols, alpha, gs, p.hq, p.hs, hsm[warp], false, p.p2);
    else up_warp<2, kL, kI>(w, sc, R0, p.xq, p.xs, list, p.topk, cols, alpha, gs, p.hq, p.hs, hsm[warp], false, p.p2);
  }
}

// ---------------------------------------------------------------- down: routed pairs, combined per row

template <int NT>
__device__ __forceinline__ void down_warp(const Params& p, const uint8_t* w, const uint8_t* sc, int R0, const int* list,
                                          int cols, float alpha) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  Fp4Op<NT> op;
  op.rg = w + static_cast<size_t>(R0 + g) * (kI / 2) + 16 * t;
  op.rg8 = op.rg + 8 * (kI / 2);
  op.sr = sc + swz_row(R0 + g + 8 * (t & 1), kI / 16);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int c = nt * 8 + g;
    const int q = c < cols ? __ldg(list + c) : -1;
    op.xq[nt] = q >= 0 ? p.hq + static_cast<size_t>(q) * (kI / 2) + 16 * t : nullptr;
    op.xs[nt] = q >= 0 ? p.hs + static_cast<size_t>(q) * (kI / 16) : nullptr;
  }
  float acc[NT][4];
  run<NT, Fp4Step<NT>>(acc, op, 0, kI / 128, true);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = nt * 8 + 2 * t + (i & 1);
      if (c < cols) {
        const int q = __ldg(list + c);
        p.y[static_cast<size_t>(q) * kL + R0 + g + 8 * (i >> 1)] = acc[nt][i] * alpha * __ldg(p.wts + q);
      }
    }
}

// blockIdx.y the distinct slot, blockIdx.x the 64-feature tile.
__global__ void __launch_bounds__(kWarps * 32) down_kernel(const Params p) {
  const int d = blockIdx.y, D = __ldg(p.plan);  // the plan: from the kernel before `up`, complete before up started
  __shared__ int last;
  if (d < D) {
    const int e = __ldg(p.plan + kPlanExp + d), s0 = __ldg(p.plan + kPlanStart + d);
    const int n = __ldg(p.plan + kPlanStart + d + 1) - s0;
    const int R0 = blockIdx.x * kBlockRows + (threadIdx.x >> 5) * 16;
    const uint8_t* w = p.w2 + static_cast<size_t>(e) * kL * (kI / 2);
    const uint8_t* sc = p.s2 + static_cast<size_t>(e) * kL * (kI / 16);
    const float alpha = __ldg(p.alpha2 + e);
    __shared__ float dt[kWarps][16][kMaxRows + 1];
    for (int base = 0; base < n; base += kMaxRows) {
      const int cols = min(kMaxRows, n - base);
      const int* list = p.order + s0 + base;
      if (p.p2) {
        float (*tile)[kMaxRows + 1] = dt[threadIdx.x >> 5];
        p2_fp4_any<kI>(w, sc, R0, p.hq, p.hs, cols, [&](int c) { return __ldg(list + c); }, 0, (kI / 128 + 1) / 2, base == 0,
                       tile);
        __syncwarp();
        for (int i = threadIdx.x & 31; i < 16 * cols; i += 32) {
          const int r = i & 15, c = i >> 4, q = __ldg(list + c);
          p.y[static_cast<size_t>(q) * kL + R0 + r] = tile[r][c] * alpha * __ldg(p.wts + q);
        }
        __syncwarp();
      } else if (cols <= 8) {
        down_warp<1>(p, w, sc, R0, list, cols, alpha);
      } else {
        down_warp<2>(p, w, sc, R0, list, cols, alpha);
      }
    }
    // the last of the call's D blocks on this 64-feature tile sums every row's pairs, in selection order
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(p.counters + blockIdx.x, 1) == D - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
  } else if (!(D == 0 && d == 0)) {
    return;  // D == 0 (every weight 0): block (x, 0) writes zeros
  } else {
    grid_wait();
  }
  for (int i = threadIdx.x; i < p.M * kBlockRows; i += kWarps * 32) {
    const int r = i / kBlockRows, f = blockIdx.x * kBlockRows + i % kBlockRows;
    float v = 0.f;
    for (int k = 0; k < p.topk; ++k) {
      const int q = r * p.topk + k;
      if (__ldg(p.wts + q) != 0.f) v += __ldcg(p.y + static_cast<size_t>(q) * kL + f);
    }
    p.routed[static_cast<size_t>(r) * kL + f] = __float2bfloat16_rn(v);
  }
  if (threadIdx.x == 0 && D > 0) p.counters[blockIdx.x] = 0;
}

// ---------------------------------------------------------------- out: fc2_latent (BF16) + shared down (NVFP4)

// 16 output rows a block; the 32 BF16 steps of fc2 (K 1024) and the 42 NVFP4 steps of shared down (K 5376) split
// over the 8 warps as one list of 74, summed in shared memory in warp order.
template <int NT>
__device__ __forceinline__ void out_block(const Params& p, float (*red)[16][kMaxRows + 1]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int R0 = blockIdx.x * 16;
  constexpr int A = kL / 32, B = kS / 128, T = A + B;
  const int v0 = warp * T / kSplitWarps, v1 = (warp + 1) * T / kSplitWarps;
  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;
  if (v0 < A) {
    Bf16Op<NT> op;
    op.rg = reinterpret_cast<const uint8_t*>(p.fc2 + static_cast<size_t>(R0 + g) * kL) + 16 * t;
    op.rg8 = op.rg + 8 * kL * 2;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int c = nt * 8 + g;
      op.x[nt] = c < p.M ? reinterpret_cast<const uint8_t*>(p.routed + static_cast<size_t>(c) * kL) + 16 * t : nullptr;
    }
    run<NT, Bf16Step<NT>>(acc, op, v0, min(v1, A), true);
  }
  if (v1 > A) {
    Fp4Op<NT> op;
    op.rg = p.wsd + static_cast<size_t>(R0 + g) * (kS / 2) + 16 * t;
    op.rg8 = op.rg + 8 * (kS / 2);
    op.sr = p.ssd + swz_row(R0 + g + 8 * (t & 1), kS / 16);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int c = nt * 8 + g;
      op.xq[nt] = c < p.M ? p.hsq + static_cast<size_t>(c) * (kS / 2) + 16 * t : nullptr;
      op.xs[nt] = c < p.M ? p.hss + static_cast<size_t>(c) * (kS / 16) : nullptr;
    }
    float acc4[NT][4];
    run<NT, Fp4Step<NT>>(acc4, op, max(v0, A) - A, v1 - A, true);
    const float alpha = __ldg(p.alpha_sd);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[nt][i] += alpha * acc4[nt][i];
  }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][g + 8 * (i >> 1)][nt * 8 + 2 * t + (i & 1)] = acc[nt][i];
}

// The paired-step split: fc2's 16 pairs (K 1024) and the shared down's 21 (K 5376) over the 8 warps as one list of 37;
// a warp's part of each matrix into its own tile, the shared one's scaled, summed in shared memory.
__device__ __forceinline__ void out_block2(const Params& p, float (*red)[16][kMaxRows + 1]) {
  __shared__ float part[kSplitWarps][16][kMaxRows + 1];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, R0 = blockIdx.x * 16;
  constexpr int A = kL / 64, B = kS / 256 + 1, T = A + B;  // 16 + 21 pairs (kS / 128 = 42 steps)
  const int v0 = warp * T / kSplitWarps, v1 = (warp + 1) * T / kSplitWarps, M = p.M;
  auto id = [](int c) { return c; };
  for (int i = lane; i < 16 * M; i += 32) red[warp][i & 15][i >> 4] = part[warp][i & 15][i >> 4] = 0.f;
  __syncwarp();
  if (v0 < A) {
    if (M <= 4) p2_bf16<1>(p.fc2, kL, R0, p.routed, M, id, v0, min(v1, A), true, red[warp]);
    else if (M <= 8) p2_bf16<2>(p.fc2, kL, R0, p.routed, M, id, v0, min(v1, A), true, red[warp]);
    else p2_bf16<4>(p.fc2, kL, R0, p.routed, M, id, v0, min(v1, A), true, red[warp]);
  }
  if (v1 > A) {
    p2_fp4_any<kS>(p.wsd, p.ssd, R0, p.hsq, p.hss, M, id, max(v0, A) - A, v1 - A, true, part[warp]);
    __syncwarp();
    const float alpha = __ldg(p.alpha_sd);
    for (int i = lane; i < 16 * M; i += 32) red[warp][i & 15][i >> 4] += alpha * part[warp][i & 15][i >> 4];
  }
  __syncwarp();
}

__global__ void __launch_bounds__(kSplitWarps * 32) out_kernel(const Params p) {
  __shared__ float red[kSplitWarps][16][kMaxRows + 1];
  if (p.p2) out_block2(p, red);
  else if (p.M <= 8) out_block<1>(p, red);
  else out_block<2>(p, red);
  __syncthreads();
  for (int i = threadIdx.x; i < 16 * p.M; i += kSplitWarps * 32) {
    const int r = i & 15, c = i >> 4;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < kSplitWarps; ++w) v += red[w][r][c];
    p.out[static_cast<size_t>(c) * kH + blockIdx.x * 16 + r] = __float2bfloat16_rn(v);
  }
}

// ---------------------------------------------------------------- host

// Launch with programmatic stream serialization (`pdl`): the kernel may start while the previous one finishes.
void launch(void (*kernel)(const Params), dim3 grid, dim3 block, cudaStream_t stream, bool pdl, const Params& p) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}

template <class T>
const T* ptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }

Params make_params(const torch::Tensor& x, const std::vector<torch::Tensor>& w, int M, bool shared = true) {
  // w: wr, fc2, w1, s1, w2, s2, wsu, ssu, wsd, ssd, alpha1, alpha2, gs1, gs2, alpha_su, alpha_sd, gs_su, gs_sd, counters
  TORCH_CHECK(w.size() == 19, "weights: 19 tensors");
  TORCH_CHECK(!shared || (x.dtype() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(0) == M && x.size(1) == kH),
              "x: contiguous bf16 [M, 4096]");
  TORCH_CHECK(M >= 1 && M <= kMaxRows, "1..16 rows");
  TORCH_CHECK(w[0].numel() == static_cast<int64_t>(kRouter) * kH && w[1].numel() == static_cast<int64_t>(kH) * kL, "router / fc2");
  // no NVFP4 experts (empty tensors): the block's 3-bit experts run outside (`shared_up`, `out` around w3_moe)
  const bool experts = w[2].numel() > 0;
  TORCH_CHECK(!experts || (w[2].numel() == static_cast<int64_t>(kE) * kI * kL / 2 && w[4].numel() == static_cast<int64_t>(kE) * kL * kI / 2), "expert weights");
  TORCH_CHECK(!experts || (w[3].nbytes() == static_cast<size_t>(kE) * kI * kL / 16 && w[5].nbytes() == static_cast<size_t>(kE) * kL * kI / 16), "expert scales");
  TORCH_CHECK(w[6].numel() == static_cast<int64_t>(kS) * kH / 2 && w[8].numel() == static_cast<int64_t>(kH) * kS / 2, "shared weights");
  TORCH_CHECK(w[7].nbytes() == static_cast<size_t>(kS) * kH / 16 && w[9].nbytes() == static_cast<size_t>(kH) * kS / 16, "shared scales");
  for (int i = 10; i < 18; ++i) TORCH_CHECK(w[i].dtype() == at::kFloat && w[i].is_contiguous(), "scales: fp32");
  TORCH_CHECK(w[18].dtype() == at::kInt && w[18].numel() > kRouteCounter, "counters");
  Params p{};
  p.x = shared ? ptr<__nv_bfloat16>(x) : nullptr;
  p.up_shared = shared ? kUpShared : 0;
  p.wr = ptr<__nv_bfloat16>(w[0]);
  p.fc2 = ptr<__nv_bfloat16>(w[1]);
  p.w1 = ptr<uint8_t>(w[2]);
  p.s1 = ptr<uint8_t>(w[3]);
  p.w2 = ptr<uint8_t>(w[4]);
  p.s2 = ptr<uint8_t>(w[5]);
  p.wsu = ptr<uint8_t>(w[6]);
  p.ssu = ptr<uint8_t>(w[7]);
  p.wsd = ptr<uint8_t>(w[8]);
  p.ssd = ptr<uint8_t>(w[9]);
  p.alpha1 = ptr<float>(w[10]);
  p.alpha2 = ptr<float>(w[11]);
  p.gs1 = ptr<float>(w[12]);
  p.gs2 = ptr<float>(w[13]);
  p.alpha_su = ptr<float>(w[14]);
  p.alpha_sd = ptr<float>(w[15]);
  p.gs_su = ptr<float>(w[16]);
  p.gs_sd = ptr<float>(w[17]);
  p.counters = static_cast<int*>(w[18].data_ptr());
  p.M = M;
  p.pad_id = -1;
  return p;
}

// The call's scratch, one byte tensor: the plan, the pairs' order, the activations in NVFP4, the pairs' outputs.
struct Layout {
  size_t plan, order, xq, xs, xsq, xss, hq, hs, hsq, hss, y, routed, total;
};

Layout layout(int M) {
  const size_t P = static_cast<size_t>(M) * kMaxTopk;
  Layout l{};
  size_t at = 0;
  auto take = [&](size_t bytes) {
    const size_t here = at;
    at += (bytes + 255) / 256 * 256;
    return here;
  };
  l.plan = take(kPlanWords * 4);
  l.order = take(P * 4);
  l.xq = take(M * kL / 2);
  l.xs = take(M * kL / 16);
  l.xsq = take(M * kH / 2);
  l.xss = take(M * kH / 16);
  l.hq = take(P * kI / 2);
  l.hs = take(P * kI / 16);
  l.hsq = take(M * kS / 2);
  l.hss = take(M * kS / 16);
  l.y = take(P * kL * 4);
  l.routed = take(M * kL * 2);
  l.total = at;
  return l;
}

void bind(Params& p, const torch::Tensor& ws, int M) {
  const Layout l = layout(M);
  TORCH_CHECK(ws.dtype() == at::kByte && static_cast<size_t>(ws.numel()) >= l.total, "workspace");
  uint8_t* b = ws.data_ptr<uint8_t>();
  p.plan = reinterpret_cast<int*>(b + l.plan);
  p.order = reinterpret_cast<int*>(b + l.order);
  p.xq = b + l.xq;
  p.xs = b + l.xs;
  p.xsq = b + l.xsq;
  p.xss = b + l.xss;
  p.hq = b + l.hq;
  p.hs = b + l.hs;
  p.hsq = b + l.hsq;
  p.hss = b + l.hss;
  p.y = reinterpret_cast<float*>(b + l.y);
  p.routed = reinterpret_cast<__nv_bfloat16*>(b + l.routed);
}

torch::Tensor routed_view(const torch::Tensor& ws, int M) {
  return ws.narrow(0, layout(M).routed, M * kL * 2).view(at::kBFloat16).view({M, kL});
}

void check_ids(const torch::Tensor& ids, const torch::Tensor& wts, int M) {
  TORCH_CHECK(ids.dtype() == at::kInt && ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == M, "ids: int32 [M, k]");
  TORCH_CHECK(ids.size(1) >= 1 && ids.size(1) <= kMaxTopk, "1..32 experts a row");
  TORCH_CHECK(wts.dtype() == at::kFloat && wts.is_contiguous() && wts.sizes() == ids.sizes(), "weights: fp32 like ids");
}

// up (+ down, + out by `stages`) after the plan.
void launch_rest(const Params& p, int stages, bool pdl, cudaStream_t stream) {
  const int slots = std::max(1, std::min(p.M * p.topk, kE));
  if (stages >= 2) launch(up_kernel, p.up_shared + (kI / kBlockRows) * slots, kWarps * 32, stream, pdl, p);
  if (stages >= 3) launch(down_kernel, dim3(kL / kBlockRows, slots), kWarps * 32, stream, pdl, p);
  if (stages >= 4) launch(out_kernel, kH / 16, kSplitWarps * 32, stream, pdl, p);
}

}  // namespace

// The router|fc1 GEMV for x [M, 4096] bf16 (M <= 16) -> [fused FP32 [M, 1536] (router logits | latent), ids int32
// [M, k], weights fp32 [M, k], the call's workspace]. With `route`, the kernel also routes each row to its top k
// (`moe.py`'s route_topk: bias for the choice, renormalized sigmoid weights; with `slots`, rows r of requests with
// slots[r / steps] == 0 take row 0's top expert at weight 0) and plans the call; else ids / weights are left
// unwritten (the caller routes from fused and passes its own to `block` with `plan`). Either way it quantizes x and the
// latent for the rest of the block. `simt`: the GEMV on the CUDA cores up to 8 rows.
std::vector<torch::Tensor> router(torch::Tensor x, std::vector<torch::Tensor> w, torch::Tensor bias, int64_t k,
                                  torch::Tensor slots, int64_t steps, bool route, bool pdl, bool simt, int64_t pad_id) {
  const int M = x.size(0);
  Params p = make_params(x, w, M);
  TORCH_CHECK(k >= 1 && k <= kMaxTopk, "1..32 experts a row");
  TORCH_CHECK(bias.dtype() == at::kFloat && bias.numel() == kE && bias.is_contiguous(), "bias: fp32 [512]");
  TORCH_CHECK(slots.numel() == 0 || (slots.dtype() == at::kInt && slots.is_contiguous() && steps >= 1 && slots.numel() * steps >= M),
              "slots: int32, a request every `steps` rows");
  auto fused = torch::empty({M, kRouter}, x.options().dtype(at::kFloat));
  auto ids = torch::empty({M, k}, x.options().dtype(at::kInt));
  auto wts = torch::empty({M, k}, x.options().dtype(at::kFloat));
  auto ws = torch::empty({static_cast<int64_t>(layout(M).total)}, x.options().dtype(at::kByte));
  auto lat = torch::empty({M, kL}, x.options());
  bind(p, ws, M);
  p.lat_out = reinterpret_cast<__nv_bfloat16*>(lat.data_ptr());
  p.pad_id = static_cast<int>(pad_id);
  p.fused = fused.data_ptr<float>();
  p.bias = bias.data_ptr<float>();
  p.slots = slots.numel() ? slots.data_ptr<int>() : nullptr;
  p.steps = static_cast<int>(steps);
  p.route = route;
  p.simt = simt;
  p.topk = static_cast<int>(k);
  p.ids_out = ids.data_ptr<int>();
  p.wts_out = wts.data_ptr<float>();
  launch(router_kernel, kRouter / 16, kSplitWarps * 32, c10::cuda::getCurrentCUDAStream(), pdl, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {fused, ids, wts, ws, lat};
}

// The block after `router`: x [M, 4096] bf16, its workspace, ids / weights [M, k] (int32 / fp32; weights
// renormalized, routed_scaling_factor rides in alpha2) -> bf16 [M, 4096] = fc2_latent(routed sum) + shared expert.
// `plan`: plan the call from ids (when `router` did not route). `stages` (profiling): 1 plan, 2 + up, 3 + down
// (returns the routed sum [M, 1024]), 4 everything. `pdl`: programmatic dependent launches. Graph-capturable;
// deterministic.
torch::Tensor block(torch::Tensor x, torch::Tensor ws, torch::Tensor ids, torch::Tensor wts, std::vector<torch::Tensor> w,
                    bool plan, int64_t stages, bool pdl, bool p2) {
  const int M = x.size(0);
  Params p = make_params(x, w, M);
  check_ids(ids, wts, M);
  bind(p, ws, M);
  p.ids = ids.data_ptr<int>();
  p.wts = wts.data_ptr<float>();
  p.topk = ids.size(1);
  p.p2 = p2;
  auto out = torch::empty({M, kH}, x.options());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (plan) launch(prep_kernel, 1, kE, stream, pdl, p);
  launch_rest(p, static_cast<int>(stages), pdl, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return stages >= 4 ? out : routed_view(ws, M);
}

// The routed experts alone for latent [M, 1024] bf16 and ids / weights [M, k] -> the routed sum, bf16 [M, 1024]
// (plan and latent quantization, up, down).
torch::Tensor routed(torch::Tensor lat, torch::Tensor ids, torch::Tensor wts, std::vector<torch::Tensor> w, bool pdl, bool p2) {
  const int M = lat.size(0);
  Params p = make_params(lat, w, M, false);
  TORCH_CHECK(lat.dtype() == at::kBFloat16 && lat.is_contiguous() && lat.dim() == 2 && lat.size(1) == kL, "lat: bf16 [M, 1024]");
  check_ids(ids, wts, M);
  auto ws = torch::empty({static_cast<int64_t>(layout(M).total)}, lat.options().dtype(at::kByte));
  bind(p, ws, M);
  p.lat = ptr<__nv_bfloat16>(lat);
  p.ids = ids.data_ptr<int>();
  p.wts = wts.data_ptr<float>();
  p.topk = ids.size(1);
  p.p2 = p2;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch(prep_kernel, 1 + M, kE, stream, pdl, p);
  launch_rest(p, 3, pdl, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return routed_view(ws, M);
}

// The block around 3-bit routed experts (`kernels/w3_moe.py`, which reads the router's BF16 latent and ids): after
// `router`, `shared_up` (the shared expert's up, relu^2 and NVFP4 quantization), then the caller's routed sum, then
// `out` (fc2_latent of that sum plus the shared expert's down) -> bf16 [M, 4096].
void shared_up(torch::Tensor x, torch::Tensor ws, std::vector<torch::Tensor> w, bool pdl, bool p2) {
  const int M = x.size(0);
  Params p = make_params(x, w, M);
  bind(p, ws, M);
  p.topk = 1;
  p.p2 = p2;
  launch(up_kernel, kUpShared, kWarps * 32, c10::cuda::getCurrentCUDAStream(), pdl, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor out(torch::Tensor x, torch::Tensor ws, torch::Tensor routed, std::vector<torch::Tensor> w, bool pdl, bool p2) {
  const int M = x.size(0);
  Params p = make_params(x, w, M);
  TORCH_CHECK(routed.dtype() == at::kBFloat16 && routed.is_contiguous() && routed.dim() == 2 && routed.size(0) == M &&
              routed.size(1) == kL, "routed: contiguous bf16 [M, 1024]");
  bind(p, ws, M);
  p.routed = reinterpret_cast<__nv_bfloat16*>(routed.data_ptr());
  p.topk = 1;
  p.p2 = p2;
  auto o = torch::empty({M, kH}, x.options());
  p.out = reinterpret_cast<__nv_bfloat16*>(o.data_ptr());
  launch(out_kernel, kH / 16, kSplitWarps * 32, c10::cuda::getCurrentCUDAStream(), pdl, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_up", &shared_up, "Nemotron MoE shared expert's up (relu^2, NVFP4) after the router, 1..16 rows");
  m.def("out", &out, "Nemotron MoE fc2_latent of a routed sum plus the shared expert's down, 1..16 rows");
  m.def("router", &router, "Nemotron MoE router|fc1 GEMV (+ routing and planning), 1..16 rows (SM120/121)");
  m.def("block", &block, "Nemotron latent MoE block after the router, NVFP4 experts, 1..16 rows (SM120/121)");
  m.def("routed", &routed, "Nemotron routed NVFP4 experts alone, 1..16 rows (SM120/121)");
}
