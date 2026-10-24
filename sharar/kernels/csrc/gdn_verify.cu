// The Gated DeltaNet recurrence of a speculative verify, and the commit that replays its accepted prefix
// (sharar/kernels/gdn_verify.py), on SM120/121.
//
// A block is one sequence's value head: four warps, lane l of warp w owns value row v = 32 w + l of the head's
// FP32 state [V, K] and holds its 128 key entries in registers. Every token then needs no cross-lane reduction:
// h *= exp(g); u = h . k; v' = (v - u) * beta; h += k v'; o = h . q, each dot product four in-lane FMA chains over
// k mod 4 summed pairwise. The token's normalized q and k (L2 over the head's 128 keys, one warp each) and its
// gate and beta are computed once and broadcast through shared memory. The verify starts from the committed state
// and leaves it untouched; it writes each token's output and, for the commit, the raw inputs (pre-norm k, v, gate,
// beta) to the round's ring, or the state after every token. The commit kernel replays a ring's first `count`
// tokens with the same arithmetic (`step`), so the committed state is the verify's own, bit for bit.

#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

constexpr int K = 128;  // key dim
constexpr int V = 128;  // value dim
constexpr int THREADS = V;

// L2 normalization of a 128-vector held four per lane (elements 4 lane .. + 3), in place; one warp.
__device__ __forceinline__ void l2norm4(float (&x)[4]) {
  float ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = 1.f / sqrtf(ss + 1e-6f);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= inv;
}

// One token on a lane's state row h: h *= eg; u = h . kn; returns v' = (v - u) * beta and h += kn v'.
__device__ __forceinline__ float step(float (&h)[K], const float* __restrict__ kn, float eg, float v, float beta) {
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int i = 0; i < K / 4; ++i) {
    const float4 kv = reinterpret_cast<const float4*>(kn)[i];
    h[4 * i] *= eg;
    h[4 * i + 1] *= eg;
    h[4 * i + 2] *= eg;
    h[4 * i + 3] *= eg;
    a0 = fmaf(h[4 * i], kv.x, a0);
    a1 = fmaf(h[4 * i + 1], kv.y, a1);
    a2 = fmaf(h[4 * i + 2], kv.z, a2);
    a3 = fmaf(h[4 * i + 3], kv.w, a3);
  }
  const float vp = (v - ((a0 + a1) + (a2 + a3))) * beta;
#pragma unroll
  for (int i = 0; i < K / 4; ++i) {
    const float4 kv = reinterpret_cast<const float4*>(kn)[i];
    h[4 * i] = fmaf(kv.x, vp, h[4 * i]);
    h[4 * i + 1] = fmaf(kv.y, vp, h[4 * i + 1]);
    h[4 * i + 2] = fmaf(kv.z, vp, h[4 * i + 2]);
    h[4 * i + 3] = fmaf(kv.w, vp, h[4 * i + 3]);
  }
  return vp;
}

__device__ __forceinline__ float dot(const float (&h)[K], const float* __restrict__ x) {
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int i = 0; i < K / 4; ++i) {
    const float4 xv = reinterpret_cast<const float4*>(x)[i];
    a0 = fmaf(h[4 * i], xv.x, a0);
    a1 = fmaf(h[4 * i + 1], xv.y, a1);
    a2 = fmaf(h[4 * i + 2], xv.z, a2);
    a3 = fmaf(h[4 * i + 3], xv.w, a3);
  }
  return (a0 + a1) + (a2 + a3);
}

__device__ __forceinline__ void load_row(float (&h)[K], const float* __restrict__ src) {
#pragma unroll
  for (int i = 0; i < K / 4; ++i) {
    const float4 x = reinterpret_cast<const float4*>(src)[i];
    h[4 * i] = x.x; h[4 * i + 1] = x.y; h[4 * i + 2] = x.z; h[4 * i + 3] = x.w;
  }
}

template <typename T>
__device__ __forceinline__ void store_row(T* __restrict__ dst, const float (&h)[K]) {
  if constexpr (std::is_same_v<T, float>) {
#pragma unroll
    for (int i = 0; i < K / 4; ++i) reinterpret_cast<float4*>(dst)[i] = make_float4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
  } else {
#pragma unroll
    for (int i = 0; i < K / 2; ++i) reinterpret_cast<__nv_bfloat162*>(dst)[i] = __floats2bfloat162_rn(h[2 * i], h[2 * i + 1]);
  }
}


// A warp's 32 state rows (contiguous: 16 KB) to and from their lanes (lane l holds row l) through an 8-row tile
// [8][PITCH] floats: coalesced 512-byte row reads and writes instead of a lane walking its own row (whose 32 rows in
// flight per warp overflowed L1 from eight requests on). PITCH = K + 4 keeps the lanes' float4 tile reads conflict-free.
constexpr int PITCH = K + 4;

__device__ __forceinline__ void load_rows(float (&h)[K], const float* __restrict__ rows, float* tile, int lane) {
  float4 next[8];  // the next pass's pieces, in flight while this pass's go through the tile
#pragma unroll
  for (int i = 0; i < 8; ++i) next[i] = __ldcs(reinterpret_cast<const float4*>(rows + i * K) + lane);
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
#pragma unroll
    for (int i = 0; i < 8; ++i) *reinterpret_cast<float4*>(tile + i * PITCH + 4 * lane) = next[i];
    if (pass < 3) {
#pragma unroll
      for (int i = 0; i < 8; ++i) next[i] = __ldcs(reinterpret_cast<const float4*>(rows + ((pass + 1) * 8 + i) * K) + lane);
    }
    __syncwarp();
    if ((lane >> 3) == pass) {
      const float* r = tile + (lane & 7) * PITCH;
#pragma unroll
      for (int j = 0; j < K / 4; ++j) {
        const float4 x = reinterpret_cast<const float4*>(r)[j];
        h[4 * j] = x.x; h[4 * j + 1] = x.y; h[4 * j + 2] = x.z; h[4 * j + 3] = x.w;
      }
    }
    __syncwarp();
  }
}

__device__ __forceinline__ void store_rows(float* __restrict__ rows, const float (&h)[K], float* tile, int lane) {
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    if ((lane >> 3) == pass) {
      float* r = tile + (lane & 7) * PITCH;
#pragma unroll
      for (int j = 0; j < K / 4; ++j) reinterpret_cast<float4*>(r)[j] = make_float4(h[4 * j], h[4 * j + 1], h[4 * j + 2], h[4 * j + 3]);
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 8; ++i)
      reinterpret_cast<float4*>(rows + (pass * 8 + i) * K)[lane] = *reinterpret_cast<const float4*>(tile + i * PITCH + 4 * lane);
    __syncwarp();
  }
}

__device__ __forceinline__ float bf(const __nv_bfloat16* p) { return __bfloat162float(*p); }

struct VerifyParams {
  const __nv_bfloat16 *q, *k, *v, *a, *b;  // tokens on their own strides (elements)
  int64_t stride_q, stride_k, stride_v, stride_a, stride_b;
  const float* A_log;
  const __nv_bfloat16* dt_bias;  // BF16, or FP32 in dt_bias_f
  const float* dt_bias_f;
  float softplus_beta, softplus_threshold, scale;
  const float* h0;          // committed states: slot pitch h0_stride, then [HV, V, K]
  int64_t h0_stride;
  const int32_t* h0_indices;
  const int32_t* cu_seqlens;  // [N + 1], or null: N sequences of T tokens
  int T;
  __nv_bfloat16* o;         // [tokens, HV, V]
  // the state after every token (`intermediate`), or null
  void* inter;
  bool inter_bf16;
  const int32_t* inter_indices;
  int64_t inter_steps;       // tokens a slot of `inter` holds
  // the commit ring, or null: raw k [slot, H, L, K], raw v [slot, HV, L, V], gate and beta [slot, HV, L]
  __nv_bfloat16 *ring_k, *ring_v;
  float *ring_g, *ring_beta;
  int64_t ring_k_slot, ring_v_slot, ring_g_slot, ring_beta_slot;
  int ring_len;
  const int32_t* ring_indices;  // the ring slot of each sequence (h0_indices or the scratch indices)
  int H, HV;
};

constexpr int TB = 16;  // tokens staged at a time: their normalized q and k, gates and betas in shared memory

__global__ void __launch_bounds__(THREADS) verify_kernel(const VerifyParams p) {
  __shared__ __align__(16) float sk[TB][K], sq[TB][K];
  __shared__ __align__(16) float tiles[THREADS / 32][8 * PITCH];
  __shared__ float sg[TB], sbeta[TB];
  const int n = blockIdx.x, hv = blockIdx.y, head = hv / (p.HV / p.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row = threadIdx.x;  // row: this lane's value index
  int bos, T;
  if (p.cu_seqlens) {
    bos = p.cu_seqlens[n];
    T = p.cu_seqlens[n + 1] - bos;
  } else {
    bos = n * p.T;
    T = p.T;
  }
  const int64_t slot = p.h0_indices[n];
  const int64_t ring_slot = p.ring_k ? p.ring_indices[n] : -1;
  const int64_t inter_slot = p.inter ? p.inter_indices[n] : -1;
  const float neg_a = -__expf(p.A_log[hv]);
  const float dtb = p.dt_bias_f ? p.dt_bias_f[hv] : __bfloat162float(p.dt_bias[hv]);
  const bool ring_head = hv % (p.HV / p.H) == 0;  // one value head of each key head writes the raw k
  float h[K];
  if (slot >= 0) {  // the committed state is not written by the producer: read it before the dependency wait
    load_rows(h, p.h0 + slot * p.h0_stride + (static_cast<int64_t>(hv) * V + 32 * warp) * K, tiles[warp], lane);
  } else {
#pragma unroll
    for (int i = 0; i < K; ++i) h[i] = 0.f;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");  // the producer (the verify convolution) has finished
  for (int t0 = 0; t0 < T; t0 += TB) {
    const int tn = min(TB, T - t0);
    // stage: warp w normalizes the q and k of tokens w, w + 4, ...; threads 0..tn-1 form the gates and betas
    for (int j = warp; j < 2 * tn; j += THREADS / 32) {
      const int t = t0 + (j >> 1);
      const bool isq = j & 1;
      const int64_t tok = bos + t;
      const __nv_bfloat16* src = (isq ? p.q + tok * p.stride_q : p.k + tok * p.stride_k) + head * K + 4 * lane;
      const uint2 raw = *reinterpret_cast<const uint2*>(src);
      const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
      float x[4] = {__low2float(r2[0]), __high2float(r2[0]), __low2float(r2[1]), __high2float(r2[1])};
      if (!isq && ring_slot >= 0 && t < p.ring_len && ring_head)
        *reinterpret_cast<uint2*>(p.ring_k + ring_slot * p.ring_k_slot + (static_cast<int64_t>(head) * p.ring_len + t) * K + 4 * lane) = raw;
      l2norm4(x);
      if (isq) {
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] *= p.scale;
      }
      *reinterpret_cast<float4*>((isq ? sq[j >> 1] : sk[j >> 1]) + 4 * lane) = make_float4(x[0], x[1], x[2], x[3]);
    }
    if (threadIdx.x < tn) {
      const int t = t0 + threadIdx.x;
      const int64_t tok = bos + t;
      const float x = bf(p.a + tok * p.stride_a + hv) + dtb;
      const float bx = p.softplus_beta * x;
      const float sp = bx <= p.softplus_threshold ? (1.f / p.softplus_beta) * logf(1.f + __expf(bx)) : x;
      const float g = neg_a * sp;
      const float beta = 1.f / (1.f + __expf(-bf(p.b + tok * p.stride_b + hv)));
      sg[threadIdx.x] = g;
      sbeta[threadIdx.x] = beta;
      if (ring_slot >= 0 && t < p.ring_len) {
        p.ring_g[ring_slot * p.ring_g_slot + static_cast<int64_t>(hv) * p.ring_len + t] = g;
        p.ring_beta[ring_slot * p.ring_beta_slot + static_cast<int64_t>(hv) * p.ring_len + t] = beta;
      }
    }
    __nv_bfloat16 vraw[TB];
#pragma unroll
    for (int j = 0; j < TB; ++j) {
      const int t = t0 + j;
      if (j < tn) {
        vraw[j] = p.v[(bos + t) * p.stride_v + static_cast<int64_t>(hv) * V + row];
        if (ring_slot >= 0 && t < p.ring_len)
          p.ring_v[ring_slot * p.ring_v_slot + (static_cast<int64_t>(hv) * p.ring_len + t) * V + row] = vraw[j];
      }
    }
    __syncthreads();
#pragma unroll 1
    for (int j = 0; j < tn; ++j) {
      const int t = t0 + j;
      step(h, sk[j], __expf(sg[j]), __bfloat162float(vraw[j]), sbeta[j]);
      p.o[((bos + t) * p.HV + hv) * V + row] = __float2bfloat16(dot(h, sq[j]));
      if (inter_slot >= 0) {
        const int64_t at = ((inter_slot * p.inter_steps + t) * p.HV + hv) * V * K + static_cast<int64_t>(row) * K;
        if (p.inter_bf16) store_row(static_cast<__nv_bfloat16*>(p.inter) + at, h);
        else store_row(static_cast<float*>(p.inter) + at, h);
      }
    }
    __syncthreads();  // the next block of tokens overwrites the staged ones
  }
  asm volatile("griddepcontrol.launch_dependents;");
}

// The commit: the first counts[row] ring tokens of each accepted row over its committed state, every layer.
struct CommitParams {
  float* state;             // [layers] x layer pitch, slot pitch, then [HV, V, K]
  int64_t state_layer, state_slot;
  const __nv_bfloat16 *raw_k, *raw_v;  // [layers, B, H, S, K], [layers, B, HV, S, V]
  const float *gates, *betas;          // [layers, B, HV, S]
  const int32_t *slots, *counts;
  float* out;               // the state itself, or a separate [layers, B, HV, V, K] (the test oracle)
  bool inplace;
  int B, H, HV, S;
};

__global__ void __launch_bounds__(THREADS) commit_kernel(const CommitParams p) {
  __shared__ __align__(16) float sk[TB][K];
  __shared__ __align__(16) float tiles[THREADS / 32][8 * PITCH];
  const int r = blockIdx.x, lh = blockIdx.y, layer = lh / p.HV, hv = lh % p.HV, head = hv / (p.HV / p.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row = threadIdx.x;
  const int64_t slot = p.slots[r];
  const int count = p.counts[r];
  float h[K];
  load_rows(h, p.state + layer * p.state_layer + slot * p.state_slot + (static_cast<int64_t>(hv) * V + 32 * warp) * K,
            tiles[warp], lane);
  const int64_t lb = static_cast<int64_t>(layer) * p.B + r;
  for (int t0 = 0; t0 < count; t0 += TB) {
    const int tn = min(TB, count - t0);
    // warp w normalizes the keys of tokens w, w + 4, ... (the verify's `l2norm4`), then every lane runs them in turn
    for (int j = warp; j < tn; j += THREADS / 32) {
      const uint2 raw = *reinterpret_cast<const uint2*>(p.raw_k + ((lb * p.H + head) * p.S + t0 + j) * K + 4 * lane);
      const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
      float x[4] = {__low2float(r2[0]), __high2float(r2[0]), __low2float(r2[1]), __high2float(r2[1])};
      l2norm4(x);
      *reinterpret_cast<float4*>(sk[j] + 4 * lane) = make_float4(x[0], x[1], x[2], x[3]);
    }
    __syncthreads();
#pragma unroll 1
    for (int j = 0; j < tn; ++j) {
      const int64_t gi = (lb * p.HV + hv) * p.S + t0 + j;
      step(h, sk[j], __expf(p.gates[gi]), __bfloat162float(p.raw_v[gi * V + row]), p.betas[gi]);
    }
    __syncthreads();
  }
  float* dst = p.inplace ? p.out + layer * p.state_layer + slot * p.state_slot + (static_cast<int64_t>(hv) * V + 32 * warp) * K
                         : p.out + ((lb * p.HV + hv) * V + 32 * warp) * static_cast<int64_t>(K);
  store_rows(dst, h, tiles[warp], lane);
}

const __nv_bfloat16* bf16_ptr(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }

}  // namespace

torch::Tensor gdn_verify(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
                         torch::Tensor A_log, torch::Tensor dt_bias, double softplus_beta, double softplus_threshold, double scale,
                         torch::Tensor h0, torch::Tensor h0_indices, c10::optional<torch::Tensor> cu_seqlens, int64_t n_seqs,
                         c10::optional<torch::Tensor> inter, c10::optional<torch::Tensor> inter_indices,
                         c10::optional<torch::Tensor> ring_k, c10::optional<torch::Tensor> ring_v, c10::optional<torch::Tensor> ring_g,
                         c10::optional<torch::Tensor> ring_beta, c10::optional<torch::Tensor> ring_indices) {
  // q, k [1, T, H, K]; v [1, T, HV, V] (token strides free, heads contiguous); a, b [T, HV] (token stride free)
  const int64_t tokens = q.size(1);
  const int H = static_cast<int>(q.size(2)), HV = static_cast<int>(v.size(2));
  TORCH_CHECK(q.size(3) == K && v.size(3) == V && q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1 &&
              q.stride(2) == K && k.stride(2) == K && v.stride(2) == V && HV % H == 0, "gdn_verify: K = V = 128 heads");
  TORCH_CHECK(a.stride(-1) == 1 && b.stride(-1) == 1 && h0.dtype() == torch::kFloat && h0.stride(-1) == 1, "gdn_verify: layouts");
  TORCH_CHECK(h0.size(1) == HV && h0.size(2) == V && h0.size(3) == K && h0.stride(1) == V * K, "gdn_verify: FP32 [slots, HV, V, K] states");
  auto o = torch::empty({1, tokens, HV, V}, q.options());
  VerifyParams p{};
  p.q = bf16_ptr(q); p.k = bf16_ptr(k); p.v = bf16_ptr(v); p.a = bf16_ptr(a); p.b = bf16_ptr(b);
  p.stride_q = q.stride(1); p.stride_k = k.stride(1); p.stride_v = v.stride(1); p.stride_a = a.stride(-2); p.stride_b = b.stride(-2);
  p.A_log = A_log.data_ptr<float>();
  TORCH_CHECK(A_log.dtype() == torch::kFloat && (dt_bias.dtype() == torch::kBFloat16 || dt_bias.dtype() == torch::kFloat) &&
              q.dtype() == torch::kBFloat16 && k.dtype() == q.dtype() && v.dtype() == q.dtype() && a.dtype() == q.dtype() &&
              b.dtype() == q.dtype(), "gdn_verify: BF16 inputs, FP32 A_log");
  if (dt_bias.dtype() == torch::kFloat) p.dt_bias_f = dt_bias.data_ptr<float>();
  else p.dt_bias = bf16_ptr(dt_bias);
  p.softplus_beta = static_cast<float>(softplus_beta);
  p.softplus_threshold = static_cast<float>(softplus_threshold);
  p.scale = static_cast<float>(scale);
  p.h0 = h0.data_ptr<float>();
  p.h0_stride = h0.stride(0);
  p.h0_indices = h0_indices.data_ptr<int32_t>();
  p.cu_seqlens = cu_seqlens ? cu_seqlens->data_ptr<int32_t>() : nullptr;
  p.T = cu_seqlens ? 0 : static_cast<int>(tokens / n_seqs);
  p.o = reinterpret_cast<__nv_bfloat16*>(o.data_ptr());
  if (inter) {
    TORCH_CHECK(inter->stride(-1) == 1 && (inter->dtype() == torch::kFloat || inter->dtype() == torch::kBFloat16), "gdn_verify: intermediate states");
    p.inter = inter->data_ptr();
    p.inter_bf16 = inter->dtype() == torch::kBFloat16;
    p.inter_indices = inter_indices->data_ptr<int32_t>();
    p.inter_steps = inter->stride(0) / (static_cast<int64_t>(HV) * V * K);
  }
  if (ring_k) {
    p.ring_k = reinterpret_cast<__nv_bfloat16*>(ring_k->data_ptr());
    p.ring_v = reinterpret_cast<__nv_bfloat16*>(ring_v->data_ptr());
    p.ring_g = ring_g->data_ptr<float>();
    p.ring_beta = ring_beta->data_ptr<float>();
    p.ring_k_slot = ring_k->stride(0); p.ring_v_slot = ring_v->stride(0);
    p.ring_g_slot = ring_g->stride(0); p.ring_beta_slot = ring_beta->stride(0);
    p.ring_len = static_cast<int>(ring_v->size(-2));
    p.ring_indices = ring_indices->data_ptr<int32_t>();
  }
  p.H = H;
  p.HV = HV;
  // programmatic dependent launch: the block's setup overlaps its producer's tail (`griddepcontrol.wait` fences)
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(n_seqs), static_cast<unsigned>(HV));
  config.blockDim = dim3(THREADS);
  config.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, verify_kernel, p));
  return o;
}

void gdn_commit(torch::Tensor state, torch::Tensor raw_k, torch::Tensor raw_v, torch::Tensor gates, torch::Tensor betas,
                torch::Tensor slots, torch::Tensor counts, torch::Tensor out) {
  // state [layers, slots, HV, V, K] FP32; raw_k [layers, B, H, S, K]; raw_v [layers, B, HV, S, V]; gates, betas [layers, B, HV, S]
  TORCH_CHECK(state.dtype() == torch::kFloat && state.size(3) == V && state.size(4) == K && state.stride(4) == 1 &&
              state.stride(2) == V * K, "gdn_commit: FP32 [layers, slots, HV, V, K] states");
  TORCH_CHECK(raw_k.is_contiguous() && raw_v.is_contiguous() && gates.is_contiguous() && betas.is_contiguous(), "gdn_commit: rings");
  const int layers = static_cast<int>(raw_v.size(0));
  CommitParams p{};
  p.state = state.data_ptr<float>();
  p.state_layer = state.stride(0);
  p.state_slot = state.stride(1);
  p.raw_k = bf16_ptr(raw_k);
  p.raw_v = bf16_ptr(raw_v);
  p.gates = gates.data_ptr<float>();
  p.betas = betas.data_ptr<float>();
  p.slots = slots.data_ptr<int32_t>();
  p.counts = counts.data_ptr<int32_t>();
  p.inplace = out.data_ptr() == state.data_ptr();
  p.out = out.data_ptr<float>();
  p.B = static_cast<int>(raw_v.size(1));
  p.H = static_cast<int>(raw_k.size(2));
  p.HV = static_cast<int>(raw_v.size(2));
  p.S = static_cast<int>(raw_v.size(3));
  commit_kernel<<<dim3(static_cast<unsigned>(slots.numel()), static_cast<unsigned>(layers * p.HV)), THREADS, 0,
                  c10::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_verify", &gdn_verify, "Gated DeltaNet verify recurrence, K = V = 128 (SM120/121)");
  m.def("gdn_commit", &gdn_commit, "replay of a verify ring's accepted prefix over the committed states");
}
