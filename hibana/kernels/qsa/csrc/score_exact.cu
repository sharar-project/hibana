// SPDX-License-Identifier: Apache-2.0
// QSA group scores with the `original` scorer's arithmetic (selection.py `_score_kernel`), tiled so a block reuses
// each compressed key across 64 query rows. That kernel's layout (Triton 3.7: [64 groups, 128] keys, size per thread
// [1, 8], 16 lanes along the dimension) fixes its FP32 order: each lane sums its 8 dimensions in order (the first a
// product, then FMAs; BF16 products are exact in FP32, so an FMA rounds as a product and an add would), and the
// 16 lane sums meet in a butterfly that adds lanes 8 apart first, then 4, 2 and 1. Each head's sum is relu'd and
// added to an FP32 accumulator from +0 in head order, then scaled. This kernel evaluates the same tree per
// (row, group) in one thread, so every score has the same bits; tests/gpu/test_qsa_exact.py checks them against
// the Triton kernel.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

namespace qsa_exact {
constexpr int HEADS = 4, DIM = 128, RATIO = 4;
constexpr int ROWS = 64, GROUPS = 64, THREADS = 256;  // block tile; each thread 4 rows x 4 groups
constexpr int PITCH = DIM + 4;                        // FP32 smem row pitch (16 B pad: no bank conflicts)

// One lane's 8-dimension chain for 4 rows x 4 groups: product of the first, then FMAs in dimension order.
__device__ __forceinline__ void chain(const float* q, const float* k, int c, int ty, int tx, float (&p)[4][4]) {
  const int d0 = c * 8;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float4 qv[4], kv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = *reinterpret_cast<const float4*>(q + (ty + 16 * i) * PITCH + d0 + 4 * half);
#pragma unroll
    for (int j = 0; j < 4; ++j) kv[j] = *reinterpret_cast<const float4*>(k + (tx + 16 * j) * PITCH + d0 + 4 * half);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (half == 0) p[i][j] = __fmul_rn(qv[i].x, kv[j].x);
        else p[i][j] = __fmaf_rn(kv[j].x, qv[i].x, p[i][j]);
        p[i][j] = __fmaf_rn(kv[j].y, qv[i].y, p[i][j]);
        p[i][j] = __fmaf_rn(kv[j].z, qv[i].z, p[i][j]);
        p[i][j] = __fmaf_rn(kv[j].w, qv[i].w, p[i][j]);
      }
  }
}

__device__ __forceinline__ void add(float (&a)[4][4], const float (&b)[4][4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) a[i][j] = __fadd_rn(a[i][j], b[i][j]);
}

// s_i = p_i + p_{i+8}
__device__ __forceinline__ void pair(const float* q, const float* k, int c, int ty, int tx, float (&s)[4][4]) {
  float p[4][4];
  chain(q, k, c, ty, tx, s);
  chain(q, k, c + 8, ty, tx, p);
  add(s, p);
}

// t_i = s_i + s_{i+4}
__device__ __forceinline__ void quad(const float* q, const float* k, int c, int ty, int tx, float (&t)[4][4]) {
  float s[4][4];
  pair(q, k, c, ty, tx, t);
  pair(q, k, c + 4, ty, tx, s);
  add(t, s);
}

// u_i = t_i + t_{i+2}
__device__ __forceinline__ void octet(const float* q, const float* k, int c, int ty, int tx, float (&u)[4][4]) {
  float t[4][4];
  quad(q, k, c, ty, tx, u);
  quad(q, k, c + 2, ty, tx, t);
  add(u, t);
}

// BF16 x 8 -> FP32 smem
__device__ __forceinline__ void stage(float* dst, const uint4& raw) {
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
  *reinterpret_cast<float4*>(dst) =
      make_float4(__bfloat162float(b[0]), __bfloat162float(b[1]), __bfloat162float(b[2]), __bfloat162float(b[3]));
  *reinterpret_cast<float4*>(dst + 4) =
      make_float4(__bfloat162float(b[4]), __bfloat162float(b[5]), __bfloat162float(b[6]), __bfloat162float(b[7]));
}

// A block scores one tile of rows against TILES_PER_BLOCK group tiles. The next head's queries (or the next group
// tile's keys and first head) load into registers while the current head computes.
constexpr int LOADS = ROWS * 16 / THREADS;  // 16 B loads per thread per operand tile
static_assert(ROWS == GROUPS, "one load shape for both operands");

__global__ __launch_bounds__(THREADS, 1) void score_kernel(
    const __nv_bfloat16* __restrict__ query, int64_t stride_qr, int64_t stride_qh,
    const __nv_bfloat16* __restrict__ aux, int64_t stride_page, int64_t key_offset,
    const int32_t* __restrict__ table, const int32_t* __restrict__ base, const int64_t* __restrict__ positions,
    const int32_t* __restrict__ tiles, float* __restrict__ out, int64_t stride_or, int gcap, int tiles_per_block,
    float scale) {
  extern __shared__ float smem[];
  float* qs = smem;                    // [ROWS][PITCH], one head at a time
  float* ks = smem + ROWS * PITCH;     // [GROUPS][PITCH]
  __shared__ int visible[ROWS];
  __shared__ int max_visible;
  // a warp covers 4 x 8 (row, group) threads: its shared loads are one wavefront each
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  const int ty = (warp / 2) * 4 + lane / 8, tx = (warp % 2) * 8 + lane % 8;
  const int row0 = tiles[2 * blockIdx.x], rows = tiles[2 * blockIdx.x + 1];
  if (t == 0) max_visible = 0;
  __syncthreads();
  if (t < ROWS) {
    const int v = t < rows ? static_cast<int>((positions[row0 + t] + 1) / RATIO) : 0;
    visible[t] = v;
    atomicMax(&max_visible, v);
  }
  __syncthreads();
  const int first = blockIdx.y * tiles_per_block;
  const int last = min(first + tiles_per_block, (gcap + GROUPS - 1) / GROUPS);
  const int live_end = min(last, (max_visible + GROUPS - 1) / GROUPS);  // tiles past it are all -inf
  for (int tile = max(first, live_end); tile < last; ++tile)
    for (int e = t; e < rows * GROUPS; e += THREADS) {
      const int r = e / GROUPS, g = tile * GROUPS + e % GROUPS;
      if (g < gcap) out[(row0 + r) * stride_or + g] = -CUDART_INF_F;
    }
  if (first >= live_end) return;
  const int64_t seq = base[row0];  // a tile's rows share a sequence (the caller's tiling)
  const int mv = max_visible;
  auto load_k = [&](int tile, uint4 (&kr)[LOADS]) {
#pragma unroll
    for (int i = 0; i < LOADS; ++i) {
      const int e = t + i * THREADS, g = tile * GROUPS + e / 16, d = (e % 16) * 8;
      kr[i] = make_uint4(0, 0, 0, 0);
      if (g < mv) {
        const int64_t page = table[seq + static_cast<int64_t>(g) * RATIO + RATIO - 1];
        kr[i] = *reinterpret_cast<const uint4*>(aux + page * stride_page + key_offset + d);
      }
    }
  };
  auto load_q = [&](int h, uint4 (&qr)[LOADS]) {
#pragma unroll
    for (int i = 0; i < LOADS; ++i) {
      const int e = t + i * THREADS, r = e / 16, d = (e % 16) * 8;
      qr[i] = r < rows ? *reinterpret_cast<const uint4*>(query + (row0 + r) * stride_qr + h * stride_qh + d)
                       : make_uint4(0, 0, 0, 0);
    }
  };
  auto put = [&](float* dst, const uint4 (&regs)[LOADS]) {
#pragma unroll
    for (int i = 0; i < LOADS; ++i) {
      const int e = t + i * THREADS;
      stage(dst + (e / 16) * PITCH + (e % 16) * 8, regs[i]);
    }
  };
  uint4 kr[LOADS], qr[LOADS];
  load_k(first, kr);
  load_q(0, qr);
#pragma unroll 1
  for (int tile = first; tile < live_end; ++tile) {
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
#pragma unroll 1
    for (int h = 0; h < HEADS; ++h) {
      __syncthreads();  // the previous head's operands are consumed
      if (h == 0) put(ks, kr);
      put(qs, qr);
      __syncthreads();
      if (h + 1 < HEADS) load_q(h + 1, qr);
      else if (tile + 1 < live_end) { load_k(tile + 1, kr); load_q(0, qr); }
      float u0[4][4], u1[4][4];
      octet(qs, ks, 0, ty, tx, u0);
      octet(qs, ks, 1, ty, tx, u1);
      add(u0, u1);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = __fadd_rn(acc[i][j], fmaxf(u0[i][j], 0.f));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = ty + 16 * i;
      if (r >= rows) continue;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int g = tile * GROUPS + tx + 16 * j;
        if (g < gcap) out[(row0 + r) * stride_or + g] = g < visible[r] ? __fmul_rn(acc[i][j], scale) : -CUDART_INF_F;
      }
    }
  }
}

void score(tvm::ffi::TensorView query, tvm::ffi::TensorView aux, tvm::ffi::TensorView table,
           tvm::ffi::TensorView base, tvm::ffi::TensorView positions, tvm::ffi::TensorView tiles,
           tvm::ffi::TensorView out, double scale, int64_t tiles_per_block) {
  const auto dev = query.device();
  tvm::ffi::CUDADeviceGuard guard(dev.device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  const int n_tiles = tiles.size(0), gcap = out.size(1);
  if (!n_tiles || !gcap) return;
  const int smem = (ROWS + GROUPS) * PITCH * sizeof(float);
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(score_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  const int group_tiles = (gcap + GROUPS - 1) / GROUPS;
  const int per_block = static_cast<int>(tiles_per_block);
  dim3 grid(n_tiles, (group_tiles + per_block - 1) / per_block);
  score_kernel<<<grid, THREADS, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(query.data_ptr()), query.stride(0), query.stride(1),
      static_cast<const __nv_bfloat16*>(aux.data_ptr()), aux.stride(0), aux.stride(1),
      static_cast<const int32_t*>(table.data_ptr()), static_cast<const int32_t*>(base.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()), static_cast<const int32_t*>(tiles.data_ptr()),
      static_cast<float*>(out.data_ptr()), out.stride(0), gcap, per_block, static_cast<float>(scale));
  auto error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

// ---- top-512 and page assembly for long score rows -------------------------------------------------------------
// The selection of topk_pages.cu (largest scores, equal scores by ascending group index, then sorted the same way,
// nonfinite scores excluded) for rows of any length, whose keys no longer fit in shared memory. Two passes over the
// row (L2): an 11-bit histogram finds the bin holding the 512th key; the second takes the keys above it and stages
// that bin's keys in shared memory, where the last 21 bits are resolved. A bin too full to stage is resolved over
// the row instead, and a tie at the threshold is cut by an ordered pass, as topk_pages.cu does.
constexpr int LT = 512, TOP = 512, WIDTH = 2051, BINS = 2048, CAP = 4096, UNROLL = 4;

__device__ __forceinline__ uint32_t score_key(float x) {
  if (!isfinite(x)) return 0;
  uint32_t bits = __float_as_uint(x == 0.f ? 0.f : x);
  return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

// Visit every key of the row (in index order per thread, UNROLL loads in flight).
template <typename F>
__device__ __forceinline__ void for_keys(const float* row, int cols, F f) {
  for (int first = threadIdx.x; first < cols; first += LT * UNROLL) {
    float v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int g = first + u * LT;
      v[u] = g < cols ? row[g] : -CUDART_INF_F;
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) f(first + u * LT, score_key(v[u]));
  }
}

__global__ __launch_bounds__(LT) void topk_long_kernel(
    const float* __restrict__ scores, int64_t stride, int cols, const int32_t* __restrict__ table,
    const int64_t* __restrict__ base, const int64_t* __restrict__ positions, int32_t* pages, int32_t* counts) {
  __shared__ uint32_t hist[BINS];
  __shared__ uint64_t winners[TOP];
  __shared__ uint32_t cand[CAP];  // staged keys of the threshold bin
  __shared__ int cand_index[CAP];
  __shared__ int valid_count, cursor, n_cand, greater, remaining, threshold_ties, ties_seen, warp_sum[LT / 32];
  __shared__ uint32_t prefix, mask, top_bin;
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  const int64_t row = blockIdx.x;
  const float* srow = scores + row * stride;
  for (int i = t; i < BINS; i += LT) hist[i] = 0;
  for (int i = t; i < TOP; i += LT) winners[i] = 0;
  for (int i = t; i < WIDTH; i += LT) pages[row * WIDTH + i] = -1;
  if (t == 0) { valid_count = 0; cursor = 0; n_cand = 0; ties_seen = 0; threshold_ties = 0; }
  __syncthreads();
  int mine = 0;
  for_keys(srow, cols, [&](int, uint32_t key) {
    if (key) { ++mine; atomicAdd(hist + (key >> 21), 1u); }
  });
  atomicAdd(&valid_count, mine);
  __syncthreads();
  const int valid = valid_count;
  auto take = [&](uint32_t key, int g) {  // order-free: the winners are sorted at the end
    winners[atomicAdd(&cursor, 1)] = (static_cast<uint64_t>(key) << 32) | (0xffffffffu - g);
  };
  if (valid <= TOP) {
    for_keys(srow, cols, [&](int g, uint32_t key) { if (key) take(key, g); });
  } else {
    // the bin of the 512th key: thread t owns bins 4t..4t+3; suffix sums over threads from the top
    uint32_t own = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) own += hist[4 * t + b];
    uint32_t suffix = own;  // within the warp, lanes >= lane
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      const uint32_t v = __shfl_down_sync(0xffffffff, suffix, d);
      if (lane + d < 32) suffix += v;
    }
    if (lane == 0) warp_sum[warp] = suffix;
    __syncthreads();
    uint32_t above = suffix - own;
    for (int w = warp + 1; w < LT / 32; ++w) above += warp_sum[w];
    if (above < TOP && above + own >= TOP) {
      for (int b = 3; b >= 0; --b) {
        const uint32_t n = hist[4 * t + b];
        if (above + n >= TOP) { top_bin = 4 * t + b; greater = above; break; }
        above += n;
      }
    }
    __syncthreads();
    const uint32_t bin = top_bin;
    const bool staged = hist[bin] <= CAP;
    for_keys(srow, cols, [&](int g, uint32_t key) {
      if (!key) return;
      if ((key >> 21) > bin) take(key, g);
      else if (staged && (key >> 21) == bin) { const int i = atomicAdd(&n_cand, 1); cand[i] = key; cand_index[i] = g; }
    });
    if (t == 0) { remaining = TOP - greater; prefix = bin << 21; mask = 0xffe00000u; }
    __syncthreads();
    // the threshold key: 8, 8 and 5 more bits over the staged keys (or the row)
    const int n = n_cand;
    for (int shift = 13; shift >= 0; shift = shift == 13 ? 5 : shift == 5 ? 0 : -1) {
      for (int i = t; i < 256; i += LT) hist[i] = 0;
      __syncthreads();
      const uint32_t p = prefix, m = mask;
      if (staged) {
        for (int i = t; i < n; i += LT)
          if ((cand[i] & m) == p) atomicAdd(hist + ((cand[i] >> shift) & 255), 1u);
      } else {
        for_keys(srow, cols, [&](int, uint32_t key) {
          if (key && (key & m) == p) atomicAdd(hist + ((key >> shift) & 255), 1u);
        });
      }
      __syncthreads();
      if (t == 0) {
        int left = remaining;
        for (int b = 255; b >= 0; --b) {
          if (left > static_cast<int>(hist[b])) left -= hist[b];
          else {
            prefix |= static_cast<uint32_t>(b) << shift;
            mask |= 255u << shift;
            remaining = left;
            if (shift == 0) threshold_ties = hist[b];
            break;
          }
        }
      }
      __syncthreads();
    }
    const uint32_t threshold = prefix;
    const bool cut = threshold_ties > remaining;  // only some of the threshold's ties fit
    if (staged) {
      for (int i = t; i < n; i += LT)
        if (cand[i] > threshold || (!cut && cand[i] == threshold)) take(cand[i], cand_index[i]);
    } else {
      for_keys(srow, cols, [&](int g, uint32_t key) {
        if ((key >> 21) == bin && (key > threshold || (!cut && key == threshold))) take(key, g);
      });
    }
    if (cut) {
      // the first `remaining` ties in group order
      const int need = remaining;
      for (int first = 0; first < cols; first += LT) {
        const int g = first + t;
        const bool equal = g < cols && score_key(srow[g]) == threshold;
        const uint32_t ballot = __ballot_sync(0xffffffff, equal);
        if (lane == 0) warp_sum[warp] = __popc(ballot);
        __syncthreads();
        if (t == 0) {
          int sum = ties_seen;
          for (int w = 0; w < LT / 32; ++w) { const int k = warp_sum[w]; warp_sum[w] = sum; sum += k; }
          ties_seen = sum;
        }
        __syncthreads();
        const int rank = warp_sum[warp] + __popc(ballot & ((1u << lane) - 1));
        if (equal && rank < need) winners[TOP - need + rank] = (static_cast<uint64_t>(threshold) << 32) | (0xffffffffu - g);
        __syncthreads();
        if (ties_seen >= need) break;
      }
    }
  }
  __syncthreads();
  const int target = min(valid, TOP);
  // descending score, ascending group index; empty slots (0) sort last
  for (int size = 2; size <= TOP; size *= 2) {
    for (int distance = size / 2; distance; distance /= 2) {
      for (int i = t; i < TOP; i += LT) {
        const int j = i ^ distance;
        if (j > i) {
          const uint64_t a = winners[i], b = winners[j];
          const bool descending = (i & size) == 0;
          if (descending ? a < b : a > b) { winners[i] = b; winners[j] = a; }
        }
      }
      __syncthreads();
    }
  }
  const int64_t offset = base[row], position = positions[row];
  for (int i = t; i < target * 4; i += LT) {
    const uint32_t group = 0xffffffffu - static_cast<uint32_t>(winners[i / 4]);
    pages[row * WIDTH + i] = table[offset + static_cast<int64_t>(group) * 4 + i % 4];
  }
  const int tail = (position + 1) % 4;
  if (t < tail) pages[row * WIDTH + target * 4 + t] = table[offset + ((position + 1) / 4) * 4 + t];
  if (t == 0) counts[row] = target * 4 + tail;
}

void topk_pages_long(tvm::ffi::TensorView scores, tvm::ffi::TensorView table, tvm::ffi::TensorView base,
                     tvm::ffi::TensorView positions, tvm::ffi::TensorView pages, tvm::ffi::TensorView counts) {
  const auto dev = scores.device();
  tvm::ffi::CUDADeviceGuard guard(dev.device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  if (!scores.size(0)) return;
  topk_long_kernel<<<scores.size(0), LT, 0, stream>>>(
      static_cast<const float*>(scores.data_ptr()), scores.stride(0), scores.size(1),
      static_cast<const int32_t*>(table.data_ptr()), static_cast<const int64_t*>(base.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()), static_cast<int32_t*>(pages.data_ptr()),
      static_cast<int32_t*>(counts.data_ptr()));
  auto error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}
}  // namespace qsa_exact
TVM_FFI_DLL_EXPORT_TYPED_FUNC(score, qsa_exact::score);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_pages_long, qsa_exact::topk_pages_long);
