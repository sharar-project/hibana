// Token rows between KV pool pages and the SSD tier's linear payloads, for every pool the tier stores, in one launch.
//
// A payload holds each pool's rows of one object back to back (cache/ssd.py: an 8 MiB block of the 27B is 32 such
// segments). Two int64 tables describe a launch:
//   pools [P, 2]:      each pool's address and row bytes (fixed for a tier, in device memory)
//   objs  [n, 2 + P]:  per object its rows, the index of its first page in `pages`, then per pool the address of
//                      its first row on the linear side (registered host memory the GPU reads in place)
// Block (x, y) copies rows of object x / blocks_per_obj for pool y, one warp per row in 16-byte (or narrower)
// words. The kernel touches only the pages `pages` lists.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kWarps = 8;

template <typename Word, typename Page, bool kToPool>
__global__ void __launch_bounds__(kWarps * 32) kv_copy_kernel(const int64_t* __restrict__ pools,
                                                              const int64_t* __restrict__ objs,
                                                              const Page* __restrict__ pages, int width,
                                                              int blocks_per_obj) {
  const int obj = blockIdx.x / blocks_per_obj;
  const int part = blockIdx.x - obj * blocks_per_obj;
  const int pool = blockIdx.y;
  const int64_t* o = objs + static_cast<int64_t>(obj) * width;
  const int64_t rows = o[0];
  const Page* page = pages + o[1];
  char* lin = reinterpret_cast<char*>(o[2 + pool]);
  char* base = reinterpret_cast<char*>(pools[2 * pool]);
  const int64_t row_bytes = pools[2 * pool + 1];
  const int words = static_cast<int>(row_bytes / static_cast<int64_t>(sizeof(Word)));
  const int lane = threadIdx.x % 32;
  const int64_t stride = static_cast<int64_t>(blocks_per_obj) * kWarps;
  for (int64_t r = static_cast<int64_t>(part) * kWarps + threadIdx.x / 32; r < rows; r += stride) {
    Word* p = reinterpret_cast<Word*>(base + static_cast<int64_t>(page[r]) * row_bytes);
    Word* l = reinterpret_cast<Word*>(lin + r * row_bytes);
    for (int w = lane; w < words; w += 32) {
      if constexpr (kToPool) {
        p[w] = l[w];
      } else {
        l[w] = p[w];
      }
    }
  }
}

template <typename Word, typename Page>
void launch(bool to_pool, dim3 grid, cudaStream_t stream, const int64_t* pools, const int64_t* objs, const void* pages,
            int width, int blocks_per_obj) {
  auto* pg = static_cast<const Page*>(pages);
  if (to_pool) {
    kv_copy_kernel<Word, Page, true><<<grid, kWarps * 32, 0, stream>>>(pools, objs, pg, width, blocks_per_obj);
  } else {
    kv_copy_kernel<Word, Page, false><<<grid, kWarps * 32, 0, stream>>>(pools, objs, pg, width, blocks_per_obj);
  }
}

template <typename Word>
void launch_word(bool pages64, bool to_pool, dim3 grid, cudaStream_t stream, const int64_t* pools, const int64_t* objs,
                 const void* pages, int width, int blocks_per_obj) {
  if (pages64) {
    launch<Word, int64_t>(to_pool, grid, stream, pools, objs, pages, width, blocks_per_obj);
  } else {
    launch<Word, int32_t>(to_pool, grid, stream, pools, objs, pages, width, blocks_per_obj);
  }
}

}  // namespace

// Called with the GIL held: the launch takes microseconds, and the engine thread, had it let the GIL go, would wait up
// to a switch interval to get it back from the tier's I/O threads or the API.
void kv_copy(int64_t pools, int64_t objs, int64_t pages, bool pages64, int64_t n_pools, int64_t n_objs,
             int64_t max_rows, int64_t word, bool to_pool, int64_t stream) {
  if (n_objs == 0 || max_rows == 0) {
    return;
  }
  TORCH_CHECK(n_pools > 0 && n_pools <= 65535, "kv_copy: pools out of range");
  const int width = static_cast<int>(2 + n_pools);
  const int blocks_per_obj = static_cast<int>((max_rows + 8 * kWarps - 1) / (8 * kWarps));  // about 8 rows a warp
  TORCH_CHECK(n_objs * blocks_per_obj < (int64_t{1} << 31), "kv_copy: too many objects");
  const dim3 grid(static_cast<unsigned>(n_objs * blocks_per_obj), static_cast<unsigned>(n_pools));
  auto s = reinterpret_cast<cudaStream_t>(stream);
  auto* pt = reinterpret_cast<const int64_t*>(pools);
  auto* ot = reinterpret_cast<const int64_t*>(objs);
  auto* pg = reinterpret_cast<const void*>(pages);
  switch (word) {
    case 16: launch_word<uint4>(pages64, to_pool, grid, s, pt, ot, pg, width, blocks_per_obj); break;
    case 8: launch_word<uint2>(pages64, to_pool, grid, s, pt, ot, pg, width, blocks_per_obj); break;
    case 4: launch_word<uint32_t>(pages64, to_pool, grid, s, pt, ot, pg, width, blocks_per_obj); break;
    case 2: launch_word<uint16_t>(pages64, to_pool, grid, s, pt, ot, pg, width, blocks_per_obj); break;
    case 1: launch_word<uint8_t>(pages64, to_pool, grid, s, pt, ot, pg, width, blocks_per_obj); break;
    default: TORCH_CHECK(false, "kv_copy: word must be 1, 2, 4, 8 or 16 bytes");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv_copy", &kv_copy, "token rows between KV pool pages and linear payloads, every pool in one launch");
}
