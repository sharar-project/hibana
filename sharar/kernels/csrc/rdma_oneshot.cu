// Two-rank one-shot all-reduce over RDMA (kernels/rdma_oneshot.py). DGX Spark has no GPUDirect RDMA, so the NIC reads
// and writes cudaHostAlloc buffers registered with it (NVIDIA's guidance for the GB10). A kernel copies the partial into
// a send slot and rings a doorbell in that memory; a busy-polling CPU thread writes half the bytes through each RDMA
// device into the peer's receive slot, each half followed by an 8-byte flag on the same queue pair (written after the
// data it covers); the peer's kernel waits for both flags and adds in FP32, rounded once to BF16 as NCCL and torch do,
// so both ranks hold the same bits. Sequence numbers live in device memory, so the kernels replay inside CUDA graphs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

constexpr int NDEV = 2;     // the ConnectX-7's two PCIe halves, paired with the peer's in name order
constexpr int NSLOT = 4;    // the ranks are never more than one collective apart
constexpr int BLOCKS = 32;  // at most; all resident at once
// A peer this late is gone: trap rather than hang. NCCL's default in torch, since a rank can legitimately wait
// minutes for the other (a cold kernel cache compiling on one side at startup).
constexpr uint64_t TIMEOUT_NS = 600ull * 1000 * 1000 * 1000;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(std::string(msg) + ": " + std::strerror(errno)); } while (0)
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while (0)

struct Info {  // what the peer needs to write into this rank's buffers
  uint64_t addr;
  uint32_t rkey[NDEV];
  uint32_t qpn[NDEV];
  uint8_t gid[NDEV][16];
};

__device__ __forceinline__ uint64_t now_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint64_t load_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint64_t load_acquire_gpu(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void store_release_gpu(uint64_t* p, uint64_t v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

template <typename Load>
__device__ __forceinline__ void wait_for(const uint64_t* p, uint64_t seq, Load load) {
  const uint64_t start = now_ns();
  while (load(p) < seq)
    if (now_ns() - start > TIMEOUT_NS) __trap();
}

__global__ void send_kernel(const uint4* x, uint4* send, size_t slot_vec, size_t n, uint64_t* seq_dev, unsigned* arrive,
                            uint64_t* doorbell, uint64_t* sizes) {
  const uint64_t seq = *seq_dev + 1;  // every block reads it before the last one to finish publishes it
  const int slot = seq % NSLOT;
  uint4* dst = send + slot * slot_vec;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] = x[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(arrive, 1) == gridDim.x - 1) {
    *arrive = 0;
    *seq_dev = seq;
    sizes[slot] = n * 16;
    __threadfence_system();
    *(volatile uint64_t*)&doorbell[slot] = seq;
  }
}

// One poller on host memory: GPU threads spinning on the flags the NIC writes make the NIC win the lines back each
// time (a 160 KiB exchange waited 17 us with 32 pollers, 4 us with one). Block 0 polls; the others wait on device memory.
__global__ void recv_add_kernel(uint4* x, const uint4* recv, size_t slot_vec, size_t n, const uint64_t* seq_dev,
                                const uint64_t* flags, uint64_t* go) {
  const uint64_t seq = *seq_dev;
  const int slot = seq % NSLOT;
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) {
      for (int d = 0; d < NDEV; ++d) wait_for(&flags[slot * NDEV + d], seq, load_acquire_sys);
      store_release_gpu(go, seq);
    } else {
      wait_for(go, seq, load_acquire_gpu);
    }
  }
  __syncthreads();
  const uint4* src = recv + slot * slot_vec;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 a = x[i], b = __ldcv(&src[i]);
    __nv_bfloat162* pa = reinterpret_cast<__nv_bfloat162*>(&a);
    const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 fa = __bfloat1622float2(pa[j]), fb = __bfloat1622float2(pb[j]);
      pa[j] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
    }
    x[i] = a;
  }
}

class OneShot {
 public:
  OneShot(std::vector<std::string> devices, std::vector<int64_t> gid_index, int64_t max_bytes) : max_bytes_(max_bytes) {
    TORCH_CHECK(devices.size() == NDEV && gid_index.size() == NDEV, "two RDMA devices");
    region_bytes_ = 2 * NSLOT * max_bytes_ + NSLOT * NDEV * 8 + 2 * NSLOT * 8;
    CUDA_CHECK(cudaHostAlloc(&host_, region_bytes_, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(host_, 0, region_bytes_);
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_, host_, 0));
    CUDA_CHECK(cudaMalloc(&state_, 3 * 8));  // sequence, arrivals, go
    CUDA_CHECK(cudaMemset(state_, 0, 3 * 8));
    int n = 0;
    ibv_device** list = ibv_get_device_list(&n);
    for (int d = 0; d < NDEV; ++d) {
      gid_index_[d] = gid_index[d];
      ibv_device* found = nullptr;
      for (int i = 0; i < n; ++i)
        if (devices[d] == ibv_get_device_name(list[i])) found = list[i];
      CHECK(found, "no RDMA device " + devices[d]);
      ctx_[d] = ibv_open_device(found);
      CHECK(ctx_[d], "ibv_open_device");
      pd_[d] = ibv_alloc_pd(ctx_[d]);
      CHECK(pd_[d], "ibv_alloc_pd");
      mr_[d] = ibv_reg_mr(pd_[d], host_, region_bytes_, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      CHECK(mr_[d], "ibv_reg_mr");
      cq_[d] = ibv_create_cq(ctx_[d], 4096, nullptr, nullptr, 0);
      CHECK(cq_[d], "ibv_create_cq");
      ibv_qp_init_attr init{};
      init.send_cq = init.recv_cq = cq_[d];
      init.qp_type = IBV_QPT_RC;
      init.cap.max_send_wr = 4096;
      init.cap.max_recv_wr = 1;
      init.cap.max_send_sge = init.cap.max_recv_sge = 1;
      init.cap.max_inline_data = 64;
      qp_[d] = ibv_create_qp(pd_[d], &init);
      CHECK(qp_[d], "ibv_create_qp");
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_INIT;
      a.port_num = 1;
      a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
      CHECK(!ibv_modify_qp(qp_[d], &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "QP to INIT");
      CHECK(!ibv_query_gid(ctx_[d], 1, gid_index_[d], &gid_[d]), "ibv_query_gid");
    }
    ibv_free_device_list(list);
  }

  ~OneShot() {
    running_ = false;
    if (proxy_.joinable()) proxy_.join();
    for (int d = 0; d < NDEV; ++d) {
      ibv_destroy_qp(qp_[d]);
      ibv_destroy_cq(cq_[d]);
      ibv_dereg_mr(mr_[d]);
      ibv_dealloc_pd(pd_[d]);
      ibv_close_device(ctx_[d]);
    }
    cudaFree(state_);
    cudaFreeHost(host_);
  }

  py::bytes info() const {
    Info i{};
    i.addr = reinterpret_cast<uint64_t>(host_);
    for (int d = 0; d < NDEV; ++d) {
      i.rkey[d] = mr_[d]->rkey;
      i.qpn[d] = qp_[d]->qp_num;
      std::memcpy(i.gid[d], gid_[d].raw, 16);
    }
    return py::bytes(reinterpret_cast<const char*>(&i), sizeof(i));
  }

  void connect(py::bytes peer) {
    std::string s = peer;
    TORCH_CHECK(s.size() == sizeof(Info), "peer info");
    std::memcpy(&peer_, s.data(), sizeof(Info));
    for (int d = 0; d < NDEV; ++d) {
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_RTR;
      a.path_mtu = IBV_MTU_4096;
      a.dest_qp_num = peer_.qpn[d];
      a.max_dest_rd_atomic = 1;
      a.min_rnr_timer = 12;
      a.ah_attr.is_global = 1;
      std::memcpy(a.ah_attr.grh.dgid.raw, peer_.gid[d], 16);
      a.ah_attr.grh.sgid_index = gid_index_[d];
      a.ah_attr.grh.hop_limit = 64;
      a.ah_attr.port_num = 1;
      CHECK(!ibv_modify_qp(qp_[d], &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                             IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), "QP to RTR");
      a = {};
      a.qp_state = IBV_QPS_RTS;
      a.timeout = 14;
      a.retry_cnt = 7;
      a.rnr_retry = 7;
      a.max_rd_atomic = 1;
      CHECK(!ibv_modify_qp(qp_[d], &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                             IBV_QP_MAX_QP_RD_ATOMIC), "QP to RTS");
    }
    running_ = true;
    proxy_ = std::thread([this] { proxy(); });
  }

  int64_t max_bytes() const { return max_bytes_; }

  void all_reduce(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "a contiguous BF16 CUDA tensor");
    const size_t bytes = x.numel() * 2;
    TORCH_CHECK(bytes % (16 * NDEV) == 0 && (int64_t)bytes <= max_bytes_, "size");
    const size_t n = bytes / 16, slot_vec = max_bytes_ / 16;
    const int threads = 256, blocks = (int)std::min<size_t>(BLOCKS, (n + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    char* base = static_cast<char*>(dev_);
    send_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(x.data_ptr()), reinterpret_cast<uint4*>(base),
                                                slot_vec, n, state_, reinterpret_cast<unsigned*>(state_ + 1),
                                                doorbell(base), sizes(base));
    recv_add_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<uint4*>(x.data_ptr()),
                                                    reinterpret_cast<const uint4*>(base + NSLOT * max_bytes_), slot_vec, n,
                                                    state_, flags(base), state_ + 2);
  }

 private:
  uint64_t* flags(char* base) const { return reinterpret_cast<uint64_t*>(base + 2 * NSLOT * max_bytes_); }
  uint64_t* doorbell(char* base) const { return flags(base) + NSLOT * NDEV; }
  uint64_t* sizes(char* base) const { return doorbell(base) + NSLOT; }

  void proxy() {
    char* base = static_cast<char*>(host_);
    volatile uint64_t* bell = doorbell(base);
    volatile uint64_t* size = sizes(base);
    uint64_t seq = 1, posted = 0;
    ibv_wc wc[16];
    while (running_.load(std::memory_order_relaxed)) {
      const int slot = seq % NSLOT;
      if (bell[slot] != seq) {  // idle: reap the signaled completions that free send-queue entries
        for (int d = 0; d < NDEV; ++d) ibv_poll_cq(cq_[d], 16, wc);
        continue;
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint64_t part = size[slot] / NDEV;
      for (int d = 0; d < NDEV; ++d) {
        ibv_sge data{reinterpret_cast<uint64_t>(base + slot * max_bytes_ + d * part), (uint32_t)part, mr_[d]->lkey};
        uint64_t value = seq;
        ibv_sge flag{reinterpret_cast<uint64_t>(&value), 8, 0};
        ibv_send_wr wr[2]{}, *bad = nullptr;
        wr[0].opcode = wr[1].opcode = IBV_WR_RDMA_WRITE;
        wr[0].sg_list = &data;
        wr[1].sg_list = &flag;
        wr[0].num_sge = wr[1].num_sge = 1;
        wr[0].wr.rdma.remote_addr = peer_.addr + (NSLOT + slot) * max_bytes_ + d * part;
        wr[1].wr.rdma.remote_addr = peer_.addr + 2 * NSLOT * max_bytes_ + (slot * NDEV + d) * 8;
        wr[0].wr.rdma.rkey = wr[1].wr.rdma.rkey = peer_.rkey[d];
        wr[0].next = &wr[1];
        wr[1].send_flags = IBV_SEND_INLINE | (posted % 64 == 0 ? IBV_SEND_SIGNALED : 0);
        if (ibv_post_send(qp_[d], wr, &bad)) throw std::runtime_error("ibv_post_send");
      }
      ++posted;
      ++seq;
    }
  }

  int64_t max_bytes_;
  size_t region_bytes_;
  void* host_ = nullptr;
  void* dev_ = nullptr;
  uint64_t* state_ = nullptr;
  int gid_index_[NDEV]{};
  ibv_context* ctx_[NDEV]{};
  ibv_pd* pd_[NDEV]{};
  ibv_mr* mr_[NDEV]{};
  ibv_cq* cq_[NDEV]{};
  ibv_qp* qp_[NDEV]{};
  ibv_gid gid_[NDEV]{};
  Info peer_{};
  std::atomic<bool> running_{false};
  std::thread proxy_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<OneShot>(m, "OneShot")
      .def(py::init<std::vector<std::string>, std::vector<int64_t>, int64_t>())
      .def("info", &OneShot::info)
      .def("connect", &OneShot::connect)
      .def("max_bytes", &OneShot::max_bytes)
      .def("all_reduce", &OneShot::all_reduce);
}
