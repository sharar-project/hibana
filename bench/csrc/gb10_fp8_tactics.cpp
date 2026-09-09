// Benchmark-only cuBLASLt heuristic enumeration on real target FP8 operands.
// Each tactic retains its own opaque algorithm descriptor for this process.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <vector>
#include <set>

static void check(cublasStatus_t status) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(status));
}

class Tactics {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> choices;
  int64_t m, n, k;
  size_t workspace_bytes;

 public:
  Tactics(int64_t m_, int64_t n_, int64_t k_, int requested, size_t workspace_)
      : m(m_), n(n_), k(k_), workspace_bytes(workspace_) {
    TORCH_CHECK(m == 16 && n > 0 && k > 0 && requested > 0 && requested <= 256);
    check(cublasLtCreate(&handle));
    check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    int8_t fast = 0;
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
    // Row-major q @ w.T is column-major w.T @ q in the same storage.
    check(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, k, n, k));
    check(cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, k, m, k));
    check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));
    cublasLtMatmulPreference_t pref = nullptr;
    check(cublasLtMatmulPreferenceCreate(&pref));
    check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_bytes, sizeof(workspace_bytes)));
    std::vector<cublasLtMatmulHeuristicResult_t> found(requested);
    int count = 0;
    auto status = cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, c, pref,
                                                requested, found.data(), &count);
    cublasLtMatmulPreferenceDestroy(pref);
    check(status);
    for (int i = 0; i < count; ++i)
      if (found[i].state == CUBLAS_STATUS_SUCCESS) choices.push_back(found[i]);
  }
  ~Tactics() {
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (desc) cublasLtMatmulDescDestroy(desc);
    if (handle) cublasLtDestroy(handle);
  }
  Tactics(const Tactics&) = delete;
  int expand() {
    // Vary the custom implementation option around every returned tile/stage.
    // AlgoCheck rejects unsupported combinations before any device work.
    auto seeds = choices;
    std::set<std::vector<int64_t>> seen;
    for (auto row : metadata()) seen.insert(row);
    for (const auto& seed : seeds) {
      uint32_t maximum = 0;
      size_t written = 0;
      check(cublasLtMatmulAlgoCapGetAttribute(&seed.algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
                                             &maximum, sizeof(maximum), &written));
      TORCH_CHECK(maximum <= 256, "unexpected custom-option range");
      for (uint32_t option = 0; option <= maximum; ++option) {
        auto trial = seed;
        if (cublasLtMatmulAlgoConfigSetAttribute(&trial.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                                 &option, sizeof(option)) != CUBLAS_STATUS_SUCCESS) continue;
        cublasLtMatmulHeuristicResult_t checked{};
        if (cublasLtMatmulAlgoCheck(handle, desc, a, b, c, c, &trial.algo, &checked) != CUBLAS_STATUS_SUCCESS
            || checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > workspace_bytes) continue;
        trial.workspaceSize = checked.workspaceSize;
        choices.push_back(trial);
        auto key = metadata().back();
        if (!seen.insert(key).second) choices.pop_back();
        TORCH_CHECK(choices.size() <= 1024, "bounded tactic screen exceeded");
      }
    }
    return int(choices.size());
  }
  std::vector<std::vector<int64_t>> metadata() const {
    std::vector<std::vector<int64_t>> result;
    for (const auto& h : choices) {
      std::vector<int64_t> row{int64_t(h.workspaceSize)};
      for (auto attribute : {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
                             CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                             CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                             CUBLASLT_ALGO_CONFIG_STAGES_ID}) {
        int32_t value = 0;
        size_t written = 0;
        check(cublasLtMatmulAlgoConfigGetAttribute(&h.algo, attribute, &value, sizeof(value), &written));
        row.push_back(value);
      }
      result.push_back(row);
    }
    return result;
  }
  torch::Tensor run(torch::Tensor q, torch::Tensor w, torch::Tensor qs,
                    torch::Tensor ws, int index, torch::Tensor workspace) {
    TORCH_CHECK(q.is_cuda() && w.device() == q.device() && qs.device() == q.device()
                && ws.device() == q.device() && workspace.device() == q.device());
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn && w.scalar_type() == at::kFloat8_e4m3fn);
    TORCH_CHECK(q.is_contiguous() && w.is_contiguous() && q.dim() == 2 && w.dim() == 2);
    TORCH_CHECK(q.size(0) == m && q.size(1) == k && w.size(0) == n && w.size(1) == k);
    TORCH_CHECK(qs.scalar_type() == at::kFloat && ws.scalar_type() == at::kFloat && qs.numel() == 1 && ws.numel() == 1);
    TORCH_CHECK(index >= 0 && index < int(choices.size()));
    TORCH_CHECK(workspace.scalar_type() == at::kByte && workspace.is_contiguous()
                && workspace.numel() >= int64_t(workspace_bytes));
    const void* w_scale = ws.data_ptr();
    const void* q_scale = qs.data_ptr();
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scale, sizeof(w_scale)));
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &q_scale, sizeof(q_scale)));
    auto out = torch::empty({m, n}, q.options().dtype(at::kBFloat16));
    const float alpha = 1, beta = 0;
    check(cublasLtMatmul(handle, desc, &alpha, w.data_ptr(), a, q.data_ptr(), b, &beta,
                         out.data_ptr(), c, out.data_ptr(), c, &choices[index].algo,
                         workspace.data_ptr(), workspace_bytes, c10::cuda::getCurrentCUDAStream()));
    return out;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<Tactics>(module, "Tactics")
      .def(pybind11::init<int64_t, int64_t, int64_t, int, size_t>())
      .def("metadata", &Tactics::metadata)
      .def("expand", &Tactics::expand)
      .def("run", &Tactics::run);
}
