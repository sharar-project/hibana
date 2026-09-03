# Third-party notices

Sharar is an independent engine. It vendors a small number of kernels and
ports a small amount of model and weight-layout code. Every vendored or ported
file keeps its original copyright header, and this file lists the origins.
Verify the license text of each upstream at the exact commit before vendoring
a new file, and add a row here in the same commit.

| Origin | What Sharar takes from it | License |
|---|---|---|
| SGLang, <https://github.com/sgl-project/sglang>, snapshot `6d34a4d3ce1005593fdb89478651ded47c0d5277` | Qwen3.8 hybrid model math, ModelOpt NVFP4/FP8 weight layout handling, GDN backend structure, reasoning and tool-call parsers, GB10 benchmark harness | Apache License 2.0 |
| flash-linear-attention, <https://github.com/fla-org/flash-linear-attention> (Songlin Yang, Yu Zhang) | Gated DeltaNet Triton kernels: chunked prefill, fused recurrent decode, gated RMSNorm, L2 norm | MIT License |
| causal-conv1d, <https://github.com/Dao-AILab/causal-conv1d> (Tri Dao) and vLLM, <https://github.com/vllm-project/vllm> | Triton causal Conv1D prefill and state-update kernels | BSD 3-Clause (causal-conv1d), Apache License 2.0 (vLLM) |
| FlashInfer, <https://github.com/flashinfer-ai/flashinfer> | Used as a dependency, not vendored: paged attention, NVFP4/FP8 GEMM, activation quantization, sampling | Apache License 2.0 |

The reference SGLang runtime used as Sharar's correctness oracle lives in the
separate `sparklang` repository and is not distributed with Sharar.
