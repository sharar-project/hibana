# Third-party notices

Sharar is an independent engine. It vendors a small number of kernels and
ports a small amount of model and weight-layout code. Every vendored or ported
file keeps its original copyright header, and this file lists the origins.
Verify the license text of each upstream at the exact commit before vendoring
a new file, and add a row here in the same commit.

| Origin | What Sharar takes from it | License |
|---|---|---|
| SGLang, <https://github.com/sgl-project/sglang>, snapshot `6d34a4d3ce1005593fdb89478651ded47c0d5277` | Qwen3.8 hybrid model math, ModelOpt NVFP4/FP8 weight layout handling, GDN backend structure, reasoning and tool-call parsers, GB10 benchmark harness | Apache License 2.0 |
| SGLang, <https://github.com/sgl-project/sglang>, snapshot `6b36f9d0a214890ad458e21ae4f0982229368b1e` | DSpark/DFlash2 model equations, YaRN, selector/Markov and block-local convolution semantics in `sharar/model/parallel_draft.py`; license checked at this commit | Apache License 2.0 |
| Z Lab DFlash, <https://github.com/z-lab/dflash>, snapshot `07ebd93db9f472af339b644bb70221ad8428328a` | DFlash2 reference convolution arithmetic order and selector semantics; independent test oracle, excluded from runtime dependencies | MIT, Copyright (c) 2026 Z Lab |
| flash-linear-attention, <https://github.com/fla-org/flash-linear-attention> (Songlin Yang, Yu Zhang) | Gated DeltaNet Triton kernels: chunked prefill, fused recurrent decode, gated RMSNorm, L2 norm | MIT License |
| causal-conv1d, <https://github.com/Dao-AILab/causal-conv1d> (Tri Dao) and vLLM, <https://github.com/vllm-project/vllm> | Triton causal Conv1D prefill and state-update kernels | BSD 3-Clause (causal-conv1d), Apache License 2.0 (vLLM) |
| FlashInfer, <https://github.com/flashinfer-ai/flashinfer> | Used as a dependency, not vendored: paged attention, NVFP4/FP8 GEMM, activation quantization, sampling | Apache License 2.0 |

Sharar's early numerical parity was checked against a reference SGLang runtime, which was never
distributed with Sharar; that tooling was removed on 2026-10-02.

## Z Lab DFlash MIT notice

Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
