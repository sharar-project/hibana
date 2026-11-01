# Third-party notices

Sharar is an independent engine. It vendors a small number of kernels and
ports a small amount of model and weight-layout code. Every vendored or ported
file keeps its original copyright header, and this file lists the origins.
Verify the license text of each upstream at the exact commit before vendoring
a new file, and add a row here in the same commit.

| Origin | What Sharar takes from it | License |
|---|---|---|
| SGLang, <https://github.com/sgl-project/sglang>, snapshot `6d34a4d3ce1005593fdb89478651ded47c0d5277` | Qwen3.8 hybrid model math, ModelOpt NVFP4/FP8 weight layout handling, GDN backend structure, reasoning and tool-call parsers | Apache License 2.0 |
| SGLang, <https://github.com/sgl-project/sglang>, snapshot `6b36f9d0a214890ad458e21ae4f0982229368b1e` | DFlash2 model equations, YaRN, selector and block-local convolution semantics in `sharar/model/parallel_draft.py`; license checked at this commit | Apache License 2.0 |
| Z Lab DFlash, <https://github.com/z-lab/dflash>, snapshot `07ebd93db9f472af339b644bb70221ad8428328a` | DFlash2 reference convolution arithmetic order and selector semantics | MIT, Copyright (c) 2026 Z Lab |
| flash-linear-attention, <https://github.com/fla-org/flash-linear-attention> (Songlin Yang, Yu Zhang) | Gated DeltaNet Triton kernels in `sharar/kernels/fla/`: chunked prefill, fused recurrent decode, gated RMSNorm, L2 norm | MIT, Copyright (c) 2023-2025, Songlin Yang, Yu Zhang |
| SGLang, snapshot `d62b9c2480f3b4b2ff16344c90e6a73e4b1d09fa` | QSA query prep and key-compression adaptations in `sharar/kernels/qsa/csrc/`; bundled `LICENSE.sglang` | Apache License 2.0 |
| SGLang v0.5.20, `srt/layers/attention/qsa/sparse_attn.py` | The QSA sparse attention kernel `_sparse_gqa_chunk_prefill` in `sharar/kernels/qsa/attention.py`, unchanged except for its wrapper | Apache License 2.0 |
| SGLang, `csrc/gemm/sm120_fp8_gemv.cuh` | The small-M FP8 GEMV in `sharar/kernels/csrc/fp8_gemv.cu`, extended from one activation row to four (attribution in its header) | Apache License 2.0 |
| vLLM, snapshot `f26d9f432ced7b1087d02d29bffed2249df1a18e` | QSA tiled scorer adapted in `sharar/kernels/qsa/tiled_scaled.py`, with fused fill/mask/scale; bundled `sharar/kernels/qsa/LICENSE.vllm` | Apache License 2.0 |
| vLLM, snapshot `73a78e6f1f38e280986b81e0f2a9aa5e1ee6fe47` | LiLiCorr lattice equations, grouped-convolution arithmetic, context-anchor and attention-sink semantics in `sharar/model/lilicorr.py`; license verified at this commit | Apache License 2.0 |
| causal-conv1d, <https://github.com/Dao-AILab/causal-conv1d> (Tri Dao) and vLLM, <https://github.com/vllm-project/vllm> | Triton causal Conv1D prefill and state-update kernels in `sharar/kernels/conv1d.py` | BSD 3-Clause, Copyright (c) 2024, Tri Dao (causal-conv1d); Apache License 2.0 (vLLM) |
| vLLM, <https://github.com/vllm-project/vllm>, `model_executor/layers/mamba/ops`, adapted from mamba_ssm v2.2.4, <https://github.com/state-spaces/mamba> (Tri Dao, Albert Gu) | Mamba-2 chunked-scan kernels (chunk state, state passing, BMM, chunk scan) in `sharar/kernels/mamba2.py` | Apache License 2.0 (vLLM and mamba_ssm) |
| FlashInfer, <https://github.com/flashinfer-ai/flashinfer> | A dependency (paged attention, NVFP4/FP8 GEMM, activation quantization, sampling); `sharar/kernels/cute/residual_norm.py` vendors the body of 0.6.17's CuTe `FusedAddRMSNormKernel` with quantizing epilogues added; `sharar/kernels/cute/gdn/` derives a 32-token-chunk GDN prefill kernel from 0.6.17's `gdn_kernels/delta_rule_dsl` (its `LICENSE` bundled) | Apache License 2.0 |
| b12x 1.5.0 (PyPI `b12x`) | A dependency on GB10 (the `gb10` extra), nothing vendored: Bolt's SM120 block-scaled GEMMs (`sharar/model/bolt/linear.py`) and fused IQ2_XXS MoE (`sharar/kernels/b12x_iq2_moe.py`) | Apache License 2.0 |
| ggml-org/llama.cpp, <https://github.com/ggml-org/llama.cpp>, snapshot `d327bf1f0780f87b8c01b367b61853b0c291d7c4` | IQ2_XS/XXS codebooks in `weights/iq2_xs_grid.py` and `weights/iq2_grid.py`, Q8_0/IQ2 layout decoding in `weights/ggml.py` and `kernels/csrc/ggml_codec.cuh` (used by the routed-expert kernel); license verified at this commit | MIT, Copyright (c) 2023-2026 The ggml authors |

## ggml MIT notice

Copyright (c) 2023-2026 The ggml authors

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


## flash-linear-attention MIT notice

MIT License

Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

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


## causal-conv1d BSD 3-Clause notice

The adapted kernels' header: Copyright (c) 2024, Tri Dao. The upstream license, whose AUTHORS file lists Tri Dao:

BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
