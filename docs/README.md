# Hibana documentation

Each page describes the code as it is; measurements and plans live in issues and pull requests. When a change alters
behavior, update its page in the same pull request (`python3 tools/check_links.py` after moving or deleting one).

| Page | Contents |
|---|---|
| [Operations](operations.md) | Prerequisites, launcher settings, serving profiles, traces, the SSD tier |
| [API](api.md) | Endpoints, request controls, tool calls, constrained output, Anthropic Messages |
| [Architecture](architecture.md) | Package layout, the seams the targets share, capability matrix |
| [Cache design](cache_design.md) | GPU pools, the radix prefix cache and the SSD tier |
| [Memory plan](memory_plan.md) | How startup sizes the KV and state pools around the host floor |
| [Interfaces](interfaces.md) | Request ownership, proposer lifecycle, device-resident acceptance, failure handling |
| [Flash-Next](flashnext.md) | Qwen3.8-Flash-Next: model, package, conventions, sparse attention, its MTP and DFlash2 drafts |
| [Nemotron](nemotron.md) | Nemotron 3.5 Super: the Mamba-2 and latent-MoE model as served, formats, its MTP draft |
| [Nemotron Bolt](bolt.md) | Packed IQ2/Q8/NVFP4 text inference on the Nemotron Mamba runtime |
| [Tensor parallel](tensor-parallel.md) | Two Sparks: setup, sharding, the RDMA all-reduce, limits |
| [3-bit calibration](w3-calibration.md) | Fitting and serving the calibrated 3-bit copies |
| [Evaluation](evaluation.md) | Measuring a build or a change, the lossy-change gate, nightly and weekly GB10 checks |
| [Profiling](../profiling/README.md) | In-process profilers |
