<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hibana-logo-dark.svg">
    <img alt="Hibana" src="docs/assets/hibana-logo.svg" width="360">
  </picture>
</p>

<p align="center"><b>Tailored to the model. Tuned to the machine. Built for agents.</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.10469">Paper</a> ·
  <a href="https://sharar-project.github.io/blog/sharar-paper/">Blog post</a>
</p>

Hibana (formerly Sharar) runs open models on your own NVIDIA GB10 (DGX Spark) and serves them through OpenAI- and
Anthropic-compatible APIs, so Claude Code, opencode, aider or any OpenAI client can use them. It brings speculative
decoding, prefix caching (with an optional SSD tier for agent sessions) and a 262K context.

## Quick start

You need one DGX Spark (or another NVIDIA GB10 machine) with DGX OS and about 135 GB of free disk for this model.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # install uv, if you don't have it
git clone https://github.com/sharar-project/hibana && cd hibana
uv sync                                             # set up Python (about a minute)
uv run hibana serve nemotron                        # serve Nemotron 3.5 Super on http://127.0.0.1:30005
```

The first launch downloads the model and builds the GPU kernels; later launches start in under a minute. Ctrl-C
stops the server. In a second terminal:

```bash
uv run hibana chat              # chat in the terminal
uv run hibana launch claude     # Claude Code on the local model (also opencode, aider)
```

Any OpenAI client works against `http://127.0.0.1:30005/v1` and Anthropic clients against `http://127.0.0.1:30005`
([API](docs/api.md)).

> [!NOTE]
> Hibana was called Sharar. The `sharar` command, `SHARAR_*` settings and the former directories keep working
> ([upgrading](docs/operations.md#upgrading-from-sharar)).

## Models

Pick a model and run its command. Hibana downloads the publisher's checkpoint from Hugging Face at a pinned revision,
and for the Qwen models and Super a calibrated 3-bit copy that the default `fast` profile decodes with.

<details>
<summary><h3><img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.102.0/icons/qwen-color.svg" width="24" height="24" alt=""> Qwen3.8-Flash-Next</h3></summary>

Qwen's 125B mixture-of-experts model, on one Spark or two.

```bash
uv run hibana serve flashnext           # http://127.0.0.1:30004
uv run hibana serve flashnext --tp 2    # on two Sparks linked by their ConnectX ports
```

- **Download:** 186 GB, the model (135 GB) and its 3-bit copy (51 GB)
- **Input:** text
- **Draft:** its own MTP head, or the DFlash2 drafter on one Spark (`HIBANA_PROPOSER=dflash2`)
- **Checkpoints:** [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4),
  [3-bit copy](https://huggingface.co/yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad)
- **More:** [Flash-Next](docs/flashnext.md), [tensor parallel](docs/tensor-parallel.md)

</details>

<details>
<summary><h3><img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.102.0/icons/qwen-color.svg" width="24" height="24" alt=""> Qwen3.8-27B</h3></summary>

Qwen's 27B dense model, on one Spark or two.

```bash
uv run hibana serve 27b                 # http://127.0.0.1:30001
uv run hibana serve 27b --tp 2          # on two Sparks linked by their ConnectX ports
```

- **Download:** 36 GB, the model (26 GB) and its 3-bit copy (10 GB)
- **Input:** text
- **Draft:** the DFlash2 drafter ([incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2))
- **Checkpoints:** [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4),
  [3-bit copy](https://huggingface.co/yemara/Qwen3.8-27B-NVFP4-sharar-w3)
- **More:** [operations](docs/operations.md), [tensor parallel](docs/tensor-parallel.md)

</details>

<details>
<summary><h3><img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.102.0/icons/nvidia-color.svg" width="24" height="24" alt=""> Nemotron 3.5 Super</h3></summary>

NVIDIA's 120B mixture-of-experts model (12B active), a Mamba-2 hybrid that also reads images, on one Spark.

```bash
uv run hibana serve nemotron            # http://127.0.0.1:30005
```

- **Download:** 124 GB, the model (75 GB) and its 3-bit copy (49 GB)
- **Input:** text and images ([images](docs/api.md#images))
- **Draft:** its own MTP head
- **Checkpoints:** [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4),
  [3-bit copy](https://huggingface.co/shadowpa0327/NVIDIA-Nemotron-3.5-Super-NVFP4-sharar-w3)
- **More:** [Nemotron](docs/nemotron.md)

</details>

<details>
<summary><h3><img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.102.0/icons/nvidia-color.svg" width="24" height="24" alt=""> Nemotron 3.5 Bolt</h3></summary>

NVIDIA's 74B mixture-of-experts model (9B active), packed to 24 GB, that also reads images, on one Spark.

```bash
uv run hibana serve bolt                # http://127.0.0.1:30006
```

- **Download:** 25 GB
- **Input:** text and images ([images](docs/api.md#images))
- **Draft:** NVIDIA's companion LiLiCorr checkpoint, seven tokens at a time
- **Checkpoints:** [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB),
  [LiLiCorr draft](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr)
- **More:** [Nemotron Bolt](docs/bolt.md)

</details>

Not sure which to pick? Side by side:

| Model | Size | Download | Sparks | Images | Command |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 125B MoE | 186 GB | 1 or 2 | | `hibana serve flashnext` |
| Qwen3.8-27B | 27B dense | 36 GB | 1 or 2 | | `hibana serve 27b` |
| Nemotron 3.5 Super | 120B MoE, 12B active | 124 GB | 1 | yes | `hibana serve nemotron` |
| Nemotron 3.5 Bolt | 74B MoE, 9B active | 25 GB | 1 | yes | `hibana serve bolt` |

## Requirements

- One NVIDIA GB10 (SM 12.1, a DGX Spark; two linked by their ConnectX ports for
  [tensor parallel](docs/tensor-parallel.md)), `aarch64`, about 128 GB unified memory.
- NVIDIA driver 580 or newer and the CUDA 13 toolkit in `/usr/local/cuda` (DGX OS ships it; the GB10 kernels compile
  on first use, about 2.5 minutes once).
- [uv](https://docs.astral.sh/uv/), which installs the tested Python 3.12.
- Disk for the model (above), plus 10 GiB that the download check keeps spare
  ([operations](docs/operations.md#prerequisites)).

## Commands

| | |
|---|---|
| `hibana serve <model>` | Serve a model in the foreground (`27b`, `flashnext`, `nemotron`, `bolt`) |
| `hibana serve <model> --detach` | The same as a background service; `hibana stop`, `hibana logs -f` |
| `hibana serve <model> --exact` | The model's own outputs, for benchmarks (the default `fast` profile is faster and slightly lossy; also `--throughput`) |
| `hibana serve <model> --tp 2` | The Qwen models on two Sparks linked by their ConnectX ports ([tensor parallel](docs/tensor-parallel.md)) |
| `hibana status` | What is serving, with which settings |
| `hibana cache` | What is cached on the GPU and on the SSD tier (`$HIBANA_RESULTS/ssd/<target>`, up to `HIBANA_SSD_GIB`) |
| `hibana chat`, `hibana launch claude\|opencode\|aider\|show` | Use the serving model |

`hibana <command> --help` lists the options. Settings (cache and results locations, profiles, ports) are flags, then
`HIBANA_*` environment variables, then an optional `config/gb10.env` ([example](config/gb10.env.example),
[operations](docs/operations.md)).

For more details:

- [Operations](docs/operations.md): settings, serving profiles and the SSD tier
- [API](docs/api.md): endpoints, tool calls, images and constrained output
- [Tensor parallel](docs/tensor-parallel.md): serving on two Sparks
- Models: [Flash-Next](docs/flashnext.md), [Nemotron Super](docs/nemotron.md), [Nemotron Bolt](docs/bolt.md)
- [Evaluation](docs/evaluation.md): measuring quality and speed
- [All documentation](docs/README.md) and [contributing](CONTRIBUTING.md)

## Acknowledgment

Hibana learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[Mamba](https://github.com/state-spaces/mamba), [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d),
[llama.cpp](https://github.com/ggml-org/llama.cpp), [b12x](https://pypi.org/project/b12x/) and
[DFlash](https://github.com/z-lab/dflash). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists what came from where.

## License

[Apache License 2.0](LICENSE).
