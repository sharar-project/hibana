# Sharar runtime image: CUDA 13 devel base (nvcc is still needed for the
# FlashInfer SM121 JIT modules and the CUDA GEMV extension until Phase 9 builds
# them ahead of time) plus only the pinned Python packages.
ARG BASE_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev ninja-build ccache git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/sharar && /opt/sharar/bin/pip install --upgrade pip
ENV PATH=/opt/sharar/bin:/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# torch and its CUDA runtime wheels from the PyTorch cu130 index
COPY constraints /opt/constraints
RUN pip install -c /opt/constraints/gb10-cu130.txt torch --index-url https://download.pytorch.org/whl/cu130
# FlashInfer with its prebuilt cu130 JIT cache (SM121 modules still JIT)
RUN pip install -c /opt/constraints/gb10-cu130.txt flashinfer-python flashinfer-jit-cache \
      --extra-index-url https://flashinfer.ai/whl/cu130
RUN pip install -c /opt/constraints/gb10-cu130.txt safetensors tokenizers jinja2 numpy \
      einops packaging fastapi uvicorn pytest httpx
COPY docker/repair_wheel_tag.py /opt/repair_wheel_tag.py
RUN python3 /opt/repair_wheel_tag.py && pip check

COPY sharar /workspace/sharar/sharar
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /workspace/sharar/
COPY tests /workspace/sharar/tests
COPY profiling /workspace/sharar/profiling
COPY tools/check.sh /workspace/sharar/tools/check.sh
COPY constraints /workspace/sharar/constraints
WORKDIR /workspace/sharar
ENV PYTHONPATH=/workspace/sharar \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SHARAR_HF_CACHE=/cache/huggingface \
    HOME=/state
RUN mkdir -p /state && chmod 777 /state && python3 -c "import sharar, flashinfer, torch; print(torch.__version__, flashinfer.__version__)"

LABEL io.sharar.target="gb10-sm121" \
      io.sharar.engine="sharar" \
      io.sharar.reference-model="RadixArk/Qwen3.8-27B-NVFP4"
EXPOSE 30001
ENTRYPOINT ["python3", "-u", "-m", "sharar"]
CMD ["serve", "--host", "0.0.0.0", "--port", "30001"]
