"""Facts for the single validated Qwen checkpoint; consumed by specialized kernels."""

MODEL_ID = "RadixArk/Qwen3.8-27B-NVFP4"
MODEL_REVISION = "319f741cce68d7914884900c138a1fbb70a42f30"
MODEL_CACHE_KEY = "models--RadixArk--Qwen3.8-27B-NVFP4"
MODEL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"  # Qwen3.8 reuses the 3.5 code path
# Calibrated 3-bit decode copies (#72): fetched on first serve when $SHARAR_RESULTS/w3-checkpoints/qwen38-current is
# missing (GPTQ from the FP8 release, 2026-10-01; Apache-2.0).
W3_REPO = "yemara/Qwen3.8-27B-NVFP4-sharar-w3"
W3_REVISION = "72997f2aadf229bdc39603d802df61b367078a91"


# Architecture facts (from config.json text_config; asserted at load time).
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
NUM_LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
NUM_ATTENTION_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROPE_THETA = 10_000_000.0
LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 48
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128
LINEAR_CONV_KERNEL = 4
VOCAB_SIZE = 248320
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 262144
NVFP4_GROUP_SIZE = 16

# GB10 measured after releasing source mappings per layer: 21.36 GiB peak
# MemAvailable drop (20 ms samples), 19.58 GiB CUDA peak. 24 GiB retains a
# 2.64 GiB margin; the separate 16 GiB host floor still applies. See the Phase 7D study (archive branch).
WEIGHT_LOAD_PEAK_BYTES = 24 * 2**30
