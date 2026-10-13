# SPDX-License-Identifier: Apache-2.0
"""Pinned Qwen3.8-Flash-Next checkpoint and geometry (RadixArk's NVFP4 export of the preview)."""

MODEL_ID = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MODEL_CACHE_KEY = "models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
MODEL_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
# Calibrated 3-bit trunk copies (#6, #72): fetched on first serve when $SHARAR_RESULTS/w3-checkpoints/flashnext-current
# is missing. Private until the redistribution terms of the NVFP4 export are checked; the fetch then fails and the
# launcher serves the FP8 trunk with a warning.
W3_REPO = "yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3"
W3_REVISION = "694e2790772b36297b52bb1dffdc33281d766671"  # 3-bit trunk + GPTQ 3-bit routed experts (#73); the trunk-only copies were 970cbfc

HIDDEN_SIZE = 2560
NUM_LAYERS = 48
# Three Gated DeltaNet layers then one sparse-attention layer, twelve times.
LAYER_TYPES = tuple("full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(NUM_LAYERS))
NUM_ATTENTION_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROPE_THETA = 10_000_000.0
LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 48
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128
LINEAR_CONV_KERNEL = 4
VOCAB_SIZE = 248_320
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 262_144
NVFP4_GROUP_SIZE = 16

NUM_EXPERTS = 512
EXPERTS_PER_TOKEN = 10
MOE_INTERMEDIATE_SIZE = 640
SHARED_EXPERT_INTERMEDIATE_SIZE = 640
HC_COUNT = 4
HC_LOWRANK = 320
PLE_LAYER_IDS = (2,)  # one-based as in the config: the PLE sits before decoder layer index 1
PLE_NGRAM = 3
PLE_HEADS_PER_NGRAM = 8
PLE_EMBED_DIM = 2560
PLE_CONV_KERNEL = 4
INDEXER_BUDGET = 2048
INDEXER_COMPRESS_RATIO = 4
INDEXER_N_HEADS = 4
INDEXER_HEAD_DIM = 128
MTP_LAYERS = 1

# Resident weights: NVFP4 experts ~68 GB + FP8 trunk ~3.5 GB + BF16 embeddings 1.3 GB + FP8 head 0.6 GB,
# plus the staging of one layer's expert shards during the load.
WEIGHT_LOAD_PEAK_BYTES = 80 * 2**30
PREFILL_CHUNK = 4096  # prompt tokens a request prefills per step (target.py)
