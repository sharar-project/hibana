"""Facts for the single validated Qwen checkpoint; consumed by specialized kernels."""

MODEL_ID = "RadixArk/Qwen3.8-27B-NVFP4"
MODEL_REVISION = "319f741cce68d7914884900c138a1fbb70a42f30"
MODEL_CACHE_KEY = "models--RadixArk--Qwen3.8-27B-NVFP4"
MODEL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"  # Qwen3.8 reuses the 3.5 code path
# Calibrated 3-bit decode copies: fetched on first serve when $HIBANA_RESULTS/w3-checkpoints/qwen38-current is
# missing (GPTQ from the FP8 release; Apache-2.0).
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
# 2.64 GiB margin; the separate host floor still applies.
WEIGHT_LOAD_PEAK_BYTES = 24 * 2**30

# The pinned parallel draft (repository, revision): DFlash2 (proposers/parallel.py; the launcher downloads it).
DRAFT_PINS = {
    'dflash2': ('incoai/Qwen3.8-27B-DFlash2', 'dedf8df68adfb1afeaf7b7480c0a0243108177b4'),
}
# Each pinned draft's acceptance calibration (tools/fit_draft_calibration.py refits one): a position's probability of
# acceptance, given the positions before it were accepted, is sigmoid(w . (logit p, gap, j / 15, 1)) with p the
# lattice's top conditional probability at position j and gap its top-1 minus top-2 log-probability. Fitted on 27,046
# drafted positions (one Spark, exact greedy, nine prompts: six chats and the three benchmark workloads at 10K, the
# tool's first prompt set); AUC 0.92 on prompts left out of the fit.
DRAFT_CALIBRATION = {
    'dflash2': (0.9693, -0.3068, -0.8203, 0.0344),
}
