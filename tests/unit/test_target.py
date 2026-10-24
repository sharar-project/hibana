"""The target descriptor reproduces the 27B's pinned constants and derived geometry."""
import pytest

from sharar import config
from sharar.target import DEFAULT_TARGET, FLASHNEXT, NEMOTRON, QWEN38, TARGETS, get
from sharar.cache import ledger, state_pool
from sharar.cache.spec_budget import verify_bytes


def test_default_target_matches_the_pinned_constants():
    t = QWEN38
    assert get(None) is DEFAULT_TARGET is t and get("qwen38") is t and TARGETS == {"qwen38": t, "flashnext": FLASHNEXT, "nemotron": NEMOTRON}
    assert (t.model_id, t.model_revision, t.model_cache_key, t.architecture) == (config.MODEL_ID, config.MODEL_REVISION, config.MODEL_CACHE_KEY, config.MODEL_ARCHITECTURE)
    assert (t.hidden_size, t.num_layers, t.vocab_size, t.context_length) == (config.HIDDEN_SIZE, config.NUM_LAYERS, config.VOCAB_SIZE, config.CONTEXT_LENGTH)
    assert (t.num_attention_heads, t.num_kv_heads, t.head_dim, t.rotary_dim) == (config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM, int(config.HEAD_DIM * config.PARTIAL_ROTARY_FACTOR))
    assert t.attention_layer_ids == [i for i in range(config.NUM_LAYERS) if (i + 1) % config.FULL_ATTENTION_INTERVAL == 0]
    assert t.gdn_layer_ids == [i for i in range(config.NUM_LAYERS) if (i + 1) % config.FULL_ATTENTION_INTERVAL != 0]
    assert len(t.attention_layer_ids) == 16 and len(t.gdn_layer_ids) == 48


def test_derived_bytes_match_the_ledger_and_pool_constants():
    t = QWEN38
    assert t.kv_bytes_per_token == ledger.KV_BYTES_PER_TOKEN == 2 * 4 * 256 * 16
    assert t.nvfp4_kv_bytes_per_token * 16 == t.kv_bytes_per_token * 9  # codes and scales: 4.5 bits a value
    assert t.state_bytes_per_slot == ledger.STATE_BYTES_PER_SLOT
    assert (t.conv_dim, t.conv_state_len) == (state_pool.CONV_DIM, state_pool.CONV_STATE_LEN) == (10240, 3)
    # The production server reports these verify buffers for k=15 at batch four.
    assert verify_bytes(4, 16, "reconstruct") == 240254976
    assert verify_bytes(4, 16, "reconstruct", t) == verify_bytes(4, 16, "reconstruct")


def test_unknown_target_is_refused():
    with pytest.raises(ValueError):
        get("qwen39")
    with pytest.raises(ValueError):
        QWEN38.__class__(**{**QWEN38.__dict__, "layer_types": ("full_attention",) * 3})


def test_flashnext_target_geometry():
    t = FLASHNEXT
    assert get("flashnext") is t and t.family == "flashnext" and t.output_gate == "sigmoid"
    assert (t.hidden_size, t.num_layers, t.vocab_size) == (2560, 48, 248320)
    assert len(t.attention_layer_ids) == 12 and len(t.gdn_layer_ids) == 36 and t.attention_layer_ids[:2] == [3, 7]
    assert (t.conv_dim, t.conv_state_len) == (QWEN38.conv_dim, QWEN38.conv_state_len)  # the GDN geometry is the 27B's
    assert t.kv_bytes_per_token == 2 * 2 * 256 * 12 + 2 * 128 * 2 * 12 and t.rotary_dim == 64  # FP8 K/V plus BF16 raw and compressed index keys
    assert t.nvfp4_kv_bytes_per_token == 2 * 2 * 256 * 12 * 9 // 16 + 2 * 128 * 2 * 12  # the index keys stay BF16
    assert (t.num_experts, t.experts_per_token, t.moe_intermediate_size) == (512, 10, 640)
    assert t.ple_layer_ids == (2,) and t.hc_count == 4 and t.indexer_budget == 2048


def test_nemotron_maps_mamba_onto_the_recurrent_pools():
    """Nemotron's blocks keep the checkpoint's numbering: Mamba blocks are the recurrent layers (Mamba's [heads, head dim,
    state] FP32 state on the pool's [HV, V, K]), attention blocks the KV layers, MoE blocks hold no state."""
    t = NEMOTRON
    assert (t.num_layers, len(t.gdn_layer_ids), len(t.attention_layer_ids), t.layer_types.count("moe")) == (88, 40, 8, 40)
    assert t.attention_layer_ids == [7, 16, 25, 36, 47, 58, 69, 78] and t.gdn_layer_ids[:4] == [0, 2, 4, 6]
    assert t.conv_dim == 8192 + 2 * 8 * 128 and t.state_bytes_per_layer == 10240 * 3 * 2 + 128 * 64 * 128 * 4
    assert t.rotary_dim == 0 and t.kv_bytes_per_token == 2 * 2 * 128 * 8 and t.stop_token_ids == (11, 2)
