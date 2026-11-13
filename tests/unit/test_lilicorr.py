"""LiLiCorr metadata, target compatibility and trained-length boundaries."""
import json
from pathlib import Path

import pytest

from hibana.model.lilicorr_config import validate_config
from hibana.target import NEMOTRON_BOLT, QWEN38

CONFIG = Path(__file__).resolve().parents[1] / 'data/bolt_lilicorr_config.json'


def test_published_lilicorr_configuration_and_matching_target():
    cfg = json.loads(CONFIG.read_text())
    assert validate_config(cfg, NEMOTRON_BOLT) == 7
    assert cfg['dflash_config']['target_layer_ids'] == [1, 18, 35, 51, 68, 85]
    assert NEMOTRON_BOLT.draft_pin('dflash2') == (
        'nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr', '449d24140160b8e51871bfc0f0a7338abf640aa1')
    with pytest.raises(ValueError, match='width/vocabulary'):
        validate_config(cfg, QWEN38)


@pytest.mark.parametrize('field,value', [('conv_kernel_size', 3), ('conv_group_size', 8), ('causal', True),
                                       ('lilicorr_enabled', False), ('attention_sink_bias', False),
                                       ('lilicorr_candidate_topk', 3), ('target_layer_ids', [88])])
def test_reject_incompatible_checkpoint_geometry(field, value):
    cfg = json.loads(CONFIG.read_text())
    cfg['dflash_config'][field] = value
    with pytest.raises(ValueError):
        validate_config(cfg, NEMOTRON_BOLT)


def test_reject_a_norm_epsilon_the_backbone_does_not_use():
    cfg = json.loads(CONFIG.read_text())
    cfg['rms_norm_eps'] = 1e-5
    with pytest.raises(ValueError, match='eps'):
        validate_config(cfg, NEMOTRON_BOLT)


def test_parallel_draft_directory_accepts_lilicorr(tmp_path):
    from hibana.proposers.factory import draft_snapshot
    (tmp_path / 'config.json').write_text(CONFIG.read_text())
    (tmp_path / 'model.safetensors').touch()
    assert draft_snapshot('dflash2', str(tmp_path), NEMOTRON_BOLT) == tmp_path
