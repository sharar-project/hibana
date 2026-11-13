"""Backend-independent validation of a LiLiCorr checkpoint."""

def validate_config(cfg, target=None):
    if cfg.get('architectures') != ['LiLiCorrDraftModel']:
        raise ValueError('expected LiLiCorrDraftModel')
    d = cfg['dflash_config']
    if (not d.get('lilicorr_enabled') or d.get('projector_type') != 'lilicorr'
            or d.get('conv_kernel_size') != 2 or d.get('conv_group_size') != 16
            or d.get('causal') is not False):
        raise ValueError('unsupported LiLiCorr backbone or convolution geometry')
    if not d.get('attention_sink_bias') or not d.get('use_swa') or cfg['sliding_window'] != d['swa_window_size']:
        raise ValueError('LiLiCorr requires sliding attention and trained sink biases')
    if cfg.get('rms_norm_eps', 1e-6) != 1e-6:
        raise ValueError('the parallel draft backbone normalizes with eps 1e-6')
    if cfg.get('has_own_lm_head', False):
        raise ValueError('this LiLiCorr loader shares the target vocabulary head')
    topk = d['lilicorr_candidate_topk']
    if topk not in (1, 2, 4, 8, 16) or cfg['block_size'] < 2:
        raise ValueError('unsupported LiLiCorr lattice geometry')
    h = d['lilicorr_hidden_size']
    if h % d['lilicorr_num_heads'] or cfg['hidden_size'] % 16:
        raise ValueError('unsupported LiLiCorr head geometry')
    if target is not None:
        if cfg['hidden_size'] != target.hidden_size or cfg['vocab_size'] != target.vocab_size:
            raise ValueError('LiLiCorr embedding width/vocabulary differ from the target')
        taps = d['target_layer_ids']
        if len(set(taps)) != len(taps) or any(not 0 <= lid < target.num_layers for lid in taps):
            raise ValueError('LiLiCorr feature taps are outside the target')
    return cfg['block_size'] - 1

