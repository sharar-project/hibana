from sharar import config


def test_layer_types_pattern(snapshot):
    cfg = config.load_model_config(snapshot)
    types = config.layer_types(cfg)
    assert types.count("full_attention") == 16
    assert types.count("linear_attention") == 48
    assert all(t == "full_attention" for t in types[3::4])


def test_check_without_gpu(snapshot):
    report = config.check(require_gpu=False)
    assert report["revision"] == config.MODEL_REVISION
    assert report["snapshot"] == str(snapshot)
