# SPDX-License-Identifier: Apache-2.0
"""The Flash-Next video path's pure math (model/flashnext/vision.py + api/media.py): the clip-level smart resize
against the 25,165,824-pixel total budget, the frame sampling and timestamps (the Qwen3VL processor's arithmetic),
the token counts, and the engine's video pair bytes round trip. GPU parity (tower, mrope) is tools/video_parity.py."""

import numpy as np
import pytest

from hibana.engine import media
from hibana.model.flashnext import vision as fv


def test_video_size_rounds_and_budgets():
    # a 32-frame 640x480 clip fits the total budget untouched
    assert fv.video_size(32, 640, 480) == (640, 480)
    # 128 frames of 640x480: 128*480*640 = 39.3M > 25.2M -> shrunk to 32-aligned sides
    w, h = fv.video_size(128, 640, 480)
    assert round(128 * h * w) <= fv.VIDEO_MAX_PIXELS and h % 32 == 0 and w % 32 == 0
    # the min-pixels clamp grows a tiny clip's frames
    w, h = fv.video_size(4, 40, 24)
    assert 4 * h * w >= fv.VIDEO_MIN_PIXELS
    # a side below the factor scales up first
    assert fv.video_size(2, 10, 10)[0] >= 32
    with pytest.raises(ValueError):
        fv.video_size(4, 6100, 30)  # aspect > 200


def test_video_grid_pairs_frames():
    t, rows, cols = fv.video_grid(6, 448, 320)
    assert t == 3 and (rows, cols) == (320 // 32, 448 // 32)
    assert fv.video_tokens(6, 448, 320) == 3 * 10 * 14
    t, rows, cols = fv.video_grid(5, 448, 320)  # an odd frame count pairs with its last repeated
    assert t == 3


def test_sample_indices_and_timestamps():
    # 2 fps of a 24 fps clip: int(48 / 24 * 2) = 4 uniform samples
    assert fv.sample_indices(48, 24.0) == [0, 16, 31, 47]
    # a short clip floors toward min_frames but never past what exists
    idx = fv.sample_indices(3, 24.0)
    assert len(idx) == 3 and idx[0] == 0 and idx[-1] == 2
    idx = fv.sample_indices(30, 24.0)
    assert len(idx) == 4 and idx[0] == 0 and idx[-1] == 29  # int(30/24*2)=2 -> min_frames 4
    # a long clip caps at HIBANA_VIDEO_MAX_FRAMES
    assert len(fv.sample_indices(10000, 24.0)) == fv.MAX_FRAMES
    # timestamps: one a pair, each pair's two frames' mean
    ts = fv.timestamps([0, 16, 31, 47], 24.0)
    assert ts == pytest.approx([(0 + 16 / 24) / 2, (31 / 24 + 47 / 24) / 2])
    # an odd index list pads with its last index, as the processor does
    assert len(fv.timestamps([0, 5, 9], 10.0)) == 2
    assert fv.timestamp_text(ts[0]) == "<0.3 seconds>"


def test_pair_bytes_roundtrip():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 255, (9, 8, 3), dtype=np.uint8), rng.integers(0, 255, (9, 8, 3), dtype=np.uint8)
    raw = media.pair_bytes(a, b)
    assert raw.startswith(media.PAIR_MAGIC)
    back = media.decode(raw)  # media.decode dispatches pair bytes
    assert back.shape == (2, 9, 8, 3) and (back[0] == a).all() and (back[1] == b).all()
    with pytest.raises(ValueError):
        media.pair_bytes(a, b[:-1])  # the two frames must match
    with pytest.raises(ValueError):
        media.decode_pair(raw[:-2])
