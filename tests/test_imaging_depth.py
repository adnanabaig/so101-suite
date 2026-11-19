"""Depth frames in the browser feed: colorized uint16, uint8 frames untouched."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from so101.services import imaging
from so101.services.imaging import DEPTH_LUT, colorize_depth, encode, is_depth_frame


def test_the_lut_has_256_colors_and_none_is_black() -> None:
    assert DEPTH_LUT.shape == (256, 3) and DEPTH_LUT.dtype == np.uint8
    assert (DEPTH_LUT.sum(axis=1) > 0).all()
    assert DEPTH_LUT[0].argmax() == 2 and DEPTH_LUT[-1].argmax() == 0  # blue near, red far


def test_no_data_is_black_and_near_and_far_get_the_ends_of_the_scale() -> None:
    depth = np.tile(np.linspace(300, 1500, 100, dtype=np.uint16), (10, 1))
    depth[:, :5] = 0
    rgb = colorize_depth(depth)
    assert rgb.shape == (10, 100, 3) and rgb.dtype == np.uint8
    assert (rgb[:, :5] == 0).all()
    assert (rgb[:, 6] == DEPTH_LUT[0]).all() and (rgb[:, -1] == DEPTH_LUT[-1]).all()
    assert (colorize_depth(depth[:, :, None]) == rgb).all()


def test_all_zero_and_flat_frames_are_safe() -> None:
    assert (colorize_depth(np.zeros((4, 6, 1), np.uint16)) == 0).all()
    flat = np.full((4, 6), 800, np.uint16)
    flat[0, 0] = 0
    rgb = colorize_depth(flat)
    assert (rgb[0, 0] == 0).all() and (rgb[1:, :] == DEPTH_LUT[128]).all()


def test_large_frames_estimate_the_range_from_a_sample() -> None:
    depth = np.random.default_rng(0).integers(1, 4000, size=(480, 640), dtype=np.uint16)
    assert colorize_depth(depth).shape == (480, 640, 3)


@pytest.mark.parametrize(
    ("shape", "dtype", "is_depth"),
    [
        ((4, 6), np.uint16, True),
        ((4, 6, 1), np.uint16, True),
        ((4, 6, 3), np.uint16, False),
        ((4, 6, 1), np.uint8, False),
        ((4, 6, 3), np.uint8, False),
    ],
)
def test_is_depth_frame(shape: tuple[int, ...], dtype: type, is_depth: bool) -> None:
    assert is_depth_frame(np.zeros(shape, dtype)) is is_depth


def test_encode_colorizes_depth_and_passes_rgb_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    seen: list[np.ndarray] = []
    real_png = imaging._png
    monkeypatch.setattr(imaging, "_png", lambda frame: seen.append(frame) or real_png(frame))
    depth = np.full((8, 8, 1), 1000, np.uint16)
    payload, content_type = encode(depth)
    assert content_type == "image/png" and payload.startswith(b"\x89PNG")
    assert seen[-1].shape == (8, 8, 3) and seen[-1].dtype == np.uint8
    rgb = np.full((8, 8, 3), 7, np.uint8)
    encode(rgb)
    assert seen[-1] is rgb
