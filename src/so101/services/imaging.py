"""Encode RGB numpy frames to bytes for the browser.

Prefers JPEG (cv2 or Pillow) for speed/size; falls back to a dependency-free
PNG encoder (stdlib zlib) so the video feed works in pure-mock mode with
nothing installed. Depth frames (uint16, HxW or HxWx1, 0 = no data) are
colorized first so a depth stream can share the same video tiles.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

#: Depth display is ranged per frame on these percentiles of the valid pixels.
DEPTH_RANGE_PERCENTILES = (2.0, 98.0)
#: Percentiles are estimated from at most about this many valid pixels.
_PERCENTILE_SAMPLES = 65_536


def _jet_lut() -> np.ndarray:
    # Classic jet: dark blue (near) through cyan, yellow, to dark red (far). Its
    # darkest entry is still blue, so black stays free to mean "no data".
    x = np.linspace(0.0, 1.0, 256)
    channels = [np.clip(1.5 - np.abs(4.0 * x - offset), 0.0, 1.0) for offset in (3.0, 2.0, 1.0)]
    return np.rint(np.stack(channels, axis=-1) * 255).astype(np.uint8)


DEPTH_LUT = _jet_lut()


def is_depth_frame(frame: np.ndarray) -> bool:
    return frame.dtype == np.uint16 and (
        frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1)
    )


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """uint16 depth (HxW or HxWx1, 0 = no data) to HxWx3 uint8 RGB for display.

    Each frame is ranged on its own valid pixels, so colors show relative depth
    within the frame, not absolute distance. No-data pixels are black.
    """

    values2d = depth[:, :, 0] if depth.ndim == 3 else depth
    rgb = np.zeros((*values2d.shape, 3), dtype=np.uint8)
    valid = values2d > 0
    if not valid.any():
        return rgb
    values = values2d[valid].astype(np.float32)
    step = max(1, values.size // _PERCENTILE_SAMPLES)
    low, high = np.percentile(values[::step], DEPTH_RANGE_PERCENTILES)
    if high > low:
        index = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    else:  # a flat frame: one color, mid-scale
        index = np.full(values.shape, 128, dtype=np.uint8)
    rgb[valid] = DEPTH_LUT[index]
    return rgb


def _png(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8))
    # Each scanline is prefixed with a filter-type byte (0 = none).
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    idat = zlib.compress(raw, 6)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def encode(rgb: np.ndarray) -> tuple[bytes, str]:
    """Return ``(bytes, content_type)`` for an RGB frame or a uint16 depth frame."""
    if is_depth_frame(rgb):
        rgb = colorize_depth(rgb)
    try:
        import cv2

        bgr = rgb[:, :, ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            return buf.tobytes(), "image/jpeg"
    except Exception:
        pass
    try:
        import io

        from PIL import Image

        bio = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(bio, format="JPEG", quality=80)
        return bio.getvalue(), "image/jpeg"
    except Exception:
        pass
    return _png(rgb), "image/png"
