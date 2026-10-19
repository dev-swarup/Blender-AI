from __future__ import annotations

import struct
import zlib

import numpy as np

_COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}


def to_uint8(pixels: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(np.asarray(arr, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    return arr


def encode_png(pixels: np.ndarray, level: int = 6) -> bytes:
    arr = to_uint8(pixels)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in _COLOR_TYPES:
        raise ValueError(f"unsupported image shape {arr.shape}")
    h, w, c = arr.shape
    rows = arr.reshape(h, w * c)
    up = rows.copy()
    up[1:] = rows[1:] - rows[:-1]
    raw = np.empty((h, w * c + 1), dtype=np.uint8)
    raw[:, 0] = 2
    raw[:, 1:] = up
    ihdr = struct.pack(">IIBBBBB", w, h, 8, _COLOR_TYPES[c], 0, 0, 0)
    return b"".join((b"\x89PNG\r\n\x1a\n", _chunk(b"IHDR", ihdr), _chunk(b"IDAT", zlib.compress(raw.tobytes(), level)), _chunk(b"IEND", b"")))


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def downscale(pixels: np.ndarray, max_side: int) -> np.ndarray:
    h, w = pixels.shape[:2]
    factor = int(np.ceil(max(h, w) / max_side))
    if factor <= 1:
        return pixels
    h2, w2 = h // factor, w // factor
    cropped = pixels[: h2 * factor, : w2 * factor]
    shape = (h2, factor, w2, factor, *pixels.shape[2:])
    return cropped.reshape(shape).mean(axis=(1, 3)).astype(pixels.dtype)
