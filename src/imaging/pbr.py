from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NormalConvention = Literal["OPENGL", "DIRECTX"]

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class PBRParams:
    seamless: bool = True
    seam_blend: float = 0.18
    height_contrast: float = 1.0
    height_invert: bool = False
    normal_strength: float = 1.0
    normal_convention: NormalConvention = "OPENGL"
    roughness_base: float = 0.6
    roughness_contrast: float = 0.5
    ao_radius: float = 6.0
    ao_strength: float = 1.0
    metallic: float = 0.0


def as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    return np.ascontiguousarray(arr[:, :, :3])


def luminance(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb[..., :3] @ _LUMA, dtype=np.float32)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(image, dtype=np.float32)
    h, w = image.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    response = np.exp(-2.0 * (np.pi**2) * (sigma**2) * (fx**2 + fy**2))
    if image.ndim == 2:
        return np.fft.irfft2(np.fft.rfft2(image) * response, s=(h, w)).astype(np.float32)
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        out[..., c] = np.fft.irfft2(np.fft.rfft2(image[..., c]) * response, s=(h, w))
    return out


def _edge_window(n: int, blend: float) -> np.ndarray:
    ramp = max(1.0, blend * n)
    t = np.minimum(np.arange(n), np.arange(n)[::-1]) / ramp
    t = np.clip(t, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def make_seamless(image: np.ndarray, blend: float = 0.18) -> np.ndarray:
    h, w = image.shape[:2]
    shifted = np.roll(image, (h // 2, w // 2), axis=(0, 1))
    weight = _edge_window(h, blend)[:, None] * _edge_window(w, blend)[None, :]
    if image.ndim == 3:
        weight = weight[:, :, None]
    return (image * weight + shifted * (1.0 - weight)).astype(np.float32)


def _normalize(x: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return np.full_like(x, 0.5, dtype=np.float32) if hi - lo < 1e-6 else np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def height_from_albedo(rgb: np.ndarray, contrast: float = 1.0, invert: bool = False) -> np.ndarray:
    lum = luminance(rgb)
    h, w = lum.shape
    broad = gaussian_blur(lum, sigma=max(h, w) / 10.0)
    detail = _normalize(lum - broad + broad.mean())
    if invert:
        detail = 1.0 - detail
    return np.clip(0.5 + (detail - 0.5) * contrast, 0.0, 1.0).astype(np.float32)


def _sobel(height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def r(dy: int, dx: int) -> np.ndarray:
        return np.roll(height, (dy, dx), axis=(0, 1))

    gx = (r(1, -1) + 2 * r(0, -1) + r(-1, -1) - r(1, 1) - 2 * r(0, 1) - r(-1, 1)) / 8.0
    gy = (r(-1, 1) + 2 * r(-1, 0) + r(-1, -1) - r(1, 1) - 2 * r(1, 0) - r(1, -1)) / 8.0
    return gx.astype(np.float32), gy.astype(np.float32)


def normal_from_height(height: np.ndarray, strength: float = 1.0, convention: NormalConvention = "OPENGL") -> np.ndarray:
    gx, g_row = _sobel(height)
    k = strength * 8.0 * (max(height.shape) / 1024.0)
    nx = -gx * k
    ny = g_row * k
    if convention == "DIRECTX":
        ny = -ny
    nz = np.ones_like(nx)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def decode_normals(encoded: np.ndarray) -> np.ndarray:
    return encoded.astype(np.float32) * 2.0 - 1.0


def roughness_from_albedo(rgb: np.ndarray, base: float = 0.6, contrast: float = 0.5) -> np.ndarray:
    lum = luminance(rgb)
    rough = base + (float(lum.mean()) - lum) * contrast
    return np.clip(rough, 0.0, 1.0).astype(np.float32)


def ao_from_height(height: np.ndarray, radius: float = 6.0, strength: float = 1.0) -> np.ndarray:
    cavity = np.clip(gaussian_blur(height, radius) - height, 0.0, 1.0)
    return np.clip(1.0 - cavity * (4.0 * strength), 0.0, 1.0).astype(np.float32)


def metallic_map(shape: tuple[int, int], value: float = 0.0, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        return np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0) * float(value or 1.0)
    return np.full(shape, float(value), dtype=np.float32)


DEFAULT_PARAMS = PBRParams()


def derive(image: np.ndarray, params: PBRParams = DEFAULT_PARAMS) -> dict[str, np.ndarray]:
    rgb = as_rgb(image)
    if params.seamless:
        rgb = make_seamless(rgb, params.seam_blend)
    height = height_from_albedo(rgb, params.height_contrast, params.height_invert)
    return {
        "base_color": rgb,
        "height": height,
        "normal": normal_from_height(height, params.normal_strength, params.normal_convention),
        "roughness": roughness_from_albedo(rgb, params.roughness_base, params.roughness_contrast),
        "ao": ao_from_height(height, params.ao_radius, params.ao_strength),
        "metallic": metallic_map(height.shape, params.metallic),
    }
