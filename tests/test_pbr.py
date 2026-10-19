from __future__ import annotations

import numpy as np
import pytest

from blender_ai.imaging import pbr


def smooth_field(h: int = 128, w: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return pbr.gaussian_blur(rng.random((h, w, 3)).astype(np.float32), 4.0)


def test_seamless_edges_match() -> None:
    h, w = 96, 128
    ramp = (np.linspace(0, 1, w)[None, :] * 0.7 + np.linspace(0, 1, h)[:, None] * 0.3).astype(np.float32)
    img = np.repeat(ramp[:, :, None], 3, axis=2)
    tiled = pbr.make_seamless(img)
    assert np.abs(img[:, 0] - img[:, -1]).mean() > 0.5
    assert np.abs(tiled[:, 0] - tiled[:, -1]).mean() < 0.02
    assert np.abs(tiled[0] - tiled[-1]).mean() < 0.02
    assert tiled.shape == img.shape and tiled.dtype == np.float32


def test_flat_height_gives_flat_normal() -> None:
    normal = pbr.normal_from_height(np.full((32, 32), 0.5, dtype=np.float32))
    assert np.allclose(normal, [0.5, 0.5, 1.0], atol=1e-6)


def test_normals_are_unit_length() -> None:
    height = pbr.height_from_albedo(smooth_field())
    n = pbr.decode_normals(pbr.normal_from_height(height, strength=3.0))
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-5)


def test_normal_conventions_flip_green() -> None:
    rows = np.linspace(0, 1, 64, dtype=np.float32)[:, None].repeat(64, axis=1)
    gl = pbr.decode_normals(pbr.normal_from_height(rows, convention="OPENGL"))
    dx = pbr.decode_normals(pbr.normal_from_height(rows, convention="DIRECTX"))
    center = (32, 32)
    assert gl[center][1] > 1e-3 and dx[center][1] < -1e-3
    assert np.isclose(gl[center][0], dx[center][0]) and np.isclose(gl[center][2], dx[center][2])


def test_gaussian_blur_wraps_and_keeps_mean() -> None:
    img = np.zeros((16, 16), dtype=np.float32)
    img[0, 0] = 1.0
    out = pbr.gaussian_blur(img, 1.5)
    assert np.isclose(out.sum(), 1.0, atol=1e-5)
    assert out[-1, -1] > 0.01, "energy wraps across the corner"


def test_height_and_ao_ranges() -> None:
    height = pbr.height_from_albedo(smooth_field(), contrast=1.5)
    ao = pbr.ao_from_height(height)
    for m in (height, ao):
        assert m.min() >= 0.0 and m.max() <= 1.0 and m.dtype == np.float32
    assert ao.mean() > 0.5


def test_roughness_from_albedo() -> None:
    img = smooth_field()
    low = pbr.roughness_from_albedo(img, base=0.3, contrast=0.5)
    high = pbr.roughness_from_albedo(img, base=0.8, contrast=0.5)
    flat = pbr.roughness_from_albedo(img, base=0.6, contrast=0.0)
    assert low.shape == img.shape[:2]
    for m in (low, high, flat):
        assert m.min() >= 0.0 and m.max() <= 1.0
    assert high.mean() > low.mean()
    assert np.allclose(flat, 0.6, atol=1e-6)


def test_derive_full_set() -> None:
    maps = pbr.derive(smooth_field(64, 64))
    assert set(maps) == {"base_color", "height", "normal", "roughness", "ao", "metallic"}
    assert maps["normal"].shape == (64, 64, 3) and maps["metallic"].shape == (64, 64)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_metallic_scalar(value: float) -> None:
    assert np.all(pbr.metallic_map((4, 4), value) == value)
