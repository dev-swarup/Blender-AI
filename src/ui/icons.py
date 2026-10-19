from __future__ import annotations

import os

import bpy
import bpy.utils.previews

from .. import state

ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")
NAMES = ("logo", "mode_model", "mode_scene", "mode_animate", "mode_material")


def load() -> None:
    st = state.get()
    if st.icons is not None:
        return
    previews = bpy.utils.previews.new()
    for name in NAMES:
        path = os.path.join(ICON_DIR, f"{name}.png")
        if os.path.isfile(path):
            previews.load(name, path, "IMAGE")
    st.icons = previews


def icon(name: str) -> int:
    previews = state.get().icons
    return 0 if previews is None or name not in previews else previews[name].icon_id


def unload() -> None:
    st = state.get()
    if st.icons is not None:
        bpy.utils.previews.remove(st.icons)
        st.icons = None


def load_pixels() -> None:
    import numpy as np

    st = state.get()
    for name in NAMES:
        path = os.path.join(ICON_DIR, f"{name}.png")
        if name in st.icon_pixels or not os.path.isfile(path):
            continue
        image = bpy.data.images.load(path, check_existing=False)
        try:
            w, h = image.size
            pixels = np.empty(w * h * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
        finally:
            bpy.data.images.remove(image)
        st.icon_pixels[name] = (w, h, pixels)


def texture(name: str) -> object | None:
    st = state.get()
    if name in st.textures:
        return st.textures[name]
    entry = st.icon_pixels.get(name)
    if entry is None:
        return None
    import gpu

    w, h, pixels = entry
    tex = gpu.types.GPUTexture((w, h), format="RGBA8", data=gpu.types.Buffer("FLOAT", w * h * 4, pixels))
    st.textures[name] = tex
    return tex
