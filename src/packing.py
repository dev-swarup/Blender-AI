from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

import bpy

_PACKABLE_SOURCES = {"FILE", "GENERATED", "TILED"}
_NOT_PACKABLE = {"RENDER_RESULT", "COMPOSITING"}


def _enabled() -> bool:
    from .preferences import get_prefs

    prefs = get_prefs()
    return prefs is not None and prefs.pack_textures


def image_ids() -> set[int] | None:
    return {img.as_pointer() for img in bpy.data.images} if _enabled() else None


def pack_image(image: Any) -> bool:
    if image.packed_file is not None or image.type in _NOT_PACKABLE or image.source not in _PACKABLE_SOURCES:
        return False
    if image.source == "FILE" and not os.path.isfile(bpy.path.abspath(image.filepath)):
        return False
    if image.source == "GENERATED" and not image.is_dirty:
        return False
    try:
        image.pack()
    except RuntimeError as exc:
        print(f"Blender AI: couldn't pack image {image.name!r}: {exc}", file=sys.stderr)
        return False
    return True


def pack_new(before: set[int] | None) -> int:
    if before is None:
        return 0
    return sum(1 for img in bpy.data.images if img.as_pointer() not in before and pack_image(img))


def pack_generated_files(texture_dir: str) -> int:
    roots = [os.path.normcase(os.path.abspath(p)) for p in (texture_dir, tempfile.gettempdir()) if p]
    count = 0
    for img in bpy.data.images:
        if img.packed_file is not None or img.source != "FILE":
            continue
        path = os.path.normcase(os.path.abspath(bpy.path.abspath(img.filepath)))
        if any(path.startswith(root + os.sep) for root in roots) and pack_image(img):
            count += 1
    return count
