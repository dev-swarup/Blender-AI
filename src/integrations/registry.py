from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import addon_utils
import bpy


@dataclass(frozen=True)
class Helper:
    id: str
    name: str
    modules: tuple[str, ...]
    url: str
    bundled: bool = False


HELPERS: dict[str, Helper] = {
    h.id: h
    for h in (
        Helper("rigify", "Rigify", ("rigify",), "https://docs.blender.org/manual/en/latest/addons/rigging/rigify/", bundled=True),
        Helper("mpfb", "MPFB (MakeHuman for Blender)", ("mpfb",), "https://extensions.blender.org/add-ons/mpfb/"),
        Helper("looptools", "LoopTools", ("looptools", "mesh_looptools"), "https://extensions.blender.org/add-ons/looptools/"),
        Helper(
            "extra_objects",
            "Extra Mesh Objects",
            ("extra_mesh_objects", "add_mesh_extra_objects"),
            "https://extensions.blender.org/add-ons/extra-mesh-objects/",
        ),
    )
}


def _short(module: str) -> str:
    return module.rsplit(".", 1)[-1]


def enabled_module(helper_id: str) -> str | None:
    helper = HELPERS[helper_id]
    for addon in bpy.context.preferences.addons:
        if _short(addon.module) in helper.modules:
            return addon.module
    return None


def installed_module(helper_id: str) -> str | None:
    helper = HELPERS[helper_id]
    for mod in addon_utils.modules(refresh=False):
        if _short(mod.__name__) in helper.modules:
            return mod.__name__
    return None


def available(helper_id: str) -> bool:
    if helper_id not in HELPERS:
        return False
    if enabled_module(helper_id):
        return True
    return HELPERS[helper_id].bundled and installed_module(helper_id) is not None


def report() -> dict[str, Any]:
    return {
        h.id: {
            "name": h.name,
            "enabled": enabled_module(h.id) is not None,
            "installed": installed_module(h.id) is not None,
            "usable": available(h.id),
            "get_it": h.url,
        }
        for h in HELPERS.values()
    }
