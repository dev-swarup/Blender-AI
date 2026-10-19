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
    pkg_id: str = ""
    size: str = ""


HELPERS: dict[str, Helper] = {
    h.id: h
    for h in (
        Helper("rigify", "Rigify", ("rigify",), "https://docs.blender.org/manual/en/latest/addons/rigging/rigify/", bundled=True),
        Helper("mpfb", "MPFB (MakeHuman for Blender)", ("mpfb",), "https://extensions.blender.org/add-ons/mpfb/", pkg_id="mpfb", size="46 MB"),
        Helper(
            "looptools", "LoopTools", ("looptools", "mesh_looptools"), "https://extensions.blender.org/add-ons/looptools/", pkg_id="looptools", size="30 KB"
        ),
        Helper(
            "extra_objects",
            "Extra Mesh Objects",
            ("extra_mesh_objects", "add_mesh_extra_objects"),
            "https://extensions.blender.org/add-ons/extra-mesh-objects/",
            pkg_id="extra_mesh_objects",
            size="130 KB",
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


def missing() -> list[Helper]:
    return [h for h in HELPERS.values() if enabled_module(h.id) is None]


def status(helper: Helper) -> str:
    if enabled_module(helper.id):
        return "enabled"
    if installed_module(helper.id):
        return "installed, not enabled"
    return "not installed" + (f" ({helper.size})" if helper.size else "")


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
