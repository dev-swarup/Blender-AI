from __future__ import annotations

import importlib

from .registry import REGISTRY, ToolRegistry

MODULES = (
    "advanced",
    "animation",
    "lights_camera",
    "materials",
    "mesh",
    "modifiers",
    "objects",
    "python_fallback",
    "render",
    "rigging",
    "scene",
    "textures",
    "uv",
    "world",
)
INTEGRATIONS = ("extra_objects", "looptools", "mpfb", "rigify")


def load() -> ToolRegistry:
    for name in MODULES:
        importlib.import_module(f".{name}", __package__)
    for name in INTEGRATIONS:
        importlib.import_module(f"..integrations.{name}", __package__)
    return REGISTRY
