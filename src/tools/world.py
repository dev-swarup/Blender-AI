from __future__ import annotations

import math
import os
from typing import Any

import bpy

from .common import scene
from .registry import ToolError, color, number, obj, string, tool

PREFIX = "AI "


def _sky_types() -> list[str]:
    return list(bpy.types.ShaderNodeTexSky.bl_rna.properties["sky_type"].enum_items.keys())


@tool(
    obj(
        {
            "mode": string("color = flat background, sky = procedural daylight sky, hdri = environment image.", enum=["color", "sky", "hdri"]),
            "color": color("Background color for mode=color."),
            "strength": number("Background strength.", minimum=0, default=1.0),
            "hdri_path": string("Path to a .hdr/.exr file for mode=hdri."),
            "rotation_deg": number("Rotate the HDRI around Z.", default=0.0),
            "sun_elevation_deg": number("Sun height above the horizon, for mode=sky.", minimum=-10, maximum=90, default=35),
            "sun_rotation_deg": number("Sun direction around Z, for mode=sky.", default=45),
            "sky_type": string("Sky model; default picks the best available in this Blender version."),
        },
        ["mode"],
    ),
    group="world",
)
def set_world(
    mode: str,
    color: list[float] | None = None,
    strength: float = 1.0,
    hdri_path: str | None = None,
    rotation_deg: float = 0.0,
    sun_elevation_deg: float = 35.0,
    sun_rotation_deg: float = 45.0,
    sky_type: str | None = None,
) -> dict[str, Any]:
    """Set the world lighting: a flat color, a physical sky with a sun, or an HDRI image."""
    sc = scene()
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    for node in [n for n in nodes if n.name.startswith(PREFIX)]:
        nodes.remove(node)
    bg = next((n for n in nodes if n.type == "BACKGROUND"), None) or nodes.new("ShaderNodeBackground")
    out = next((n for n in nodes if n.type == "OUTPUT_WORLD"), None) or nodes.new("ShaderNodeOutputWorld")
    bg.location, out.location = (0, 300), (250, 300)
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Strength"].default_value = strength
    for link in list(bg.inputs["Color"].links):
        links.remove(link)

    def node(kind: str, label: str, x: float) -> Any:
        n = nodes.new(kind)
        n.name = n.label = PREFIX + label
        n.location = (x, 300)
        return n

    if mode == "color":
        bg.inputs["Color"].default_value = [*(color or (0.05, 0.05, 0.05)), 1.0]
        return {"world": world.name, "mode": mode}
    if mode == "sky":
        sky = node("ShaderNodeTexSky", "Sky", -300)
        available = _sky_types()
        if sky_type:
            if sky_type not in available:
                raise ToolError(f"Unknown sky_type {sky_type!r}; this Blender has {available}.")
            sky.sky_type = sky_type
        else:
            preferred = ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING", "HOSEK_WILKIE", "PREETHAM")
            sky.sky_type = next((t for t in preferred if t in available), available[0])
        if hasattr(sky, "sun_elevation"):
            sky.sun_elevation = math.radians(sun_elevation_deg)
            sky.sun_rotation = math.radians(sun_rotation_deg)
        links.new(sky.outputs["Color"], bg.inputs["Color"])
        return {"world": world.name, "mode": mode, "sky_type": sky.sky_type}

    if not hdri_path:
        raise ToolError("mode=hdri needs hdri_path.")
    path = bpy.path.abspath(hdri_path)
    if not os.path.isfile(path):
        raise ToolError(f"No file at {path}.")
    env = node("ShaderNodeTexEnvironment", "Environment", -300)
    env.image = bpy.data.images.load(path, check_existing=True)
    mapping = node("ShaderNodeMapping", "Mapping", -550)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
    coord = node("ShaderNodeTexCoord", "Coordinates", -750)
    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(env.outputs["Color"], bg.inputs["Color"])
    return {"world": world.name, "mode": mode, "image": env.image.name}
