from __future__ import annotations

import math

import bpy

from .common import edit_mode, get_objects
from .registry import ToolError, boolean, names, number, obj, string, tool


def _select_all_faces() -> None:
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="SELECT")


@tool(
    obj(
        {
            "names": names("Mesh objects to unwrap."),
            "method": string(
                "smart = angle-based islands (good default), unwrap = follow marked seams, cube/cylinder/sphere = projections.",
                enum=["smart", "unwrap", "cube", "cylinder", "sphere"],
                default="smart",
            ),
            "angle_limit_deg": number("Island angle limit for smart.", minimum=1, maximum=89, default=66),
            "island_margin": number("Space between islands (0..1 of UV space).", minimum=0.0, maximum=0.5, default=0.02),
            "cube_size": number("Projection size for cube, in meters.", minimum=0.001, default=1.0),
        },
        ["names"],
    ),
    group="uv",
)
def smart_uv_unwrap(names: list[str], method: str = "smart", angle_limit_deg: float = 66.0, island_margin: float = 0.02, cube_size: float = 1.0) -> str:
    """UV-unwrap meshes so image textures map cleanly. Run after modelling and before texturing."""
    objs = get_objects(names, ("MESH",))
    for ob in objs:
        if not ob.data.polygons:
            raise ToolError(f"{ob.name!r} has no faces to unwrap.")
        with edit_mode(ob):
            _select_all_faces()
            if method == "smart":
                bpy.ops.uv.smart_project(angle_limit=math.radians(angle_limit_deg), island_margin=island_margin)
            elif method == "unwrap":
                bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=island_margin)
            elif method == "cube":
                bpy.ops.uv.cube_project(cube_size=cube_size)
            elif method == "cylinder":
                bpy.ops.uv.cylinder_project()
            else:
                bpy.ops.uv.sphere_project()
    return f"Unwrapped {', '.join(o.name for o in objs)} ({method})"


@tool(
    obj(
        {
            "names": names("Mesh objects whose UV islands to pack."),
            "margin": number("Space between islands.", minimum=0.0, maximum=0.5, default=0.02),
            "rotate": boolean("Allow rotating islands for a tighter fit.", default=True),
        },
        ["names"],
    ),
    group="uv",
)
def pack_uvs(names: list[str], margin: float = 0.02, rotate: bool = True) -> str:
    """Pack UV islands into the 0..1 square with even spacing."""
    objs = get_objects(names, ("MESH",))
    for ob in objs:
        if not ob.data.uv_layers:
            raise ToolError(f"{ob.name!r} has no UVs yet; unwrap it first.")
        with edit_mode(ob):
            _select_all_faces()
            bpy.ops.uv.select_all(action="SELECT")
            bpy.ops.uv.pack_islands(margin=margin, rotate=rotate)
    return f"Packed UVs of {', '.join(o.name for o in objs)}"
