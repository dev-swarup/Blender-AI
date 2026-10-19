from __future__ import annotations

from typing import Any

import bmesh
import bpy

from ..tools.common import SELECTION, edit_mode, get_mesh_object, pick_edges, pick_faces, pick_verts, require_selection, ui_context
from ..tools.registry import ToolError, boolean, integer, number, obj, string, tool

OPS = {
    "relax": "looptools_relax",
    "circle": "looptools_circle",
    "bridge": "looptools_bridge",
    "flatten": "looptools_flatten",
    "space": "looptools_space",
    "curve": "looptools_curve",
}


def _run(name: str, selection: dict[str, Any], op_name: str, kwargs: dict[str, Any]) -> str:
    ob = get_mesh_object(name)
    op = getattr(bpy.ops.mesh, op_name, None)
    if op is None:
        raise ToolError("LoopTools isn't enabled.")
    with edit_mode(ob):
        bm = bmesh.from_edit_mesh(ob.data)
        for seq in (bm.verts, bm.edges, bm.faces):
            for el in seq:
                el.select_set(False)
        if selection.get("by") in ("sharp", "boundary", "indices", "position"):
            picked: list[Any] = pick_edges(bm, ob, selection)
        elif selection.get("by") in ("normal", "material"):
            picked = pick_faces(bm, ob, selection)
        else:
            picked = pick_verts(bm, ob, selection)
        require_selection(picked, "elements")
        for el in picked:
            el.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(ob.data)
        with bpy.context.temp_override(**(ui_context() or {}), active_object=ob, edit_object=ob):
            result = op(**kwargs)
    if "FINISHED" not in result:
        raise ToolError(f"LoopTools {op_name} didn't run on that selection ({result}).")
    return f"{op_name.replace('looptools_', 'LoopTools ')} on {len(picked)} elements of {ob.name}"


_COMMON = {"name": string("Mesh object."), "selection": {**SELECTION, "description": "Elements (loops) to act on."}}


@tool(
    obj({**_COMMON, "iterations": integer("Smoothing passes.", minimum=1, maximum=25, default=3)}, ["name", "selection"]),
    group="integrations",
    requires="looptools",
)
def loop_relax(name: str, selection: dict[str, Any], iterations: int = 3) -> str:
    """Smooth the selected loops without shrinking the shape (LoopTools Relax)."""
    nearest = min((1, 3, 5, 10, 25), key=lambda n: abs(n - iterations))
    return _run(name, selection, OPS["relax"], {"iterations": str(nearest)})


@tool(
    obj(
        {
            **_COMMON,
            "fit": string("How to fit the circle.", enum=["best", "inside"], default="best"),
            "regular": boolean("Space vertices evenly.", default=True),
        },
        ["name", "selection"],
    ),
    group="integrations",
    requires="looptools",
)
def loop_circle(name: str, selection: dict[str, Any], fit: str = "best", regular: bool = True) -> str:
    """Make the selected loops perfectly circular (LoopTools Circle), e.g. before extruding a pipe."""
    return _run(name, selection, OPS["circle"], {"fit": fit, "regular": regular})


@tool(
    obj(
        {
            **_COMMON,
            "segments": integer("Segments across the bridge.", minimum=1, maximum=64, default=1),
            "interpolation": string("Bridge shape.", enum=["cubic", "linear"], default="cubic"),
            "twist": integer("Twist offset.", minimum=-50, maximum=50, default=0),
        },
        ["name", "selection"],
    ),
    group="integrations",
    requires="looptools",
)
def loop_bridge(name: str, selection: dict[str, Any], segments: int = 1, interpolation: str = "cubic", twist: int = 0) -> str:
    """Bridge two selected loops with a smooth tube (LoopTools Bridge)."""
    return _run(name, selection, OPS["bridge"], {"segments": segments, "interpolation": interpolation, "twist": twist})


@tool(
    obj({**_COMMON, "influence": number("Strength 0..100.", minimum=0, maximum=100, default=100)}, ["name", "selection"]),
    group="integrations",
    requires="looptools",
)
def loop_flatten(name: str, selection: dict[str, Any], influence: float = 100) -> str:
    """Flatten the selection onto its best-fit plane (LoopTools Flatten)."""
    return _run(name, selection, OPS["flatten"], {"influence": influence})


@tool(
    obj({**_COMMON, "influence": number("Strength 0..100.", minimum=0, maximum=100, default=100)}, ["name", "selection"]),
    group="integrations",
    requires="looptools",
)
def loop_space(name: str, selection: dict[str, Any], influence: float = 100) -> str:
    """Space the vertices of the selected loops evenly (LoopTools Space)."""
    return _run(name, selection, OPS["space"], {"influence": influence})
