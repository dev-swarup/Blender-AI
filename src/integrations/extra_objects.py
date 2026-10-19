from __future__ import annotations

from typing import Any

import bpy

from ..tools.common import describe, op_context
from ..tools.registry import ToolError, free_object, obj, string, tool, vec3

KINDS = {
    "gear": "primitive_gear",
    "worm_gear": "primitive_worm_gear",
    "gem": "primitive_gem_add",
    "diamond": "primitive_diamond_add",
    "elbow_joint": "primitive_elbow_joint_add",
    "tee_joint": "primitive_tee_joint_add",
    "wye_joint": "primitive_wye_joint_add",
    "cross_joint": "primitive_cross_joint_add",
    "n_joint": "primitive_n_joint_add",
    "twisted_torus": "primitive_twisted_torus_add",
    "supertoroid": "primitive_supertoroid_add",
    "step_pyramid": "primitive_steppyramid_add",
    "star": "primitive_star_add",
    "round_cube": "primitive_round_cube_add",
    "honeycomb": "honeycomb_add",
}


@tool(
    obj(
        {
            "kind": string("Parametric object.", enum=sorted(KINDS)),
            "settings": free_object('Operator settings, e.g. gear {"number_of_teeth": 24, "radius": 0.5, "width": 0.1}. Unknown keys list the valid ones.'),
            "name": string("Object name."),
            "location": vec3("Location in meters."),
        },
        ["kind"],
    ),
    group="integrations",
    requires="extra_objects",
)
def add_extra_mesh(kind: str, settings: dict[str, Any] | None = None, name: str | None = None, location: list[float] | None = None) -> dict[str, Any]:
    """Add a parametric mechanical or decorative mesh from Extra Mesh Objects (gears, gems, pipe joints,
    stars, round cubes...), which would be slow to model by hand."""
    op = getattr(bpy.ops.mesh, KINDS[kind], None)
    if op is None:
        raise ToolError(f"Extra Mesh Objects has no {kind} operator in this version.")
    params = dict(settings or {})
    valid = {p.identifier for p in op.get_rna_type().properties if p.identifier != "rna_type"}
    unknown = sorted(set(params) - valid)
    if unknown:
        raise ToolError(f"{kind} has no setting(s) {', '.join(unknown)}. Valid: {', '.join(sorted(valid))}")
    if location is not None and "location" in valid:
        params["location"] = location
    before = set(bpy.data.objects)
    with op_context():
        op(**params)
    new = [o for o in bpy.data.objects if o not in before]
    if not new:
        raise ToolError(f"{kind} created nothing.")
    ob = new[0]
    if name:
        ob.name = name
    if location is not None and "location" not in valid:
        ob.location = location
    return describe(ob)
