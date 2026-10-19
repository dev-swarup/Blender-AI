from __future__ import annotations

from typing import Any

import addon_utils
import bpy

from ..tools.common import get_armature, op_context
from ..tools.registry import ToolError, number, obj, string, tool, vec3
from ..tools.rigging import armature_summary

METARIGS = {
    "human": "armature_human_metarig_add",
    "basic_human": "armature_basic_human_metarig_add",
    "basic_quadruped": "armature_basic_quadruped_metarig_add",
    "bird": "armature_bird_metarig_add",
    "cat": "armature_cat_metarig_add",
    "horse": "armature_horse_metarig_add",
    "shark": "armature_shark_metarig_add",
    "wolf": "armature_wolf_metarig_add",
}


def ensure_rigify() -> None:
    if "rigify" not in bpy.context.preferences.addons:
        try:
            addon_utils.enable("rigify", default_set=True, persistent=True)
        except (ModuleNotFoundError, RuntimeError) as exc:
            raise ToolError(f"Couldn't enable the bundled Rigify add-on: {exc}") from exc


@tool(
    obj(
        {
            "kind": string("Metarig template.", enum=list(METARIGS), default="human"),
            "name": string("Name for the metarig."),
            "location": vec3("Where to put it (feet at this point)."),
            "height": number("Scale the metarig to this height in meters (human default is about 1.8).", minimum=0.05),
        }
    ),
    group="rigging",
    requires="rigify",
)
def add_rigify_metarig(kind: str = "human", name: str | None = None, location: list[float] | None = None, height: float | None = None) -> dict[str, Any]:
    """Add a Rigify metarig template (enables the bundled Rigify add-on if needed). Fit its bones to the
    character with edit_bone, then call generate_rigify_rig."""
    ensure_rigify()
    before = set(bpy.data.objects)
    with op_context():
        getattr(bpy.ops.object, METARIGS[kind])()
    new = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"]
    if not new:
        raise ToolError("Rigify didn't create a metarig.")
    rig = new[0]
    if name:
        rig.name = name
    if location:
        rig.location = location
    if height:
        bpy.context.view_layer.update()
        rig.scale = [height / max(rig.dimensions.z, 1e-6)] * 3
        with op_context(rig, [rig]):
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return armature_summary(rig)


@tool(obj({"metarig": string("The Rigify metarig armature.")}, ["metarig"]), group="rigging", requires="rigify")
def generate_rigify_rig(metarig: str) -> dict[str, Any]:
    """Generate the final animation rig (controls, IK/FK) from a fitted Rigify metarig."""
    ensure_rigify()
    meta = get_armature(metarig)
    before = set(bpy.data.objects)
    with op_context(meta, [meta]):
        bpy.ops.pose.rigify_generate()
    new = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"]
    rig = new[0] if new else getattr(meta.data, "rigify_target_rig", None)
    if rig is None:
        raise ToolError("Rigify generation produced no rig; check the metarig in the console output.")
    controls = [b.name for b in rig.pose.bones if not b.name.startswith(("ORG-", "DEF-", "MCH-"))]
    return {"rig": rig.name, "bones": len(rig.pose.bones), "controls": controls[:80]}
