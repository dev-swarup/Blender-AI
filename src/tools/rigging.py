from __future__ import annotations

import math
from typing import Any

import bpy
from mathutils import Vector

from .common import (
    SELECTION,
    bmesh_of,
    edit_mode,
    get_armature,
    get_bone,
    get_mesh_object,
    get_object,
    get_objects,
    link,
    op_context,
    pick_verts,
    require_selection,
    set_rna,
    unique_name,
)
from .registry import ToolError, array, boolean, free_object, integer, names, number, obj, string, tool, vec3

BONE = obj(
    {
        "name": string("Bone name. Use .L/.R suffixes for sides so mirroring and Rigify work."),
        "head": vec3("Head position (world space, meters)."),
        "tail": vec3("Tail position (world space, meters)."),
        "parent": string("Parent bone name."),
        "connect": boolean("Attach the head to the parent's tail.", default=False),
        "roll_deg": number("Bone roll in degrees.", default=0.0),
        "deform": boolean("Deforms the mesh when skinned.", default=True),
    },
    ["name", "head", "tail"],
)


def _to_local(arm_ob: bpy.types.Object, p: list[float]) -> Vector:
    return arm_ob.matrix_world.inverted() @ Vector(p)


def _edit_bone(arm: bpy.types.Armature, name: str) -> bpy.types.EditBone:
    eb = arm.edit_bones.get(name)
    if eb is None:
        raise ToolError(f"No bone {name!r}. Bones: {', '.join(b.name for b in arm.edit_bones)[:400]}")
    return eb


def armature_summary(arm_ob: bpy.types.Object) -> dict[str, Any]:
    mw = arm_ob.matrix_world
    return {
        "armature": arm_ob.name,
        "bones": [
            {
                "name": b.name,
                "head": [round(x, 3) for x in mw @ b.head_local],
                "tail": [round(x, 3) for x in mw @ b.tail_local],
                "parent": b.parent.name if b.parent else None,
            }
            for b in arm_ob.data.bones
        ][:120],
    }


@tool(
    obj(
        {
            "name": string("Armature object name."),
            "bones": array(BONE, "Bones to create, parents listed before children.", min_items=1),
            "location": vec3("Armature origin (usually the character's feet or the world origin)."),
            "show_in_front": boolean("Draw the bones in front of the mesh.", default=True),
        },
        ["name", "bones"],
    ),
    group="rigging",
)
def create_armature(name: str, bones: list[dict[str, Any]], location: list[float] | None = None, show_in_front: bool = True) -> dict[str, Any]:
    """Create an armature with a full bone hierarchy in one call. Place joints inside the mesh where it
    bends (hips, knees, elbows); bones point from parent joint to child joint."""
    arm = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(unique_name(name, bpy.data.objects), arm)
    link(ob)
    ob.location = location or (0.0, 0.0, 0.0)
    ob.show_in_front = show_in_front
    bpy.context.view_layer.update()
    with edit_mode(ob):
        for spec in bones:
            eb = arm.edit_bones.new(spec["name"])
            eb.head = _to_local(ob, spec["head"])
            eb.tail = _to_local(ob, spec["tail"])
            if (eb.tail - eb.head).length < 1e-5:
                raise ToolError(f"Bone {spec['name']!r} has zero length.")
            eb.roll = math.radians(spec.get("roll_deg", 0.0))
            eb.use_deform = spec.get("deform", True)
        for spec in bones:
            if spec.get("parent"):
                eb = arm.edit_bones[spec["name"]]
                eb.parent = _edit_bone(arm, spec["parent"])
                eb.use_connect = bool(spec.get("connect", False))
    return armature_summary(ob)


@tool(obj({"armature": string("Armature object."), **BONE["properties"]}, ["armature", "name", "head", "tail"]), group="rigging")
def add_bone(
    armature: str, name: str, head: list[float], tail: list[float], parent: str | None = None, connect: bool = False, roll_deg: float = 0.0, deform: bool = True
) -> dict[str, Any]:
    """Add one bone to an existing armature."""
    ob = get_armature(armature)
    arm = ob.data
    with edit_mode(ob):
        eb = arm.edit_bones.new(name)
        eb.head, eb.tail = _to_local(ob, head), _to_local(ob, tail)
        eb.roll = math.radians(roll_deg)
        eb.use_deform = deform
        if parent:
            eb.parent = _edit_bone(arm, parent)
            eb.use_connect = connect
    return armature_summary(ob)


@tool(
    obj(
        {
            "armature": string("Armature object."),
            "bone": string("Bone to edit."),
            "head": vec3("New head (world space)."),
            "tail": vec3("New tail (world space)."),
            "roll_deg": number("New roll in degrees."),
            "parent": string("New parent bone; empty string clears it."),
            "connect": boolean("Connect to the parent."),
            "new_name": string("Rename the bone."),
            "deform": boolean("Deform flag."),
        },
        ["armature", "bone"],
    ),
    group="rigging",
)
def edit_bone(
    armature: str,
    bone: str,
    head: list[float] | None = None,
    tail: list[float] | None = None,
    roll_deg: float | None = None,
    parent: str | None = None,
    connect: bool | None = None,
    new_name: str | None = None,
    deform: bool | None = None,
) -> dict[str, Any]:
    """Move, re-roll, re-parent or rename a bone (Edit mode)."""
    ob = get_armature(armature)
    arm = ob.data
    with edit_mode(ob):
        eb = _edit_bone(arm, bone)
        if head is not None:
            eb.head = _to_local(ob, head)
        if tail is not None:
            eb.tail = _to_local(ob, tail)
        if roll_deg is not None:
            eb.roll = math.radians(roll_deg)
        if parent is not None:
            eb.parent = _edit_bone(arm, parent) if parent else None
        if connect is not None:
            eb.use_connect = connect
        if deform is not None:
            eb.use_deform = deform
        if new_name:
            eb.name = new_name
    return armature_summary(ob)


@tool(
    obj(
        {
            "armature": string("Armature object."),
            "bones": names("Bones to mirror (default: every bone on the source side)."),
            "from_side": string("Side to copy from.", enum=["L", "R"], default="L"),
        },
        ["armature"],
    ),
    group="rigging",
)
def mirror_bones(armature: str, bones: list[str] | None = None, from_side: str = "L") -> dict[str, Any]:
    """Create or update the opposite-side bones by mirroring across the armature's X axis
    (Arm.L -> Arm.R). Parents are mapped to their mirrored names when those exist."""
    ob = get_armature(armature)
    arm = ob.data
    made = []
    with edit_mode(ob):
        source = [b for b in arm.edit_bones if bones is None or b.name in bones]
        source = [b for b in source if bpy.utils.flip_name(b.name) != b.name and _side(b.name) == from_side]
        if not source:
            raise ToolError(f"No bones with a .{from_side} / _{from_side} / Left/Right suffix to mirror.")
        for eb in source:
            twin_name = bpy.utils.flip_name(eb.name)
            twin = arm.edit_bones.get(twin_name) or arm.edit_bones.new(twin_name)
            twin.head = Vector((-eb.head.x, eb.head.y, eb.head.z))
            twin.tail = Vector((-eb.tail.x, eb.tail.y, eb.tail.z))
            twin.roll = -eb.roll
            twin.use_deform = eb.use_deform
            made.append(twin_name)
        for eb in source:
            twin = arm.edit_bones[bpy.utils.flip_name(eb.name)]
            if eb.parent:
                twin.parent = arm.edit_bones.get(bpy.utils.flip_name(eb.parent.name)) or eb.parent
                twin.use_connect = eb.use_connect
    return {"mirrored": made, **armature_summary(ob)}


def _side(name: str) -> str:
    for a, b in zip(name, bpy.utils.flip_name(name), strict=False):
        if a != b:
            return "L" if a.lower() == "l" else "R"
    return ""


@tool(
    obj(
        {
            "meshes": names("Mesh objects to skin."),
            "armature": string("Armature object."),
            "method": string(
                "AUTO = automatic heat weights (usual choice), ENVELOPE = bone envelopes, NAME = empty groups named after bones, NONE = parent only.",
                enum=["AUTO", "ENVELOPE", "NAME", "NONE"],
                default="AUTO",
            ),
        },
        ["meshes", "armature"],
    ),
    group="rigging",
)
def parent_with_auto_weights(meshes: list[str], armature: str, method: str = "AUTO") -> dict[str, Any]:
    """Skin meshes to an armature (Armature modifier + vertex groups). Apply mesh scale first, and keep
    the mesh closed and manifold so automatic weights succeed. Reports how many vertices got weights."""
    arm = get_armature(armature)
    objs = get_objects(meshes, ("MESH",))
    kind = {"AUTO": "ARMATURE_AUTO", "ENVELOPE": "ARMATURE_ENVELOPE", "NAME": "ARMATURE_NAME", "NONE": "ARMATURE"}[method]
    with op_context(arm, [*objs, arm]):
        bpy.ops.object.parent_set(type=kind)
    report = {}
    for ob in objs:
        weighted = sum(1 for v in ob.data.vertices if any(g.weight > 0 for g in v.groups))
        report[ob.name] = {"vertex_groups": len(ob.vertex_groups), "weighted_vertices": weighted, "total_vertices": len(ob.data.vertices)}
    note = ""
    if method == "AUTO" and any(r["weighted_vertices"] == 0 for r in report.values()):
        note = "Automatic weights failed on some meshes (often non-manifold geometry or unapplied scale)."
    return {"skinned": report, "note": note}


@tool(
    obj(
        {
            "mesh": string("Mesh object."),
            "group": string("Vertex group name (created if missing)."),
            "selection": {**SELECTION, "description": "Vertices to change."},
            "weight": number("Weight 0..1.", minimum=0.0, maximum=1.0, default=1.0),
            "mode": string("How to combine with existing weights.", enum=["REPLACE", "ADD", "SUBTRACT"], default="REPLACE"),
            "remove": boolean("Remove these vertices from the group instead.", default=False),
        },
        ["mesh", "group", "selection"],
    ),
    group="rigging",
)
def set_vertex_group_weights(mesh: str, group: str, selection: dict[str, Any], weight: float = 1.0, mode: str = "REPLACE", remove: bool = False) -> str:
    """Paint weights numerically: set, add or remove a vertex group's weight on selected vertices (fix
    skinning, or bind parts rigidly to one bone)."""
    ob = get_mesh_object(mesh)
    vg = ob.vertex_groups.get(group) or ob.vertex_groups.new(name=group)
    with bmesh_of(ob) as bm:
        indices = [v.index for v in pick_verts(bm, ob, selection)]
    require_selection(indices, "vertices")
    if remove:
        vg.remove(indices)
    else:
        vg.add(indices, weight, mode)
    return f"{'Removed' if remove else 'Set'} {len(indices)} vertices in {vg.name!r} on {ob.name}"


@tool(
    obj(
        {
            "armature": string("Armature object."),
            "bone": string("Last bone of the chain (e.g. shin or forearm)."),
            "chain_count": integer("How many bones the IK affects (2 for a leg or arm).", minimum=0, maximum=255, default=2),
            "target_bone": string("Existing bone to use as the IK target."),
            "create_target": boolean("Create a non-deforming IK target bone at the chain end.", default=True),
            "pole_bone": string("Bone used as pole target (knee/elbow direction)."),
            "pole_angle_deg": number("Pole angle.", default=-90.0),
            "target_object": string("Use an object instead of a bone as the target."),
        },
        ["armature", "bone"],
    ),
    group="rigging",
)
def add_ik_constraint(
    armature: str,
    bone: str,
    chain_count: int = 2,
    target_bone: str | None = None,
    create_target: bool = True,
    pole_bone: str | None = None,
    pole_angle_deg: float = -90.0,
    target_object: str | None = None,
) -> str:
    """Add an inverse-kinematics constraint so a limb follows a target (bone or object)."""
    ob = get_armature(armature)
    get_bone(ob, bone)
    if not target_bone and not target_object and create_target:
        target_bone = f"IK_{bone}"
        with edit_mode(ob):
            arm = ob.data
            src = _edit_bone(arm, bone)
            eb = arm.edit_bones.get(target_bone) or arm.edit_bones.new(target_bone)
            length = max((src.tail - src.head).length * 0.4, 0.05)
            eb.head = src.tail.copy()
            eb.tail = src.tail + Vector((0, length, 0))
            eb.parent = None
            eb.use_deform = False
    pb = get_bone(ob, bone)
    con = pb.constraints.new("IK")
    con.chain_count = chain_count
    if target_object:
        con.target = get_object(target_object)
    elif target_bone:
        get_bone(ob, target_bone)
        con.target = ob
        con.subtarget = target_bone
    if pole_bone:
        get_bone(ob, pole_bone)
        con.pole_target = ob
        con.pole_subtarget = pole_bone
        con.pole_angle = math.radians(pole_angle_deg)
    target = target_object or target_bone or "none"
    return f"IK on {bone} (chain {chain_count}) targeting {target}"


@tool(
    obj(
        {
            "armature": string("Armature object."),
            "bone": string("Pose bone to constrain."),
            "type": string("Constraint type, e.g. COPY_ROTATION, DAMPED_TRACK, LIMIT_ROTATION, STRETCH_TO, CHILD_OF, TRANSFORM."),
            "target": string("Target object (often the armature itself)."),
            "subtarget": string("Target bone."),
            "props": free_object('Constraint properties by name (angles in degrees), e.g. {"use_limit_x": true, "min_x": -10}.'),
        },
        ["armature", "bone", "type"],
    ),
    group="rigging",
)
def add_bone_constraint(
    armature: str, bone: str, type: str, target: str | None = None, subtarget: str | None = None, props: dict[str, Any] | None = None
) -> str:
    """Add any constraint to a pose bone."""
    ob = get_armature(armature)
    pb = get_bone(ob, bone)
    valid = bpy.types.Constraint.bl_rna.properties["type"].enum_items.keys()
    if type not in valid:
        raise ToolError(f"Unknown constraint type {type!r}. Options: {', '.join(valid)}")
    con = pb.constraints.new(type)
    if target and hasattr(con, "target"):
        con.target = get_object(target)
        if subtarget and hasattr(con, "subtarget"):
            con.subtarget = subtarget
    if props:
        set_rna(con, props, f"Constraint {con.name!r}")
    return f"Added {con.type} to {bone}"


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "key_name": string("Shape key name."),
            "selection": {**SELECTION, "description": "Vertices to move in the new shape."},
            "translate": vec3("Offset for the selected vertices, in world-space meters."),
            "scale": vec3("Scale the selected vertices about their center."),
            "value": number("Initial value.", minimum=0.0, maximum=1.0, default=0.0),
        },
        ["name", "key_name"],
    ),
    group="rigging",
)
def add_shape_key(
    name: str,
    key_name: str,
    selection: dict[str, Any] | None = None,
    translate: list[float] | None = None,
    scale: list[float] | None = None,
    value: float = 0.0,
) -> str:
    """Add a shape key (blend shape), optionally sculpting it by moving or scaling some vertices."""
    ob = get_mesh_object(name)
    if ob.data.shape_keys is None:
        ob.shape_key_add(name="Basis", from_mix=False)
    key = ob.shape_key_add(name=key_name, from_mix=False)
    key.value = value
    moved = 0
    if selection and (translate or scale):
        with bmesh_of(ob) as bm:
            idx = [v.index for v in pick_verts(bm, ob, selection)]
        require_selection(idx, "vertices")
        offset = ob.matrix_world.to_3x3().inverted_safe() @ Vector(translate or (0, 0, 0))
        center = sum((key.data[i].co for i in idx), Vector()) / len(idx)
        for i in idx:
            co = key.data[i].co
            if scale:
                co = center + Vector([(co[k] - center[k]) * scale[k] for k in range(3)])
            key.data[i].co = co + offset
        moved = len(idx)
    return f"Shape key {key.name!r} on {ob.name} ({moved} vertices moved)"


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "key": string("Shape key name."),
            "value": number("Value 0..1.", minimum=-10, maximum=10),
            "frame": integer("Also insert a keyframe at this frame."),
        },
        ["name", "key", "value"],
    ),
    group="rigging",
)
def set_shape_key_value(name: str, key: str, value: float, frame: int | None = None) -> str:
    """Set (and optionally keyframe) a shape key value."""
    ob = get_mesh_object(name)
    keys = ob.data.shape_keys
    block = keys.key_blocks.get(key) if keys else None
    if block is None:
        have = ", ".join(k.name for k in keys.key_blocks) if keys else "none"
        raise ToolError(f"{ob.name!r} has no shape key {key!r} (has: {have}).")
    block.value = value
    if frame is not None:
        block.keyframe_insert("value", frame=frame)
    return f"{key} = {value}" + (f" keyed at frame {frame}" if frame is not None else "")


DRIVER_VAR = obj(
    {
        "name": string("Variable name used in the expression."),
        "type": string("Variable type.", enum=["SINGLE_PROP", "TRANSFORMS", "ROTATION_DIFF", "LOC_DIFF"], default="TRANSFORMS"),
        "object": string("Source object."),
        "bone": string("Source bone (for armatures)."),
        "data_path": string("Property path, for SINGLE_PROP, e.g. 'location.x' or '[\"my_prop\"]'."),
        "transform_type": string("For TRANSFORMS.", enum=["LOC_X", "LOC_Y", "LOC_Z", "ROT_X", "ROT_Y", "ROT_Z", "SCALE_X", "SCALE_Y", "SCALE_Z"]),
        "space": string("Transform space.", enum=["WORLD_SPACE", "TRANSFORM_SPACE", "LOCAL_SPACE"], default="LOCAL_SPACE"),
        "object_2": string("Second object, for ROTATION_DIFF/LOC_DIFF."),
        "bone_2": string("Second bone."),
    },
    ["name", "object"],
)


@tool(
    obj(
        {
            "target": string("Object that owns the driven property."),
            "data_path": string("Driven property, e.g. 'rotation_euler', 'location', 'scale' or 'data.energy'."),
            "index": integer("Array index (0=X...), or -1 for a single value.", minimum=-1, maximum=3, default=-1),
            "bone": string("Drive a pose bone's property instead (data_path relative to the bone)."),
            "shape_key": string("Drive this shape key's value instead (target must be a mesh)."),
            "expression": string("Driver expression using the variable names, e.g. 'var * 2' or 'max(0, var)'."),
            "variables": array(DRIVER_VAR, "Driver variables.", min_items=1),
        },
        ["target", "expression", "variables"],
    ),
    group="rigging",
)
def add_driver(
    target: str,
    expression: str,
    variables: list[dict[str, Any]],
    data_path: str | None = None,
    index: int = -1,
    bone: str | None = None,
    shape_key: str | None = None,
) -> str:
    """Drive one property from others (e.g. a shape key from a bone's rotation). Simple expressions run
    without 'Auto Run Python Scripts'; anything else needs it enabled."""
    ob = get_object(target)
    owner: Any = ob
    if shape_key:
        if ob.type != "MESH" or not ob.data.shape_keys or shape_key not in ob.data.shape_keys.key_blocks:
            raise ToolError(f"{ob.name!r} has no shape key {shape_key!r}.")
        owner = ob.data.shape_keys
        path = f'key_blocks["{shape_key}"].value'
    elif bone:
        get_bone(ob, bone)
        if not data_path:
            raise ToolError("data_path is needed for a bone driver.")
        path = f'pose.bones["{bone}"].{data_path}'
    else:
        if not data_path:
            raise ToolError("data_path is needed.")
        if data_path.startswith("data.") and ob.data is not None:
            owner, path = ob.data, data_path[5:]
        else:
            path = data_path
    try:
        fcurve = owner.driver_add(path, index)
    except TypeError as exc:
        raise ToolError(f"Can't drive {path!r}: {exc}") from exc
    if isinstance(fcurve, list):
        fcurve = fcurve[0]
    driver = fcurve.driver
    driver.type = "SCRIPTED"
    driver.expression = expression
    for spec in variables:
        var = driver.variables.new()
        var.name = spec["name"]
        var.type = spec.get("type", "TRANSFORMS")
        targets = [(spec["object"], spec.get("bone"))]
        if var.type in ("ROTATION_DIFF", "LOC_DIFF"):
            targets.append((spec.get("object_2") or spec["object"], spec.get("bone_2")))
        for slot, (ob_name, bone_name) in zip(var.targets, targets, strict=False):
            slot.id = get_object(ob_name)
            if var.type == "SINGLE_PROP":
                slot.data_path = spec.get("data_path", "")
            else:
                if bone_name:
                    slot.bone_target = bone_name
                if var.type == "TRANSFORMS":
                    slot.transform_type = spec.get("transform_type", "ROT_X")
                    slot.transform_space = spec.get("space", "LOCAL_SPACE")
    return f"Driver on {ob.name}.{path}[{index}] = {expression}"
