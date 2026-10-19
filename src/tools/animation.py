from __future__ import annotations

import math
from typing import Any

import bpy
from mathutils import Quaternion, Vector

from .common import get_action, get_armature, get_bone, get_object, link, op_context, scene, unique_name
from .registry import ToolError, array, boolean, free_object, integer, number, obj, scalar, string, tool, vec3

ROTATION_PATHS = {"rotation_euler", "delta_rotation_euler"}


def fcurves_of(id_data: Any) -> list[bpy.types.FCurve]:
    ad = getattr(id_data, "animation_data", None)
    if ad is None or ad.action is None:
        return []
    action = ad.action
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    from bpy_extras import anim_utils

    bag = anim_utils.action_get_channelbag_for_slot(action, ad.action_slot)
    return list(bag.fcurves) if bag else []


def ensure_fcurve(id_data: Any, data_path: str, index: int) -> bpy.types.FCurve:
    for fc in fcurves_of(id_data):
        if fc.data_path == data_path and fc.array_index == index:
            return fc
    ad = id_data.animation_data or id_data.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{id_data.name}Action")
    action = ad.action
    if hasattr(action, "fcurves"):
        return action.fcurves.new(data_path, index=index)
    from bpy_extras import anim_utils

    if ad.action_slot is None:
        slot = action.slots.new(id_data.id_type, id_data.name)
        ad.action_slot = slot
    bag = anim_utils.action_ensure_channelbag_for_slot(action, ad.action_slot)
    return bag.fcurves.new(data_path, index=index)


def _owner(ob: bpy.types.Object, data_path: str, bone: str | None) -> tuple[Any, str]:
    if bone:
        return get_bone(ob, bone), data_path
    return (ob.data, data_path[5:]) if data_path.startswith("data.") and ob.data is not None else (ob, data_path)


def _set_path(owner: Any, path: str, index: int, value: Any) -> None:
    head, _, attr = path.rpartition(".")
    target = owner.path_resolve(head) if head else owner
    if attr in ROTATION_PATHS or attr.startswith("rotation_euler"):
        value = [math.radians(v) for v in value] if isinstance(value, list) else math.radians(value)
    try:
        if index >= 0:
            current = getattr(target, attr)
            current[index] = value
            setattr(target, attr, current)
        else:
            setattr(target, attr, value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ToolError(f"Can't set {path} = {value!r}: {exc}") from exc


@tool(
    obj(
        {
            "name": string("Object (or armature, with bone)."),
            "data_path": string("Property path: location, rotation_euler, scale, hide_render, data.energy, data.lens, '[\"prop\"]'..."),
            "frame": integer("Frame to key."),
            "value": scalar("Value to set before keying (vectors as lists; rotation in degrees)."),
            "index": integer("Array index (0=X), or -1 for all.", minimum=-1, maximum=3, default=-1),
            "bone": string("Pose bone to key instead of the object."),
            "interpolation": string("Interpolation for the new keys.", enum=["CONSTANT", "LINEAR", "BEZIER"]),
        },
        ["name", "data_path", "frame"],
    ),
    group="animation",
)
def insert_keyframe(
    name: str, data_path: str, frame: int, value: Any = None, index: int = -1, bone: str | None = None, interpolation: str | None = None
) -> str:
    """Set a property (optionally) and insert a keyframe for it at a frame."""
    ob = get_object(name)
    owner, path = _owner(ob, data_path, bone)
    if value is not None:
        _set_path(owner, path, index, value)
    try:
        owner.keyframe_insert(path, index=index, frame=frame)
    except (TypeError, RuntimeError) as exc:
        raise ToolError(f"Can't key {data_path}: {exc}") from exc
    if interpolation:
        full = f'pose.bones["{bone}"].{path}' if bone else path
        id_data = ob.data if owner is ob.data else ob
        for fc in fcurves_of(id_data):
            if fc.data_path == full:
                for kp in fc.keyframe_points:
                    if abs(kp.co.x - frame) < 0.5:
                        kp.interpolation = interpolation
    return f"Keyed {name}{'.' + bone if bone else ''} {data_path} at frame {frame}"


def _apply_rotation(pb: bpy.types.PoseBone, quat: Quaternion) -> str:
    if pb.rotation_mode == "QUATERNION":
        pb.rotation_quaternion = quat
        return "rotation_quaternion"
    if pb.rotation_mode == "AXIS_ANGLE":
        axis, angle = quat.to_axis_angle()
        pb.rotation_axis_angle = (angle, *axis)
        return "rotation_axis_angle"
    pb.rotation_euler = quat.to_euler(pb.rotation_mode)
    return "rotation_euler"


@tool(
    obj(
        {
            "armature": string("Armature object."),
            "bone": string("Pose bone."),
            "rotation_deg": vec3("Rotation in degrees about the bone's local X, Y, Z (applied in that order)."),
            "location": vec3("Offset from the rest position, in the bone's local space."),
            "scale": vec3("Pose scale."),
            "frame": integer("Insert keyframes for the changed channels at this frame."),
            "reset": boolean("Clear the bone back to its rest pose first.", default=False),
        },
        ["armature", "bone"],
    ),
    group="animation",
)
def pose_bone(
    armature: str,
    bone: str,
    rotation_deg: list[float] | None = None,
    location: list[float] | None = None,
    scale: list[float] | None = None,
    frame: int | None = None,
    reset: bool = False,
) -> str:
    """Pose a bone (works with Euler or quaternion bones) and optionally keyframe it."""
    ob = get_armature(armature)
    pb = get_bone(ob, bone)
    keyed = []
    if reset:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)
    if rotation_deg is not None:
        quat = Quaternion()
        for axis, angle in zip("XYZ", rotation_deg, strict=True):
            vec = Vector([1.0 if a == axis else 0.0 for a in "XYZ"])
            quat = quat @ Quaternion(vec, math.radians(angle))
        keyed.append(_apply_rotation(pb, quat))
    if location is not None:
        pb.location = location
        keyed.append("location")
    if scale is not None:
        pb.scale = scale
        keyed.append("scale")
    if frame is not None:
        if not keyed:
            rotation = {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}
            keyed = ["location", rotation.get(pb.rotation_mode, "rotation_euler"), "scale"]
        for path in keyed:
            pb.keyframe_insert(path, frame=frame)
    return f"Posed {bone}" + (f", keyed {', '.join(keyed)} at {frame}" if frame is not None else "")


@tool(
    obj(
        {
            "name": string("Animated object."),
            "interpolation": string("Interpolation.", enum=["CONSTANT", "LINEAR", "BEZIER", "SINE", "QUAD", "CUBIC", "EXPO", "BACK", "BOUNCE", "ELASTIC"]),
            "easing": string("Easing for the non-Bezier curves.", enum=["AUTO", "EASE_IN", "EASE_OUT", "EASE_IN_OUT"], default="AUTO"),
            "data_path_contains": string("Only F-curves whose path contains this text (e.g. 'location' or a bone name)."),
        },
        ["name", "interpolation"],
    ),
    group="animation",
)
def set_interpolation(name: str, interpolation: str, easing: str = "AUTO", data_path_contains: str | None = None) -> str:
    """Change keyframe interpolation (e.g. LINEAR for constant-speed turntables, BOUNCE for drops)."""
    ob = get_object(name)
    count = 0
    for fc in fcurves_of(ob):
        if data_path_contains and data_path_contains not in fc.data_path:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation
            kp.easing = easing
            count += 1
    if not count:
        raise ToolError(f"{name!r} has no matching keyframes.")
    return f"Set {count} keyframes to {interpolation}"


FCURVE_MOD = obj(
    {
        "type": string("Modifier type.", enum=["CYCLES", "NOISE", "GENERATOR", "STEPPED", "LIMITS"]),
        "props": free_object('Modifier settings, e.g. NOISE {"scale": 10, "strength": 0.5}.'),
    },
    ["type"],
)


@tool(
    obj(
        {
            "name": string("Animated object (or armature with bone)."),
            "data_path": string("F-curve path, e.g. location, rotation_euler, rotation_quaternion."),
            "index": integer("Array index.", minimum=0, maximum=3, default=0),
            "bone": string("Pose bone the curve belongs to."),
            "keyframes": array(
                obj({"frame": number("Frame."), "value": number("Value (radians for rotation curves).")}, ["frame", "value"]),
                "Keyframes to insert or replace on this curve.",
            ),
            "extrapolation": string("Before/after the keys.", enum=["CONSTANT", "LINEAR"]),
            "add_modifier": FCURVE_MOD,
            "clear_modifiers": boolean("Remove existing F-curve modifiers first.", default=False),
        },
        ["name", "data_path"],
    ),
    group="animation",
)
def edit_fcurve(
    name: str,
    data_path: str,
    index: int = 0,
    bone: str | None = None,
    keyframes: list[dict[str, float]] | None = None,
    extrapolation: str | None = None,
    add_modifier: dict[str, Any] | None = None,
    clear_modifiers: bool = False,
) -> dict[str, Any]:
    """Edit one F-curve directly: set keyframe values, extrapolation, or add modifiers such as CYCLES
    (loop forever) or NOISE (camera shake, flicker)."""
    ob = get_object(name)
    path = f'pose.bones["{bone}"].{data_path}' if bone else data_path
    if bone:
        get_bone(ob, bone)
    fc = next((f for f in fcurves_of(ob) if f.data_path == path and f.array_index == index), None)
    if fc is None:
        if not keyframes:
            raise ToolError(f"No F-curve {path}[{index}] on {name!r}; give keyframes to create one.")
        fc = ensure_fcurve(ob, path, index)
    for kf in keyframes or []:
        fc.keyframe_points.insert(kf["frame"], kf["value"], options={"REPLACE"})
    if extrapolation:
        fc.extrapolation = extrapolation
    if clear_modifiers:
        for mod in list(fc.modifiers):
            fc.modifiers.remove(mod)
    if add_modifier:
        mod = fc.modifiers.new(add_modifier["type"])
        for key, value in (add_modifier.get("props") or {}).items():
            if not hasattr(mod, key):
                raise ToolError(f"F-curve modifier {add_modifier['type']} has no setting {key!r}.")
            setattr(mod, key, value)
    fc.update()
    return {
        "curve": f"{path}[{index}]",
        "keys": [[round(k.co.x, 2), round(k.co.y, 4)] for k in fc.keyframe_points][:60],
        "modifiers": [m.type for m in fc.modifiers],
    }


@tool(
    obj(
        {
            "start": integer("First frame.", minimum=0),
            "end": integer("Last frame.", minimum=0),
            "fps": integer("Frames per second.", minimum=1, maximum=240),
            "current": integer("Jump to this frame."),
        }
    ),
    group="animation",
)
def set_frame_range(start: int | None = None, end: int | None = None, fps: int | None = None, current: int | None = None) -> dict[str, Any]:
    """Set the scene frame range, frame rate and current frame."""
    sc = scene()
    if start is not None:
        sc.frame_start = start
    if end is not None:
        sc.frame_end = end
    if sc.frame_end < sc.frame_start:
        raise ToolError("end must be >= start.")
    if fps is not None:
        sc.render.fps, sc.render.fps_base = fps, 1.0
    if current is not None:
        sc.frame_set(current)
    return {"frames": [sc.frame_start, sc.frame_end], "fps": sc.render.fps, "current": sc.frame_current}


@tool(
    obj(
        {
            "name": string("Object that gets the action."),
            "action_name": string("Name for the new action."),
            "fake_user": boolean("Keep the action even when unassigned.", default=True),
        },
        ["name", "action_name"],
    ),
    group="animation",
)
def create_action(name: str, action_name: str, fake_user: bool = True) -> str:
    """Create a new empty action and make it the object's active action (start a new clip)."""
    ob = get_object(name)
    act = bpy.data.actions.new(action_name)
    act.use_fake_user = fake_user
    ad = ob.animation_data or ob.animation_data_create()
    ad.action = act
    return f"{ob.name} now animates into action {act.name!r}"


@tool(obj({"name": string("Object whose active action becomes an NLA strip.")}, ["name"]), group="animation")
def nla_push_down(name: str) -> str:
    """Move the object's active action into a new NLA track, freeing it to record a new action on top."""
    ob = get_object(name)
    ad = ob.animation_data
    if ad is None or ad.action is None:
        raise ToolError(f"{name!r} has no active action.")
    act = ad.action
    track = ad.nla_tracks.new()
    track.name = act.name
    strip = track.strips.new(act.name, int(act.frame_range[0]), act)
    ad.action = None
    return f"Pushed {act.name!r} to NLA track {track.name!r} at frame {strip.frame_start:g}"


@tool(
    obj(
        {
            "name": string("Object to add the strip to."),
            "action": string("Action to place."),
            "start_frame": integer("Where the strip starts.", default=1),
            "track": string("Track name (created if missing; default: a new track)."),
            "repeat": number("How many times the action repeats.", minimum=0.1, maximum=1000, default=1.0),
            "scale": number("Playback time scale (2 = half speed).", minimum=0.01, maximum=100, default=1.0),
            "blend_type": string("How it combines with tracks below.", enum=["REPLACE", "COMBINE", "ADD", "SUBTRACT", "MULTIPLY"], default="REPLACE"),
            "extrapolation": string("What happens after the strip ends.", enum=["HOLD", "HOLD_FORWARD", "NOTHING"], default="HOLD_FORWARD"),
        },
        ["name", "action"],
    ),
    group="animation",
)
def nla_add_strip(
    name: str,
    action: str,
    start_frame: int = 1,
    track: str | None = None,
    repeat: float = 1.0,
    scale: float = 1.0,
    blend_type: str = "REPLACE",
    extrapolation: str = "HOLD_FORWARD",
) -> dict[str, Any]:
    """Place an action as an NLA strip, e.g. repeat a walk cycle along a timeline or layer clips."""
    ob = get_object(name)
    act = get_action(action)
    ad = ob.animation_data or ob.animation_data_create()
    nla_track = ad.nla_tracks.get(track) if track else None
    if nla_track is None:
        nla_track = ad.nla_tracks.new()
        if track:
            nla_track.name = track
    try:
        strip = nla_track.strips.new(act.name, start_frame, act)
    except RuntimeError as exc:
        raise ToolError(f"Can't place the strip there (overlaps another strip?): {exc}") from exc
    strip.repeat = repeat
    strip.scale = scale
    strip.blend_type = blend_type
    strip.extrapolation = extrapolation
    return {"track": nla_track.name, "strip": strip.name, "frames": [strip.frame_start, strip.frame_end]}


_BONE_PATTERNS: dict[str, tuple[str, ...]] = {
    "thigh": ("thigh", "upperleg", "upleg"),
    "shin": ("shin", "calf", "lowerleg", "knee", "leftleg", "rightleg"),
    "upper_arm": ("upperarm", "leftarm", "rightarm"),
    "forearm": ("forearm", "lowerarm"),
}
_BONE_AVOID = ("twist", "mch", "org", "def", "parent", "tweak", "ik", "pole", "target")


def _flat(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _find_bones(ob: bpy.types.Object) -> dict[str, str]:
    from .rigging import _side

    found: dict[str, str] = {}
    names = [pb.name for pb in ob.pose.bones]
    for role, fragments in _BONE_PATTERNS.items():
        for side in ("L", "R"):
            candidates = [n for n in names if any(f in _flat(n) for f in fragments) and not any(a in _flat(n) for a in _BONE_AVOID) and _side(n) == side]
            fk = [n for n in candidates if "fk" in _flat(n)]
            pick = fk or sorted(candidates, key=len)
            if pick:
                found[f"{role}.{side}"] = pick[0]
    hips = next((n for n in names if any(k in _flat(n) for k in ("hips", "pelvis", "torso")) and not _side(n)), None)
    if hips is None:
        hips = next((pb.name for pb in ob.pose.bones if pb.parent is None and pb.children), None)
    if hips:
        found["hips"] = hips
    return found


_CYCLES = {
    "walk": {"frames": 32, "leg": 25.0, "knee": 50.0, "arm": 18.0, "elbow": 15.0, "bob": 0.03},
    "run": {"frames": 18, "leg": 40.0, "knee": 95.0, "arm": 35.0, "elbow": 75.0, "bob": 0.06},
    "idle": {"frames": 72, "leg": 0.0, "knee": 2.0, "arm": 3.0, "elbow": 6.0, "bob": 0.006},
}


@tool(
    obj(
        {
            "armature": string("Character rig."),
            "cycle": string("Which loop.", enum=list(_CYCLES)),
            "frames": integer("Cycle length in frames.", minimum=4, maximum=600),
            "start_frame": integer("First frame of the cycle.", default=1),
            "intensity": number("Scale every swing and bounce (1 = normal).", minimum=0.0, maximum=3.0, default=1.0),
            "bone_map": obj(
                {k: string() for k in ("hips", "thigh.L", "thigh.R", "shin.L", "shin.R", "upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R")},
                description="Bone names per role, when automatic detection picks the wrong bones.",
            ),
            "action_name": string("Name of the new action."),
            "set_frame_range": boolean("Fit the scene frame range to the cycle.", default=True),
        },
        ["armature", "cycle"],
    ),
    group="animation",
)
def build_cycle(
    armature: str,
    cycle: str,
    frames: int | None = None,
    start_frame: int = 1,
    intensity: float = 1.0,
    bone_map: dict[str, str] | None = None,
    action_name: str | None = None,
    set_frame_range: bool = True,
) -> dict[str, Any]:
    """Keyframe a looping walk, run or idle cycle on a humanoid rig (legs swing about the character's side
    axis, knees and elbows bend, arms counter-swing, hips bob). Bones are found by name (Rigify, MPFB,
    Mixamo, game rigs); pass bone_map to override. Loops with a Cycles modifier; push to NLA to repeat."""
    ob = get_armature(armature)
    p = dict(_CYCLES[cycle])
    length = frames or int(p["frames"])
    bones = {**_find_bones(ob), **(bone_map or {})}
    for bone_name in bones.values():
        get_bone(ob, bone_name)
    if not any(k.startswith("thigh") for k in bones) and cycle != "idle":
        raise ToolError(f"Couldn't find leg bones in {ob.name!r}; pass bone_map. Found: {bones}")

    act = bpy.data.actions.new(action_name or f"{ob.name}_{cycle}")
    act.use_fake_user = True
    ad = ob.animation_data or ob.animation_data_create()
    ad.action = act
    _use_fk(ob)

    def side_axis(bone_name: str) -> Vector:
        rest = ob.data.bones[bone_name].matrix_local.to_3x3()
        return (rest.inverted() @ Vector((1.0, 0.0, 0.0))).normalized()

    def up_axis(bone_name: str) -> Vector:
        rest = ob.data.bones[bone_name].matrix_local.to_3x3()
        return (rest.inverted() @ Vector((0.0, 0.0, 1.0))).normalized()

    k = intensity
    samples = 8
    for i in range(samples + 1):
        t = i / samples
        frame = start_frame + t * length
        c, s = math.cos(2 * math.pi * t), math.sin(2 * math.pi * t)
        angles = {
            "thigh.L": -p["leg"] * c,
            "thigh.R": p["leg"] * c,
            "shin.L": p["knee"] * max(0.0, -s) + 5.0,
            "shin.R": p["knee"] * max(0.0, s) + 5.0,
            "upper_arm.L": p["arm"] * c,
            "upper_arm.R": -p["arm"] * c,
            "forearm.L": -p["elbow"] * (1.0 + 0.3 * c),
            "forearm.R": -p["elbow"] * (1.0 - 0.3 * c),
        }
        if cycle == "idle":
            breath = math.sin(2 * math.pi * t)
            angles = {
                "upper_arm.L": p["arm"] * breath,
                "upper_arm.R": p["arm"] * breath,
                "forearm.L": -p["elbow"],
                "forearm.R": -p["elbow"],
                "shin.L": p["knee"],
                "shin.R": p["knee"],
            }
        for role, angle in angles.items():
            bone_name = bones.get(role)
            if not bone_name:
                continue
            pb = ob.pose.bones[bone_name]
            path = _apply_rotation(pb, Quaternion(side_axis(bone_name), math.radians(angle * k)))
            pb.keyframe_insert(path, frame=frame)
        if "hips" in bones:
            pb = ob.pose.bones[bones["hips"]]
            bounce = -math.cos(4 * math.pi * t) if cycle != "idle" else math.sin(2 * math.pi * t)
            pb.location = up_axis(bones["hips"]) * (p["bob"] * k * bounce)
            pb.keyframe_insert("location", frame=frame)

    for fc in fcurves_of(ob):
        fc.modifiers.new("CYCLES")
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
    if set_frame_range:
        sc = scene()
        sc.frame_start, sc.frame_end = start_frame, start_frame + length - 1
    return {"action": act.name, "frames": [start_frame, start_frame + length], "bones": bones}


def _use_fk(ob: bpy.types.Object) -> None:
    for pb in ob.pose.bones:
        if "IK_FK" in pb and pb.name.endswith(("_parent.L", "_parent.R")):
            pb["IK_FK"] = 1.0


def _track(cam: bpy.types.Object, target: bpy.types.Object | None) -> None:
    if target is None:
        return
    con = next((c for c in cam.constraints if c.type == "TRACK_TO"), None) or cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"


def _linear(id_data: Any) -> None:
    for fc in fcurves_of(id_data):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


@tool(
    obj(
        {
            "camera": string("Camera object."),
            "target": string("Object to orbit around."),
            "point": vec3("Point to orbit around, instead of target."),
            "radius": number("Orbit radius in meters (default: keep the current distance).", minimum=0.01),
            "height": number("Camera height above the pivot in meters (default: keep current)."),
            "frames": integer("Frames for the orbit.", minimum=2, default=120),
            "start_frame": integer("First frame.", default=1),
            "turns": number("Number of full turns (negative = clockwise).", default=1.0),
        },
        ["camera"],
    ),
    group="animation",
)
def camera_orbit(
    camera: str,
    target: str | None = None,
    point: list[float] | None = None,
    radius: float | None = None,
    height: float | None = None,
    frames: int = 120,
    start_frame: int = 1,
    turns: float = 1.0,
) -> dict[str, Any]:
    """Animate a camera orbiting a subject at constant speed (a pivot empty rotates; the camera tracks it)."""
    cam = get_object(camera, ("CAMERA",))
    subject = get_object(target) if target else None
    center = subject.matrix_world.translation.copy() if subject else Vector(point or (0, 0, 0))
    pivot = bpy.data.objects.new(unique_name(f"{cam.name}_Orbit", bpy.data.objects), None)
    pivot.empty_display_type = "SPHERE"
    pivot.empty_display_size = 0.2
    link(pivot)
    pivot.location = center
    offset = cam.matrix_world.translation - center
    flat = Vector((offset.x, offset.y, 0.0))
    r = radius if radius is not None else max(flat.length, 0.5)
    z = height if height is not None else offset.z
    direction = flat.normalized() if flat.length > 1e-6 else Vector((0, -1, 0))
    for con in [c for c in cam.constraints if c.type == "CHILD_OF"]:
        cam.constraints.remove(con)
    cam.parent = pivot
    cam.matrix_parent_inverse.identity()
    cam.location = direction * r + Vector((0, 0, z))
    _track(cam, subject or pivot)
    pivot.rotation_euler = (0, 0, 0)
    pivot.keyframe_insert("rotation_euler", index=2, frame=start_frame)
    pivot.rotation_euler.z = 2 * math.pi * turns
    pivot.keyframe_insert("rotation_euler", index=2, frame=start_frame + frames)
    _linear(pivot)
    return {"pivot": pivot.name, "radius": round(r, 3), "height": round(z, 3), "frames": [start_frame, start_frame + frames]}


@tool(
    obj(
        {
            "camera": string("Camera object."),
            "start": vec3("Start location (default: current)."),
            "end": vec3("End location."),
            "frames": integer("Duration in frames.", minimum=2, default=96),
            "start_frame": integer("First frame.", default=1),
            "ease": boolean("Ease in and out (otherwise constant speed).", default=True),
            "target": string("Keep the camera aimed at this object."),
        },
        ["camera", "end"],
    ),
    group="animation",
)
def camera_dolly(
    camera: str, end: list[float], start: list[float] | None = None, frames: int = 96, start_frame: int = 1, ease: bool = True, target: str | None = None
) -> str:
    """Animate a camera moving in a straight line (dolly/truck/crane), optionally aimed at a subject."""
    cam = get_object(camera, ("CAMERA",))
    cam.location = start or cam.location
    cam.keyframe_insert("location", frame=start_frame)
    cam.location = end
    cam.keyframe_insert("location", frame=start_frame + frames)
    if not ease:
        _linear(cam)
    _track(cam, get_object(target) if target else None)
    return f"Dolly on {cam.name} over frames {start_frame}-{start_frame + frames}"


@tool(
    obj(
        {
            "camera": string("Camera object."),
            "points": array(vec3(), "Path points in world space.", min_items=2),
            "frames": integer("Duration in frames.", minimum=2, default=120),
            "start_frame": integer("First frame.", default=1),
            "target": string("Keep the camera aimed at this object (otherwise it faces along the path)."),
            "closed": boolean("Close the path into a loop.", default=False),
        },
        ["camera", "points"],
    ),
    group="animation",
)
def camera_path(
    camera: str, points: list[list[float]], frames: int = 120, start_frame: int = 1, target: str | None = None, closed: bool = False
) -> dict[str, Any]:
    """Fly a camera along a smooth curve through the given points."""
    cam = get_object(camera, ("CAMERA",))
    curve = bpy.data.curves.new(f"{cam.name}_Path", "CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, p in zip(spline.bezier_points, points, strict=True):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    spline.use_cyclic_u = closed
    path_ob = bpy.data.objects.new(unique_name(f"{cam.name}_Path", bpy.data.objects), curve)
    link(path_ob)
    cam.parent = None
    cam.location = (0, 0, 0)
    con = cam.constraints.new("FOLLOW_PATH")
    con.target = path_ob
    con.use_fixed_location = True
    con.use_curve_follow = target is None
    if target is None:
        con.forward_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    con.offset_factor = 0.0
    con.keyframe_insert("offset_factor", frame=start_frame)
    con.offset_factor = 1.0
    con.keyframe_insert("offset_factor", frame=start_frame + frames)
    _linear(cam)
    _track(cam, get_object(target) if target else None)
    return {"path": path_ob.name, "frames": [start_frame, start_frame + frames]}


@tool(
    obj(
        {
            "name": string("Object or armature to bake."),
            "start": integer("First frame (default: scene start)."),
            "end": integer("Last frame (default: scene end)."),
            "visual_keying": boolean("Bake the visual result of constraints/drivers.", default=True),
            "clear_constraints": boolean("Remove constraints after baking.", default=False),
        },
        ["name"],
    ),
    group="animation",
    destructive=True,
)
def bake_action(name: str, start: int | None = None, end: int | None = None, visual_keying: bool = True, clear_constraints: bool = False) -> str:
    """Bake animation (constraints, IK, drivers, NLA) into plain keyframes on every frame of a range."""
    ob = get_object(name)
    sc = scene()
    first, last = start if start is not None else sc.frame_start, end if end is not None else sc.frame_end
    types = {"POSE"} if ob.type == "ARMATURE" else {"OBJECT"}
    with op_context(ob, [ob]):
        if ob.type == "ARMATURE":
            bpy.ops.object.mode_set(mode="POSE")
            bpy.ops.pose.select_all(action="SELECT")
        bpy.ops.nla.bake(
            frame_start=first, frame_end=last, only_selected=False, visual_keying=visual_keying, clear_constraints=clear_constraints, bake_types=types
        )
        if ob.type == "ARMATURE":
            bpy.ops.object.mode_set(mode="OBJECT")
    action = ob.animation_data.action.name if ob.animation_data and ob.animation_data.action else "?"
    return f"Baked {ob.name} frames {first}-{last} into {action!r}"
