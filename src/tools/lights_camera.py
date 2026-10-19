from __future__ import annotations

import math
from typing import Any

import bpy
from mathutils import Vector

from .common import describe, euler_deg, get_object, link, scene, set_rna, sync, unique_name
from .registry import ToolError, boolean, color, free_object, number, obj, string, tool, vec3


def _aim(ob: bpy.types.Object, target: str | None, point: list[float] | None) -> None:
    sync()
    goal = None
    if target:
        goal = get_object(target).matrix_world.translation
    elif point is not None:
        goal = Vector(point)
    if goal is not None:
        direction = goal - ob.matrix_world.translation
        if direction.length > 1e-6:
            ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


@tool(
    obj(
        {
            "type": string("Light type.", enum=["POINT", "SUN", "SPOT", "AREA"]),
            "name": string("Object name."),
            "location": vec3("Location in meters."),
            "rotation_deg": vec3("Rotation in degrees (lights shine down -Z)."),
            "target": string("Aim the light at this object."),
            "point_at": vec3("Aim the light at this point."),
            "energy": number("Power in watts (point/spot/area) or irradiance in W/m² (sun, ~1-10).", minimum=0),
            "color": color("Light color."),
            "size": number("Soft-shadow size: radius (point/spot), angle in degrees (sun) or edge length (area).", minimum=0),
            "spot_size_deg": number("Spot cone angle.", minimum=1, maximum=180),
            "spot_blend": number("Spot edge softness.", minimum=0, maximum=1),
            "area_shape": string("Area light shape.", enum=["SQUARE", "RECTANGLE", "DISK", "ELLIPSE"]),
            "size_y": number("Second size for rectangle/ellipse area lights.", minimum=0),
        },
        ["type"],
    ),
    group="lights_camera",
)
def add_light(
    type: str,
    name: str | None = None,
    location: list[float] | None = None,
    rotation_deg: list[float] | None = None,
    target: str | None = None,
    point_at: list[float] | None = None,
    energy: float | None = None,
    color: list[float] | None = None,
    size: float | None = None,
    spot_size_deg: float | None = None,
    spot_blend: float | None = None,
    area_shape: str | None = None,
    size_y: float | None = None,
) -> dict[str, Any]:
    """Add a light. Typical: key area light 200-1000 W at 2-4 m, sun 3-5 for daylight. Aim with target."""
    label = name or f"{type.title()} Light"
    data = bpy.data.lights.new(label, type)
    defaults = {"POINT": 500.0, "SPOT": 800.0, "AREA": 300.0, "SUN": 4.0}
    data.energy = energy if energy is not None else defaults[type]
    if color is not None:
        data.color = color
    if size is not None:
        if type == "SUN":
            data.angle = math.radians(size)
        elif type == "AREA":
            data.size = size
        else:
            data.shadow_soft_size = size
    if type == "SPOT":
        if spot_size_deg is not None:
            data.spot_size = math.radians(spot_size_deg)
        if spot_blend is not None:
            data.spot_blend = spot_blend
    if type == "AREA":
        if area_shape:
            data.shape = area_shape
        if size_y is not None:
            data.size_y = size_y
    ob = bpy.data.objects.new(unique_name(label, bpy.data.objects), data)
    link(ob)
    ob.location = location or (0.0, 0.0, 3.0)
    if rotation_deg is not None:
        ob.rotation_euler = euler_deg(rotation_deg)
    _aim(ob, target, point_at)
    return describe(ob)


@tool(
    obj(
        {"name": string("Light object."), "props": free_object('Light data properties, e.g. {"energy": 600, "color": [1, 0.9, 0.8], "use_shadow": true}.')},
        ["name", "props"],
    ),
    group="lights_camera",
)
def set_light(name: str, props: dict[str, Any]) -> dict[str, Any]:
    """Change a light's properties (energy, color, size, spot angle...). Angles are in degrees."""
    ob = get_object(name, ("LIGHT",))
    set_rna(ob.data, props, f"Light {ob.name!r}")
    return describe(ob)


@tool(
    obj(
        {
            "name": string("Camera object name."),
            "location": vec3("Location in meters."),
            "rotation_deg": vec3("Rotation in degrees."),
            "target": string("Aim the camera at this object."),
            "point_at": vec3("Aim the camera at this point."),
            "lens_mm": number("Focal length.", minimum=1, maximum=5000, default=50),
            "make_active": boolean("Make it the scene camera.", default=True),
            "track": boolean("Keep pointing at the target with a Track To constraint.", default=False),
        }
    ),
    group="lights_camera",
)
def add_camera(
    name: str | None = None,
    location: list[float] | None = None,
    rotation_deg: list[float] | None = None,
    target: str | None = None,
    point_at: list[float] | None = None,
    lens_mm: float = 50.0,
    make_active: bool = True,
    track: bool = False,
) -> dict[str, Any]:
    """Add a camera, aim it, and make it the active scene camera."""
    label = name or "Camera"
    data = bpy.data.cameras.new(label)
    data.lens = lens_mm
    ob = bpy.data.objects.new(unique_name(label, bpy.data.objects), data)
    link(ob)
    ob.location = location or (7.0, -7.0, 5.0)
    if rotation_deg is not None:
        ob.rotation_euler = euler_deg(rotation_deg)
    _aim(ob, target, point_at)
    if track:
        if not target:
            raise ToolError("track needs a target object.")
        con = ob.constraints.new("TRACK_TO")
        con.target = get_object(target)
        con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    if make_active:
        scene().camera = ob
    return describe(ob)


@tool(
    obj(
        {
            "name": string("Camera object."),
            "lens_mm": number("Focal length.", minimum=1, maximum=5000),
            "make_active": boolean("Make it the scene camera."),
            "focus_object": string("Enable depth of field focused on this object."),
            "fstop": number("Aperture for depth of field (lower = blurrier).", minimum=0.1, maximum=128),
            "props": free_object('Other camera data properties, e.g. {"type": "ORTHO", "ortho_scale": 6, "clip_end": 500}.'),
        },
        ["name"],
    ),
    group="lights_camera",
)
def set_camera(
    name: str,
    lens_mm: float | None = None,
    make_active: bool | None = None,
    focus_object: str | None = None,
    fstop: float | None = None,
    props: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Change camera lens, depth of field and other settings, or make it the active camera."""
    ob = get_object(name, ("CAMERA",))
    cam = ob.data
    if lens_mm is not None:
        cam.lens = lens_mm
    if focus_object:
        cam.dof.use_dof = True
        cam.dof.focus_object = get_object(focus_object)
    if fstop is not None:
        cam.dof.use_dof = True
        cam.dof.aperture_fstop = fstop
    if props:
        set_rna(cam, props, f"Camera {ob.name!r}")
    if make_active:
        scene().camera = ob
    return describe(ob)
