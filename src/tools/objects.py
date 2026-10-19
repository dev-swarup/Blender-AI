from __future__ import annotations

import math
from typing import Any

import bmesh
import bpy
from mathutils import Matrix, Vector

from .common import describe, euler_deg, get_object, get_objects, link, op_context, r, set_rna, sync, unique_name
from .registry import ToolError, array, boolean, free_object, integer, names, number, obj, string, tool, vec3

PRIMITIVES = ["cube", "uv_sphere", "ico_sphere", "cylinder", "cone", "torus", "plane", "grid", "circle", "monkey", "empty", "text", "bezier_curve"]


@tool(
    obj(
        {
            "kind": string("Primitive type.", enum=PRIMITIVES),
            "name": string("Object name."),
            "location": vec3("World location in meters."),
            "rotation_deg": vec3("Rotation in degrees (XYZ Euler)."),
            "scale": vec3("Object scale."),
            "size": number("Overall size in meters: edge length for cube/plane/grid, diameter for round shapes.", minimum=0.0001, default=1.0),
            "depth": number("Height for cylinder/cone, in meters (defaults to size).", minimum=0.0001),
            "radius_top": number("Top radius for cone (0 makes a point).", minimum=0.0),
            "minor_size": number("Tube diameter for torus (default 0.25 x size).", minimum=0.0001),
            "segments": integer("Around-segments for round shapes, or X subdivisions for grid.", minimum=3, maximum=512),
            "rings": integer("Rings for uv_sphere, tube segments for torus, Y subdivisions for grid.", minimum=1, maximum=512),
            "subdivisions": integer("Subdivisions for ico_sphere.", minimum=1, maximum=8),
            "fill": boolean("Fill the circle with a face.", default=True),
            "text": string("Body text for kind=text."),
            "extrude": number("Extrusion depth for text, in meters.", minimum=0.0),
            "points": array(vec3(), "Control points for bezier_curve (world space).", min_items=2),
            "smooth": boolean("Shade smooth.", default=False),
            "collection": string("Collection to put it in, created if missing (default: active collection)."),
        },
        ["kind"],
    ),
    group="objects",
)
def create_primitive(
    kind: str,
    name: str | None = None,
    location: list[float] | None = None,
    rotation_deg: list[float] | None = None,
    scale: list[float] | None = None,
    size: float = 1.0,
    depth: float | None = None,
    radius_top: float | None = None,
    minor_size: float | None = None,
    segments: int | None = None,
    rings: int | None = None,
    subdivisions: int | None = None,
    fill: bool = True,
    text: str | None = None,
    extrude: float | None = None,
    points: list[list[float]] | None = None,
    smooth: bool = False,
    collection: str | None = None,
) -> dict[str, Any]:
    """Create a primitive object built directly from mesh data (with UVs), an empty, a text object or a
    bezier curve. Sizes are in meters; the origin is at the shape's center."""
    label = name or kind.replace("_", " ").title().replace(" ", "")
    if kind == "empty":
        ob = bpy.data.objects.new(unique_name(label, bpy.data.objects), None)
        ob.empty_display_type = "PLAIN_AXES"
        ob.empty_display_size = size / 2
    elif kind == "text":
        curve = bpy.data.curves.new(label, "FONT")
        curve.body = text or "Text"
        curve.size = size
        curve.extrude = extrude or 0.0
        curve.align_x = "CENTER"
        ob = bpy.data.objects.new(unique_name(label, bpy.data.objects), curve)
    elif kind == "bezier_curve":
        pts = points or [[-size / 2, 0, 0], [size / 2, 0, 0]]
        curve = bpy.data.curves.new(label, "CURVE")
        curve.dimensions = "3D"
        spline = curve.splines.new("BEZIER")
        spline.bezier_points.add(len(pts) - 1)
        for bp, p in zip(spline.bezier_points, pts, strict=True):
            bp.co = p
            bp.handle_left_type = bp.handle_right_type = "AUTO"
        ob = bpy.data.objects.new(unique_name(label, bpy.data.objects), curve)
        location = None
    else:
        me = bpy.data.meshes.new(label)
        bm = bmesh.new()
        try:
            _build_mesh(bm, kind, size, depth, radius_top, minor_size, segments, rings, subdivisions, fill)
            bm.to_mesh(me)
        finally:
            bm.free()
        if smooth:
            for poly in me.polygons:
                poly.use_smooth = True
        ob = bpy.data.objects.new(unique_name(label, bpy.data.objects), me)

    link(ob, collection)
    if location is not None:
        ob.location = location
    if rotation_deg is not None:
        ob.rotation_euler = euler_deg(rotation_deg)
    if scale is not None:
        ob.scale = scale
    return describe(ob)


def _build_mesh(
    bm: bmesh.types.BMesh,
    kind: str,
    size: float,
    depth: float | None,
    radius_top: float | None,
    minor_size: float | None,
    segments: int | None,
    rings: int | None,
    subdivisions: int | None,
    fill: bool,
) -> None:
    uv = bm.loops.layers.uv.new("UVMap")
    radius = size / 2
    if kind == "cube":
        bmesh.ops.create_cube(bm, size=size, calc_uvs=True)
    elif kind == "uv_sphere":
        bmesh.ops.create_uvsphere(bm, u_segments=segments or 32, v_segments=rings or 16, radius=radius, calc_uvs=True)
    elif kind == "ico_sphere":
        bmesh.ops.create_icosphere(bm, subdivisions=subdivisions or 2, radius=radius, calc_uvs=True)
    elif kind in ("cylinder", "cone"):
        top = radius if kind == "cylinder" else (radius_top or 0.0)
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments or 32, radius1=radius, radius2=top, depth=depth or size, calc_uvs=True)
    elif kind in ("plane", "grid"):
        xs = 1 if kind == "plane" else (segments or 10)
        ys = 1 if kind == "plane" else (rings or segments or 10)
        bmesh.ops.create_grid(bm, x_segments=xs, y_segments=ys, size=size / 2, calc_uvs=True)
    elif kind == "circle":
        bmesh.ops.create_circle(bm, cap_ends=fill, segments=segments or 32, radius=radius, calc_uvs=True)
    elif kind == "monkey":
        bmesh.ops.create_monkey(bm, matrix=Matrix.Scale(radius, 4), calc_uvs=True)
    elif kind == "torus":
        _torus(bm, uv, radius, (minor_size or size * 0.25) / 2, segments or 48, rings or 12)
    else:
        raise ToolError(f"Unknown primitive {kind!r}.")


def _torus(bm: bmesh.types.BMesh, uv: Any, major: float, minor: float, seg: int, ring: int) -> None:
    grid = []
    for i in range(seg):
        u = 2 * math.pi * i / seg
        row = []
        for j in range(ring):
            v = 2 * math.pi * j / ring
            d = major + minor * math.cos(v)
            row.append(bm.verts.new((d * math.cos(u), d * math.sin(u), minor * math.sin(v))))
        grid.append(row)
    for i in range(seg):
        for j in range(ring):
            a, b = grid[i][j], grid[(i + 1) % seg][j]
            c, d = grid[(i + 1) % seg][(j + 1) % ring], grid[i][(j + 1) % ring]
            face = bm.faces.new((a, b, c, d))
            for loop, (du, dv) in zip(face.loops, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=True):
                loop[uv].uv = ((i + du) / seg, (j + dv) / ring)


@tool(
    obj(
        {
            "name": string("Object to transform."),
            "location": vec3("Location in meters."),
            "rotation_deg": vec3("Rotation in degrees (XYZ Euler)."),
            "scale": vec3("Scale factors."),
            "dimensions": vec3("Target bounding-box size in meters (sets scale)."),
            "relative": boolean("Add location/rotation and multiply scale instead of setting them.", default=False),
            "apply_scale": boolean("Bake scale into the mesh afterwards (do this before bevel, boolean, rigging).", default=False),
            "apply_rotation": boolean("Bake rotation into the mesh afterwards.", default=False),
        },
        ["name"],
    ),
    group="objects",
)
def transform_object(
    name: str,
    location: list[float] | None = None,
    rotation_deg: list[float] | None = None,
    scale: list[float] | None = None,
    dimensions: list[float] | None = None,
    relative: bool = False,
    apply_scale: bool = False,
    apply_rotation: bool = False,
) -> dict[str, Any]:
    """Move, rotate or scale an object, set its dimensions, and optionally apply scale/rotation."""
    ob = get_object(name)
    if location is not None:
        ob.location = (ob.location + Vector(location)) if relative else Vector(location)
    if rotation_deg is not None:
        rot = euler_deg(rotation_deg)
        if relative:
            ob.rotation_euler.rotate(rot)
        else:
            ob.rotation_euler = rot
    if scale is not None:
        ob.scale = [a * b for a, b in zip(ob.scale, scale, strict=True)] if relative else scale
    if dimensions is not None:
        ob.dimensions = dimensions
    if apply_scale or apply_rotation:
        bpy.context.view_layer.update()
        with op_context(ob, [ob]):
            bpy.ops.object.transform_apply(location=False, rotation=apply_rotation, scale=apply_scale)
    return describe(ob)


@tool(
    obj(
        {
            "name": string("Object to change."),
            "new_name": string("Rename the object (and its data)."),
            "shading": string("Surface shading.", enum=["smooth", "flat", "auto"]),
            "auto_smooth_deg": number("Angle for shading=auto.", minimum=0, maximum=180, default=30),
            "hide_viewport": boolean("Hide in the viewport."),
            "hide_render": boolean("Hide in renders."),
            "display_type": string("Viewport display.", enum=["BOUNDS", "WIRE", "SOLID", "TEXTURED"]),
            "props": free_object('Other Object RNA properties by name, e.g. {"show_wire": true}.'),
        },
        ["name"],
    ),
    group="objects",
)
def set_object_props(
    name: str,
    new_name: str | None = None,
    shading: str | None = None,
    auto_smooth_deg: float = 30.0,
    hide_viewport: bool | None = None,
    hide_render: bool | None = None,
    display_type: str | None = None,
    props: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rename an object, set smooth/flat/auto shading, visibility, display type or other object properties."""
    ob = get_object(name)
    if new_name:
        ob.name = new_name
        if ob.data is not None and ob.data.users == 1:
            ob.data.name = new_name
    if shading:
        if ob.type != "MESH":
            raise ToolError("Shading applies to mesh objects.")
        if shading == "auto" and hasattr(bpy.ops.object, "shade_auto_smooth"):
            with op_context(ob, [ob]):
                bpy.ops.object.shade_auto_smooth(angle=math.radians(auto_smooth_deg))
        elif shading == "auto" and hasattr(bpy.ops.object, "shade_smooth_by_angle"):
            with op_context(ob, [ob]):
                bpy.ops.object.shade_smooth_by_angle(angle=math.radians(auto_smooth_deg))
        else:
            for poly in ob.data.polygons:
                poly.use_smooth = shading != "flat"
            ob.data.update()
    if hide_viewport is not None:
        ob.hide_viewport = hide_viewport
        ob.hide_set(hide_viewport)
    if hide_render is not None:
        ob.hide_render = hide_render
    if display_type:
        ob.display_type = display_type
    if props:
        set_rna(ob, props, f"Object {ob.name!r}")
    return describe(ob)


@tool(
    obj({"names": names("Mesh objects to merge."), "into": string("The object that remains (default: the first name).")}, ["names"]),
    group="objects",
    destructive=True,
)
def join_objects(names: list[str], into: str | None = None) -> dict[str, Any]:
    """Join several mesh objects into one object."""
    objs = get_objects(names, ("MESH",))
    target = get_object(into, ("MESH",)) if into else objs[0]
    if target not in objs:
        objs.append(target)
    if len(objs) < 2:
        raise ToolError("Joining needs at least two mesh objects.")
    with op_context(target, objs):
        bpy.ops.object.join()
    return describe(target)


@tool(
    obj(
        {
            "name": string("Object to constrain."),
            "type": string("Constraint type, e.g. TRACK_TO, DAMPED_TRACK, COPY_LOCATION, COPY_ROTATION, CHILD_OF, FOLLOW_PATH, LIMIT_LOCATION, FLOOR."),
            "target": string("Target object."),
            "subtarget": string("Bone (or vertex group) on the target."),
            "props": free_object('Constraint properties by name, e.g. {"track_axis": "TRACK_NEGATIVE_Z", "up_axis": "UP_Y"}.'),
            "constraint_name": string("Name for the constraint."),
        },
        ["name", "type"],
    ),
    group="objects",
)
def add_object_constraint(
    name: str, type: str, target: str | None = None, subtarget: str | None = None, props: dict[str, Any] | None = None, constraint_name: str | None = None
) -> str:
    """Add a constraint to an object (for example make a camera track a subject with TRACK_TO)."""
    ob = get_object(name)
    valid = bpy.types.Constraint.bl_rna.properties["type"].enum_items.keys()
    if type not in valid:
        raise ToolError(f"Unknown constraint type {type!r}. Options: {', '.join(valid)}")
    con = ob.constraints.new(type)
    if constraint_name:
        con.name = constraint_name
    if target is not None and hasattr(con, "target"):
        con.target = get_object(target)
        if subtarget and hasattr(con, "subtarget"):
            con.subtarget = subtarget
    if type == "TRACK_TO" and not props:
        con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    if props:
        set_rna(con, props, f"Constraint {con.name!r}")
    return f"Added {con.type} constraint {con.name!r} to {ob.name}"


@tool(
    obj(
        {
            "name": string("Object to aim (cameras and lights look down their -Z axis)."),
            "target": string("Object to look at."),
            "point": vec3("World point to look at, instead of target."),
        },
        ["name"],
    ),
    group="objects",
)
def look_at(name: str, target: str | None = None, point: list[float] | None = None) -> dict[str, Any]:
    """Rotate an object so its -Z axis points at a target object or point (cameras, lights, spotlights)."""
    ob = get_object(name)
    sync()
    if target:
        goal = get_object(target).matrix_world.translation
    elif point is not None:
        goal = Vector(point)
    else:
        raise ToolError("Give a target object or a point.")
    direction = goal - ob.matrix_world.translation
    if direction.length < 1e-6:
        raise ToolError("The object is already at that point.")
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return {"name": ob.name, "rotation_deg": [round(math.degrees(a), 2) for a in ob.rotation_euler], "distance": r(direction.length, 3)}
