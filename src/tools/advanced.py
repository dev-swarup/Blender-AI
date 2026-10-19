from __future__ import annotations

import math
import os
import re
import urllib.parse
import urllib.request
import zipfile
from typing import Any

import bpy
from mathutils import Vector

from .common import get_object, get_objects, link, op_context, r, scene, sync, ui_context, unique_name, vec, world_bounds
from .registry import AsyncJob, ToolError, ToolKind, array, boolean, integer, names, number, obj, string, tool, vec3
from .scene import _IMPORTERS, import_asset

MAX_DOWNLOAD = 500 * 1024 * 1024
MODEL_EXTS = (".glb", ".gltf", ".fbx", ".obj", ".usdz", ".usd", ".usdc", ".usda", ".blend", ".stl", ".ply", ".abc")
_CONTENT_TYPES = {
    "model/gltf-binary": ".glb",
    "model/gltf+json": ".gltf",
    "model/obj": ".obj",
    "model/stl": ".stl",
    "model/vnd.usdz+zip": ".usdz",
    "application/zip": ".zip",
}


def _download_dir() -> str:
    from ..preferences import PACKAGE

    try:
        return bpy.utils.extension_path_user(PACKAGE, path="downloads", create=True)
    except (ValueError, AttributeError):
        path = os.path.join(bpy.utils.user_resource("DATAFILES", path="blender_ai", create=True), "downloads")
        os.makedirs(path, exist_ok=True)
        return path


def _filename(url: str, response: Any) -> str:
    disposition = response.headers.get("content-disposition") or ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disposition)
    name = urllib.parse.unquote(match.group(1)) if match else os.path.basename(urllib.parse.urlparse(url).path)
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._") or "model"
    if not os.path.splitext(name)[1]:
        ctype = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        name += _CONTENT_TYPES.get(ctype, ".glb")
    return name


def download_model(url: str, folder: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "blender-ai"})
    with urllib.request.urlopen(request, timeout=180) as response:
        name = _filename(url, response)
        path = os.path.join(folder, name)
        stem, ext = os.path.splitext(path)
        i = 1
        while os.path.exists(path):
            path, i = f"{stem}_{i}{ext}", i + 1
        size = 0
        with open(path, "wb") as f:
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise ToolError("The download is larger than 500 MB; stopped.")
                f.write(chunk)
    if path.lower().endswith(".zip"):
        target = os.path.splitext(path)[0]
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                dest = os.path.realpath(os.path.join(target, member))
                if not dest.startswith(os.path.realpath(target) + os.sep):
                    raise ToolError("The zip contains unsafe paths; not extracted.")
            archive.extractall(target)
        found = []
        for root, _dirs, files in os.walk(target):
            found += [os.path.join(root, f) for f in files if f.lower().endswith(MODEL_EXTS)]
        if not found:
            raise ToolError("The zip has no model file Blender can import.")
        found.sort(key=lambda p: MODEL_EXTS.index(os.path.splitext(p)[1].lower()))
        return found[0]
    return path


@tool(
    obj(
        {
            "url": string("http(s) link to a model file (.glb, .gltf, .fbx, .obj, .usdz, .stl, .ply) or a zip of one."),
            "name": string("Name for the imported object (the root, if there are several)."),
            "location": vec3("Where to put the model's bottom center. Default: the origin."),
            "target_size": number("Scale the model so its largest dimension is this many meters.", minimum=0.001),
            "collection": string("Put the imported objects in this collection (created if needed)."),
        },
        ["url"],
    ),
    group="scene",
    kind=ToolKind.ASYNC,
)
def import_from_url(
    url: str, name: str | None = None, location: list[float] | None = None, target_size: float | None = None, collection: str | None = None
) -> AsyncJob:
    """Download a 3D model from a URL and import it, standing on the ground at `location`. Use it for model
    links returned by MCP services (Meshy, Tripo...) or given by the user. Takes a few seconds to minutes."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ToolError("Only http(s) links can be downloaded.")
    folder = _download_dir()

    def finish(path: str) -> dict[str, Any]:
        ext = os.path.splitext(path)[1].lower()
        if ext != ".blend" and ext not in _IMPORTERS:
            raise ToolError(f"Downloaded {os.path.basename(path)}, but Blender can't import {ext} files.")
        info = import_asset(path, collection=collection)
        new = [bpy.data.objects[n] for n in info["imported"] if n in bpy.data.objects]
        if not new:
            raise ToolError("The file imported no objects.")
        roots = [ob for ob in new if ob.parent is None or ob.parent not in new]
        sync()
        lo, hi = world_bounds(new)
        pivot = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
        factor = 1.0
        if target_size:
            largest = max(hi - lo)
            factor = target_size / largest if largest > 1e-6 else 1.0
        target = vec(location)
        for ob in roots:
            ob.scale = ob.scale * factor
            ob.location = target + (ob.location - pivot) * factor
        if name and len(roots) == 1:
            roots[0].name = name
        sync()
        lo, hi = world_bounds(new)
        return {
            "file": path,
            "imported": [ob.name for ob in new][:60],
            "roots": [ob.name for ob in roots],
            "size": r(hi - lo, 3),
            "bounds": {"min": r(lo, 3), "max": r(hi, 3)},
        }

    return AsyncJob("downloading model", lambda: download_model(url, folder), finish)


def _socket(sockets: Any, name: str, kind: str | None = None) -> Any:
    for s in sockets:
        if s.name == name and (kind is None or s.type == kind):
            return s
    raise ToolError(f"Geometry node socket {name!r} not found in this Blender version.")


def _surface_area(ob: bpy.types.Object) -> float:
    mesh = ob.data
    scale = ob.matrix_world.to_3x3()
    area = 0.0
    for poly in mesh.polygons:
        verts = [scale @ mesh.vertices[i].co for i in poly.vertices]
        for i in range(1, len(verts) - 1):
            area += (verts[i] - verts[0]).cross(verts[i + 1] - verts[0]).length / 2
    return area


@tool(
    obj(
        {
            "target": string("Mesh object to scatter onto (ground, terrain, a planet...)."),
            "objects": names("Objects to scatter copies of; one is picked at random per point."),
            "count": integer("About how many copies.", minimum=1, maximum=100000, default=200),
            "scale_min": number("Smallest random scale.", minimum=0.0, default=0.8),
            "scale_max": number("Largest random scale.", minimum=0.0, default=1.2),
            "align_to_normal": boolean("Tilt copies to follow the surface (off: stay upright).", default=False),
            "random_rotation": boolean("Random spin around each copy's up axis.", default=True),
            "seed": integer("Random seed; change it for a different layout.", default=0),
            "hide_sources": boolean("Hide the source objects so only the copies show.", default=True),
            "name": string("Name of the scatter setup.", default="Scatter"),
        },
        ["target", "objects"],
    ),
    group="procedural",
)
def scatter_on_surface(
    target: str,
    objects: list[str],
    count: int = 200,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    align_to_normal: bool = False,
    random_rotation: bool = True,
    seed: int = 0,
    hide_sources: bool = True,
    name: str = "Scatter",
) -> dict[str, Any]:
    """Scatter copies of objects across a surface with a Geometry Nodes modifier (instances, so thousands
    stay light). Good for rocks, trees, grass, debris or crowds. Re-run with another seed to reshuffle;
    tweak the modifier later with set_modifier_prop or run_python."""
    ground = get_object(target, ["MESH"])
    sources = get_objects(objects)
    if ground in sources:
        raise ToolError("The target can't also be one of the scattered objects.")
    area = _surface_area(ground)
    if area <= 1e-9:
        raise ToolError(f"{target} has no faces to scatter onto.")

    col_name = unique_name(f"{name} Sources", bpy.data.collections)
    col = bpy.data.collections.new(col_name)
    scene().collection.children.link(col)
    for ob in sources:
        col.objects.link(ob)
        for other in list(ob.users_collection):
            if other is not col:
                other.objects.unlink(ob)
    if hide_sources:
        layer = bpy.context.view_layer.layer_collection.children.get(col.name)
        if layer is not None:
            layer.exclude = True

    tree = bpy.data.node_groups.new(unique_name(name, bpy.data.node_groups), "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n, links = tree.nodes, tree.links
    group_in = n.new("NodeGroupInput")
    group_out = n.new("NodeGroupOutput")
    distribute = n.new("GeometryNodeDistributePointsOnFaces")
    distribute.distribute_method = "RANDOM"
    _socket(distribute.inputs, "Density").default_value = count / area
    _socket(distribute.inputs, "Seed").default_value = seed
    info = n.new("GeometryNodeCollectionInfo")
    info.transform_space = "ORIGINAL"
    _socket(info.inputs, "Collection").default_value = col
    _socket(info.inputs, "Separate Children").default_value = True
    _socket(info.inputs, "Reset Children").default_value = True
    instance = n.new("GeometryNodeInstanceOnPoints")
    _socket(instance.inputs, "Pick Instance").default_value = True
    scale = n.new("FunctionNodeRandomValue")
    scale.data_type = "FLOAT"
    _socket(scale.inputs, "Min", "VALUE").default_value = scale_min
    _socket(scale.inputs, "Max", "VALUE").default_value = max(scale_min, scale_max)
    _socket(scale.inputs, "Seed", "INT").default_value = seed + 1
    join = n.new("GeometryNodeJoinGeometry")

    links.new(group_in.outputs[0], _socket(distribute.inputs, "Mesh"))
    links.new(_socket(distribute.outputs, "Points"), _socket(instance.inputs, "Points"))
    links.new(_socket(info.outputs, "Instances"), _socket(instance.inputs, "Instance"))
    links.new(_socket(scale.outputs, "Value", "VALUE"), _socket(instance.inputs, "Scale"))
    if align_to_normal:
        links.new(_socket(distribute.outputs, "Rotation"), _socket(instance.inputs, "Rotation"))
    last = _socket(instance.outputs, "Instances")
    if random_rotation:
        spin = n.new("FunctionNodeRandomValue")
        spin.data_type = "FLOAT_VECTOR"
        _socket(spin.inputs, "Min", "VECTOR").default_value = (0.0, 0.0, 0.0)
        _socket(spin.inputs, "Max", "VECTOR").default_value = (0.0, 0.0, 2 * math.pi)
        _socket(spin.inputs, "Seed", "INT").default_value = seed + 2
        rotate = n.new("GeometryNodeRotateInstances")
        _socket(rotate.inputs, "Local Space").default_value = True
        links.new(last, _socket(rotate.inputs, "Instances"))
        links.new(_socket(spin.outputs, "Value", "VECTOR"), _socket(rotate.inputs, "Rotation"))
        last = _socket(rotate.outputs, "Instances")
    links.new(group_in.outputs[0], join.inputs[0])
    links.new(last, join.inputs[0])
    links.new(join.outputs[0], group_out.inputs[0])
    for i, node in enumerate((group_in, distribute, info, scale, instance, join, group_out)):
        node.location = (i * 220, 0)

    mod = ground.modifiers.new(tree.name, "NODES")
    mod.node_group = tree
    sync()
    return {"modifier": mod.name, "node_group": tree.name, "sources_collection": col.name, "approx_count": count, "surface_area_m2": round(area, 3)}


_CLOTH = {
    "cotton": {"mass": 0.3, "tension_stiffness": 15, "compression_stiffness": 15, "bending_stiffness": 0.5},
    "silk": {"mass": 0.15, "tension_stiffness": 5, "compression_stiffness": 5, "bending_stiffness": 0.05},
    "denim": {"mass": 1.0, "tension_stiffness": 40, "compression_stiffness": 40, "bending_stiffness": 10},
    "leather": {"mass": 0.4, "tension_stiffness": 80, "compression_stiffness": 80, "bending_stiffness": 150},
    "rubber": {"mass": 3.0, "tension_stiffness": 15, "compression_stiffness": 15, "bending_stiffness": 25},
}


@tool(
    obj(
        {
            "objects": names("Objects to simulate."),
            "type": string(
                "rigid_body: falls and collides; rigid_body_passive: static or animated collider for rigid bodies; "
                "cloth; soft_body; collision: obstacle for cloth and soft bodies.",
                enum=["rigid_body", "rigid_body_passive", "cloth", "soft_body", "collision"],
            ),
            "mass": number("Mass in kg (rigid bodies) or per vertex (cloth).", minimum=0.0),
            "friction": number("Surface friction 0..1.", minimum=0.0, maximum=1.0),
            "bounciness": number("Restitution 0..1 (rigid bodies).", minimum=0.0, maximum=1.0),
            "shape": string("Rigid body collision shape.", enum=["CONVEX_HULL", "MESH", "BOX", "SPHERE", "CAPSULE", "CYLINDER", "CONE"]),
            "cloth_preset": string("Cloth material.", enum=list(_CLOTH)),
            "pin_group": string("Cloth: vertex group that stays pinned in place."),
        },
        ["objects", "type"],
    ),
    group="physics",
)
def add_physics(
    objects: list[str],
    type: str,
    mass: float | None = None,
    friction: float | None = None,
    bounciness: float | None = None,
    shape: str | None = None,
    cloth_preset: str | None = None,
    pin_group: str | None = None,
) -> str:
    """Add a physics simulation to objects: rigid bodies, cloth, soft bodies or colliders. Then call
    bake_physics (or play the timeline) to compute the motion."""
    obs = get_objects(objects, ["MESH"])
    done = []
    for ob in obs:
        if type.startswith("rigid_body"):
            with op_context(active=ob, selected=[ob]):
                if ob.rigid_body is None:
                    bpy.ops.rigidbody.object_add()
            body = ob.rigid_body
            body.type = "PASSIVE" if type == "rigid_body_passive" else "ACTIVE"
            body.collision_shape = shape or ("MESH" if type == "rigid_body_passive" else "CONVEX_HULL")
            if mass is not None:
                body.mass = mass
            if friction is not None:
                body.friction = friction
            if bounciness is not None:
                body.restitution = bounciness
        elif type == "cloth":
            mod = next((m for m in ob.modifiers if m.type == "CLOTH"), None) or ob.modifiers.new("Cloth", "CLOTH")
            settings = mod.settings
            for key, value in _CLOTH.get(cloth_preset or "cotton", {}).items():
                setattr(settings, key, value)
            if mass is not None:
                settings.mass = mass
            if pin_group:
                if pin_group not in ob.vertex_groups:
                    raise ToolError(f"{ob.name} has no vertex group {pin_group!r}.")
                settings.vertex_group_mass = pin_group
            mod.collision_settings.use_self_collision = True
        elif type == "soft_body":
            if not any(m.type == "SOFT_BODY" for m in ob.modifiers):
                ob.modifiers.new("Softbody", "SOFT_BODY")
            if mass is not None:
                ob.soft_body.mass = mass
        elif type == "collision":
            if not any(m.type == "COLLISION" for m in ob.modifiers):
                ob.modifiers.new("Collision", "COLLISION")
            if friction is not None:
                ob.collision.cloth_friction = friction * 80
        done.append(ob.name)
    return f"Added {type.replace('_', ' ')} physics to {', '.join(done)}. Bake with bake_physics."


@tool(
    obj({"frame_start": integer("First frame to simulate.", default=1), "frame_end": integer("Last frame to simulate.", minimum=1, default=250)}),
    group="physics",
    destructive=True,
)
def bake_physics(frame_start: int = 1, frame_end: int = 250) -> str:
    """Bake every physics simulation in the scene (rigid bodies, cloth, soft bodies) over a frame range,
    replacing earlier bakes. Can take a while for long ranges or dense cloth."""
    sc = scene()
    sc.frame_start, sc.frame_end = frame_start, max(frame_start, frame_end)
    caches = []
    if sc.rigidbody_world is not None:
        caches.append(sc.rigidbody_world.point_cache)
    for ob in sc.objects:
        for mod in ob.modifiers:
            cache = getattr(mod, "point_cache", None)
            if cache is not None:
                caches.append(cache)
    if not caches:
        raise ToolError("Nothing to bake: add physics with add_physics first.")
    for cache in caches:
        cache.frame_start, cache.frame_end = frame_start, max(frame_start, frame_end)
    with bpy.context.temp_override(**(ui_context() or {})):
        bpy.ops.ptcache.free_bake_all()
        bpy.ops.ptcache.bake_all(bake=True)
    sc.frame_set(frame_start)
    return f"Baked {len(caches)} simulation(s) over frames {frame_start}-{frame_end}."


def _to_mesh(ob: bpy.types.Object) -> bpy.types.Object:
    mesh = bpy.data.meshes.new_from_object(ob)
    name = ob.name
    mesh_ob = bpy.data.objects.new(name + "_mesh", mesh)
    mesh_ob.matrix_world = ob.matrix_world.copy()
    for col in ob.users_collection:
        col.objects.link(mesh_ob)
    data = ob.data
    bpy.data.objects.remove(ob)
    if data.users == 0:
        bpy.data.curves.remove(data)
    mesh_ob.name = mesh.name = name
    return mesh_ob


@tool(
    obj(
        {
            "points": array(vec3(), "Control points in world space, in order.", min_items=2),
            "name": string("Object name.", default="Curve"),
            "kind": string(
                "POLY: straight segments; BEZIER: smooth through points; NURBS: smooth, near points.", enum=["POLY", "BEZIER", "NURBS"], default="BEZIER"
            ),
            "closed": boolean("Join the last point back to the first.", default=False),
            "thickness": number("Tube radius in meters (0: a thin line).", minimum=0.0, default=0.0),
            "resolution": integer("Smoothness of the tube cross-section and curve.", minimum=1, maximum=64, default=8),
            "fill_caps": boolean("Close the tube ends.", default=True),
            "convert_to_mesh": boolean("Turn the result into a mesh for further modelling.", default=False),
        },
        ["points"],
    ),
    group="objects",
)
def create_curve(
    points: list[list[float]],
    name: str = "Curve",
    kind: str = "BEZIER",
    closed: bool = False,
    thickness: float = 0.0,
    resolution: int = 8,
    fill_caps: bool = True,
    convert_to_mesh: bool = False,
) -> dict[str, Any]:
    """Create a curve through points, optionally as a round tube: cables, pipes, rails, handles, wires,
    paths for camera_path, or profiles."""
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = resolution
    curve.bevel_depth = thickness
    curve.bevel_resolution = max(1, resolution // 2)
    curve.use_fill_caps = fill_caps
    spline = curve.splines.new(kind)
    if kind == "BEZIER":
        spline.bezier_points.add(len(points) - 1)
        for bp, co in zip(spline.bezier_points, points, strict=True):
            bp.co = co
            bp.handle_left_type = bp.handle_right_type = "AUTO"
    else:
        spline.points.add(len(points) - 1)
        for p, co in zip(spline.points, points, strict=True):
            p.co = (co[0], co[1], co[2], 1.0)
        if kind == "NURBS":
            spline.use_endpoint_u = not closed
            spline.order_u = min(4, len(points))
    spline.use_cyclic_u = closed
    ob = bpy.data.objects.new(unique_name(name, bpy.data.objects), curve)
    link(ob)
    if convert_to_mesh:
        ob = _to_mesh(ob)
    sync()
    lo, hi = world_bounds([ob])
    return {"name": ob.name, "type": ob.type, "points": len(points), "bounds": {"min": r(lo, 3), "max": r(hi, 3)}}


@tool(
    obj(
        {
            "text": string("The text. Use \\n for new lines."),
            "name": string("Object name.", default="Text"),
            "size": number("Letter height in meters.", minimum=0.001, default=0.5),
            "extrude": number("Depth of the letters in meters.", minimum=0.0, default=0.05),
            "bevel": number("Rounded edge size in meters.", minimum=0.0, default=0.0),
            "align": string("Horizontal alignment.", enum=["LEFT", "CENTER", "RIGHT", "JUSTIFY"], default="CENTER"),
            "location": vec3("Where to place the text."),
            "rotation": vec3("Rotation in degrees. (90, 0, 0) stands the text upright facing -Y."),
            "convert_to_mesh": boolean("Turn it into a mesh (for booleans, bevels, sculpting).", default=False),
        },
        ["text"],
    ),
    group="objects",
)
def create_text(
    text: str,
    name: str = "Text",
    size: float = 0.5,
    extrude: float = 0.05,
    bevel: float = 0.0,
    align: str = "CENTER",
    location: list[float] | None = None,
    rotation: list[float] | None = None,
    convert_to_mesh: bool = False,
) -> dict[str, Any]:
    """Create 3D text: signs, logos, labels, titles, engraved or embossed lettering."""
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = text.replace("\\n", "\n")
    curve.size = size
    curve.extrude = extrude
    curve.bevel_depth = bevel
    curve.align_x = align
    curve.align_y = "CENTER"
    ob = bpy.data.objects.new(unique_name(name, bpy.data.objects), curve)
    ob.location = vec(location)
    if rotation:
        ob.rotation_euler = [math.radians(a) for a in rotation]
    link(ob)
    if convert_to_mesh:
        ob = _to_mesh(ob)
    sync()
    lo, hi = world_bounds([ob])
    return {"name": ob.name, "type": ob.type, "size": r(hi - lo, 3)}
