from __future__ import annotations

import os
from typing import Any

import bpy
from mathutils import Matrix, Vector

from ..integrations import registry as integrations
from .common import describe, get_collection, get_object, get_objects, r, scene, select_only, ui_context, world_bounds
from .registry import ToolError, ToolKind, array, boolean, integer, names, obj, string, tool, vec3

MAX_LISTED = 150


@tool(
    obj(
        {
            "detail": boolean("Include bounds, vertex groups, bone names and collections per object.", default=False),
            "names": names("Only describe these objects (implies detail)."),
        }
    ),
    group="scene",
    mutates=False,
)
def get_scene_info(detail: bool = False, names: list[str] | None = None) -> dict[str, Any]:
    """Describe the current scene: frame range, render settings, camera, world, collections and every
    object's transform, dimensions, materials, modifiers and animation. Call this before editing objects you
    didn't create in this conversation."""
    sc = scene()
    if names:
        return {"objects": [describe(ob, True) for ob in get_objects(names)]}
    objs = list(sc.objects)
    info: dict[str, Any] = {
        "scene": sc.name,
        "frames": [sc.frame_start, sc.frame_end],
        "current_frame": sc.frame_current,
        "fps": round(sc.render.fps / sc.render.fps_base, 3),
        "render_engine": sc.render.engine,
        "resolution": [sc.render.resolution_x, sc.render.resolution_y],
        "camera": sc.camera.name if sc.camera else None,
        "world": sc.world.name if sc.world else None,
        "unit_scale": sc.unit_settings.scale_length,
        "collections": _collection_tree(sc.collection),
        "object_count": len(objs),
        "selected": [ob.name for ob in bpy.context.selected_objects][:50],
        "active": bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else None,
        "objects": [describe(ob, detail) for ob in objs[:MAX_LISTED]],
    }
    if len(objs) > MAX_LISTED:
        info["note"] = f"Only the first {MAX_LISTED} objects are listed; pass names to inspect others."
    if objs:
        lo, hi = world_bounds(objs)
        info["scene_bounds"] = {"min": r(lo, 3), "max": r(hi, 3)}
    return info


def _collection_tree(col: bpy.types.Collection) -> dict[str, Any]:
    node: dict[str, Any] = {"name": col.name, "objects": len(col.objects)}
    if col.children:
        node["children"] = [_collection_tree(c) for c in col.children]
    return node


@tool(obj({}), group="meta", mutates=False)
def get_capabilities() -> dict[str, Any]:
    """Report optional helpers (Rigify, MPFB, LoopTools, Extra Objects) and whether each is installed and
    enabled, whether image generation is configured (and with which model), the connected MCP servers, and
    the Blender version. Call this before relying on a helper add-on or on image generation."""
    from .. import state
    from ..preferences import get_prefs

    prefs = get_prefs()
    image = prefs.image_config() if prefs else None
    manager = state.get().mcp
    return {
        "blender_version": bpy.app.version_string,
        "helpers": integrations.report(),
        "image_generation": {"configured": bool(image and image.model), "model": image.model if image else None},
        "mcp_servers": manager.status() if manager is not None else {},
        "has_viewport": ui_context() is not None,
    }


@tool(obj({"steps": integer("How many undo steps to take back.", minimum=1, maximum=50, default=1)}), group="meta")
def undo(steps: int = 1) -> str:
    """Undo the last steps (each tool call is one step). Use to back out of a change that went wrong."""
    ctx = ui_context()
    if ctx is None:
        raise ToolError("Undo needs the Blender UI; it isn't available when running headless.")
    with bpy.context.temp_override(**ctx):
        for _ in range(steps):
            if bpy.ops.ed.undo.poll():
                bpy.ops.ed.undo()
    return f"Undid {steps} step(s)."


@tool(
    obj({"question": string("The question to show the user."), "options": array(string(), "Optional short answers to suggest.", max_items=6)}, ["question"]),
    group="meta",
    kind=ToolKind.QUESTION,
    mutates=False,
)
def ask_user(question: str, options: list[str] | None = None) -> None:
    """Ask the user a clarifying question and wait for the answer. Only use this when the answer changes
    what you build in a way you can't reasonably choose yourself."""


@tool(
    obj(
        {
            "names": names("Objects to select."),
            "mode": string("Replace the selection, add to it, or remove from it.", enum=["set", "add", "remove"], default="set"),
            "active": string("Object to make active (defaults to the first name)."),
        },
        ["names"],
    ),
    group="scene",
    mutates=False,
)
def select_objects(names: list[str], mode: str = "set", active: str | None = None) -> str:
    """Change the object selection and the active object."""
    objs = get_objects(names)
    if mode == "set":
        select_only(objs, get_object(active) if active else objs[0])
    else:
        for ob in objs:
            ob.select_set(mode == "add")
        if active:
            bpy.context.view_layer.objects.active = get_object(active)
    return f"Selected: {', '.join(o.name for o in bpy.context.selected_objects)}"


@tool(
    obj({"names": names("Objects to delete."), "with_children": boolean("Also delete their children.", default=False)}, ["names"]),
    group="scene",
    destructive=True,
)
def delete_objects(names: list[str], with_children: bool = False) -> str:
    """Delete objects from the file (their unused mesh/curve data is removed too)."""
    objs = get_objects(names)
    if with_children:
        objs += [c for ob in objs for c in ob.children_recursive]
    removed = []
    for ob in {o.name: o for o in objs}.values():
        data = ob.data
        removed.append(ob.name)
        bpy.data.objects.remove(ob, do_unlink=True)
        if data is not None and data.users == 0 and isinstance(data, (bpy.types.Mesh, bpy.types.Curve)):
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            else:
                bpy.data.curves.remove(data)
    return f"Deleted {len(removed)}: {', '.join(removed)}"


@tool(
    obj(
        {
            "name": string("Object to duplicate."),
            "count": integer("Number of copies.", minimum=1, maximum=200, default=1),
            "offset": vec3("Location offset between consecutive copies, in meters."),
            "linked": boolean("Share mesh data with the original (instances).", default=False),
            "new_name": string("Base name for the copies."),
        },
        ["name"],
    ),
    group="scene",
)
def duplicate_object(name: str, count: int = 1, offset: list[float] | None = None, linked: bool = False, new_name: str | None = None) -> str:
    """Duplicate an object one or more times, optionally stepping each copy by an offset."""
    src = get_object(name)
    step = Vector(offset or (0, 0, 0))
    made = []
    for i in range(1, count + 1):
        dup = src.copy()
        if src.data is not None and not linked:
            dup.data = src.data.copy()
        dup.animation_data_clear()
        if new_name:
            dup.name = new_name if count == 1 else f"{new_name}.{i:03d}"
        dup.location = src.location + step * i
        for col in src.users_collection:
            col.objects.link(dup)
        made.append(dup.name)
    return f"Created {', '.join(made)}"


@tool(
    obj(
        {
            "children": names("Objects to (un)parent."),
            "parent": string("New parent object, or omit to clear the parent."),
            "keep_transform": boolean("Keep the children where they are in the world.", default=True),
        },
        ["children"],
    ),
    group="scene",
)
def parent_objects(children: list[str], parent: str | None = None, keep_transform: bool = True) -> str:
    """Parent objects to another object, or clear their parent. Use to group the parts of an asset."""
    kids = get_objects(children)
    target = get_object(parent) if parent else None
    for ob in kids:
        world = ob.matrix_world.copy()
        ob.parent = target
        if target is not None:
            ob.matrix_parent_inverse = target.matrix_world.inverted() if keep_transform else ob.matrix_parent_inverse
        if keep_transform:
            ob.matrix_world = world
    return f"Parented {len(kids)} object(s) to {target.name}" if target else f"Cleared parent of {len(kids)}"


@tool(
    obj(
        {
            "name": string("Mesh or curve object."),
            "to": string("Where to put the origin.", enum=["geometry_center", "bounds_center", "bounds_bottom", "world_origin", "cursor", "point"]),
            "point": vec3("World-space point, for to=point."),
        },
        ["name", "to"],
    ),
    group="scene",
)
def set_origin(name: str, to: str, point: list[float] | None = None) -> str:
    """Move an object's origin without moving its geometry, e.g. to the bottom of the bounds so it sits on
    the floor when placed at z=0."""
    ob = get_object(name, ("MESH", "CURVE", "FONT"))
    mw = ob.matrix_world
    if to == "geometry_center":
        if ob.type != "MESH" or not ob.data.vertices:
            raise ToolError("geometry_center needs a mesh with vertices.")
        target = mw @ (sum((v.co for v in ob.data.vertices), Vector()) / len(ob.data.vertices))
    elif to in ("bounds_center", "bounds_bottom"):
        lo, hi = world_bounds([ob])
        target = (lo + hi) / 2
        if to == "bounds_bottom":
            target.z = lo.z
    elif to == "world_origin":
        target = Vector((0, 0, 0))
    elif to == "cursor":
        target = scene().cursor.location.copy()
    else:
        if point is None:
            raise ToolError("to=point needs a point.")
        target = Vector(point)
    local = mw.inverted() @ target
    ob.data.transform(Matrix.Translation(-local))
    ob.matrix_world = mw @ Matrix.Translation(local)
    return f"Origin of {ob.name} at {r(target, 3)}"


@tool(
    obj(
        {
            "action": string("What to do.", enum=["create", "move_objects", "link_objects", "unlink_objects", "rename", "delete", "hide", "show"]),
            "collection": string("Collection name (created if action=create)."),
            "objects": names("Objects for move/link/unlink."),
            "parent": string("Parent collection for create (default: the scene collection)."),
            "new_name": string("New name, for rename."),
        },
        ["action", "collection"],
    ),
    group="scene",
)
def collection_ops(action: str, collection: str, objects: list[str] | None = None, parent: str | None = None, new_name: str | None = None) -> str:
    """Create, rename, delete, hide or show collections, and move or link objects into them."""
    sc = scene()
    if action == "create":
        col = bpy.data.collections.get(collection) or bpy.data.collections.new(collection)
        parent_col = get_collection(parent) if parent else sc.collection
        if col.name not in parent_col.children:
            parent_col.children.link(col)
        for ob in get_objects(objects or []):
            _move(ob, col)
        return f"Collection {col.name} ready"
    col = get_collection(collection)
    if action in ("move_objects", "link_objects", "unlink_objects"):
        if not objects:
            raise ToolError(f"{action} needs objects.")
        for ob in get_objects(objects):
            if action == "move_objects":
                _move(ob, col)
            elif action == "link_objects" and ob.name not in col.objects:
                col.objects.link(ob)
            elif action == "unlink_objects" and ob.name in col.objects and len(ob.users_collection) > 1:
                col.objects.unlink(ob)
        return f"{action} done for {len(objects)} object(s)"
    if action == "rename":
        if not new_name:
            raise ToolError("rename needs new_name.")
        col.name = new_name
        return f"Renamed to {col.name}"
    if action == "delete":
        for ob in list(col.objects):
            if len(ob.users_collection) == 1:
                sc.collection.objects.link(ob)
        bpy.data.collections.remove(col)
        return f"Deleted collection {collection}; its objects moved to the scene collection"
    col.hide_viewport = action == "hide"
    col.hide_render = action == "hide"
    return f"{'Hid' if action == 'hide' else 'Showed'} {col.name}"


def _move(ob: bpy.types.Object, col: bpy.types.Collection) -> None:
    for old in list(ob.users_collection):
        old.objects.unlink(ob)
    col.objects.link(ob)


_IMPORTERS = {
    ".fbx": ("import_scene", "fbx"),
    ".obj": ("wm", "obj_import"),
    ".gltf": ("import_scene", "gltf"),
    ".glb": ("import_scene", "gltf"),
    ".stl": ("wm", "stl_import"),
    ".ply": ("wm", "ply_import"),
    ".usd": ("wm", "usd_import"),
    ".usda": ("wm", "usd_import"),
    ".usdc": ("wm", "usd_import"),
    ".usdz": ("wm", "usd_import"),
    ".abc": ("wm", "alembic_import"),
    ".dae": ("wm", "collada_import"),
}


@tool(
    obj(
        {
            "filepath": string("Absolute path of the file to import (.blend, .fbx, .obj, .gltf/.glb, .stl, .ply, .usd*, .abc)."),
            "objects": names("For .blend files: which objects to bring in (default: all)."),
            "link": boolean("For .blend files: link instead of append.", default=False),
            "collection": string("Put the imported objects in this collection (created if needed)."),
        },
        ["filepath"],
    ),
    group="scene",
)
def import_asset(filepath: str, objects: list[str] | None = None, link: bool = False, collection: str | None = None) -> dict[str, Any]:
    """Import a model file from disk into the scene. Use for existing assets in Scene mode."""
    path = bpy.path.abspath(filepath)
    if not os.path.isfile(path):
        raise ToolError(f"No file at {path}.")
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".blend":
        with bpy.data.libraries.load(path, link=link) as (src, dst):
            dst.objects = [n for n in src.objects if not objects or n in objects]
        target = bpy.context.collection
        for ob in dst.objects:
            if ob is not None:
                target.objects.link(ob)
    elif ext in _IMPORTERS:
        module, op = _IMPORTERS[ext]
        operator = getattr(getattr(bpy.ops, module), op, None)
        if operator is None or not hasattr(operator, "poll"):
            raise ToolError(f"No importer for {ext} in this Blender build.")
        with bpy.context.temp_override(**(ui_context() or {})):
            operator(filepath=path)
    else:
        raise ToolError(f"Unsupported file type {ext}.")
    new = [ob for ob in bpy.data.objects if ob not in before]
    if collection:
        col = bpy.data.collections.get(collection) or bpy.data.collections.new(collection)
        if col.name not in scene().collection.children and col not in scene().collection.children_recursive:
            scene().collection.children.link(col)
        for ob in new:
            _move(ob, col)
    lo, hi = world_bounds(new) if new else (Vector(), Vector())
    return {"imported": [ob.name for ob in new][:100], "count": len(new), "bounds": {"min": r(lo, 3), "max": r(hi, 3)}}
