from __future__ import annotations

import difflib
import math
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

from .registry import ToolError, array, integer, number, obj, string, vec3


def _missing(kind: str, name: str, pool: Iterable[str]) -> ToolError:
    close = difflib.get_close_matches(name, list(pool), n=3, cutoff=0.5)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    return ToolError(f"No {kind} named {name!r}.{hint}")


def get_object(name: str, types: Sequence[str] | None = None) -> bpy.types.Object:
    ob = bpy.data.objects.get(name)
    if ob is None:
        raise _missing("object", name, bpy.data.objects.keys())
    if types and ob.type not in types:
        raise ToolError(f"{name!r} is a {ob.type.lower()} object; this needs {' or '.join(t.lower() for t in types)}.")
    return ob


def get_objects(names: Iterable[str], types: Sequence[str] | None = None) -> list[bpy.types.Object]:
    return [get_object(n, types) for n in names]


def get_mesh_object(name: str) -> bpy.types.Object:
    return get_object(name, ("MESH",))


def get_armature(name: str) -> bpy.types.Object:
    return get_object(name, ("ARMATURE",))


def get_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        raise _missing("material", name, bpy.data.materials.keys())
    return mat


def get_image(name: str) -> bpy.types.Image:
    img = bpy.data.images.get(name)
    if img is None:
        raise _missing("image", name, bpy.data.images.keys())
    return img


def get_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        raise _missing("collection", name, bpy.data.collections.keys())
    return col


def get_action(name: str) -> bpy.types.Action:
    act = bpy.data.actions.get(name)
    if act is None:
        raise _missing("action", name, bpy.data.actions.keys())
    return act


def get_bone(arm_ob: bpy.types.Object, name: str) -> bpy.types.PoseBone:
    pb = arm_ob.pose.bones.get(name)
    if pb is None:
        raise _missing(f"bone in {arm_ob.name!r}", name, arm_ob.pose.bones.keys())
    return pb


def vec(v: Sequence[float] | None, default: Sequence[float] = (0.0, 0.0, 0.0)) -> Vector:
    return Vector(v if v is not None else default)


def euler_deg(v: Sequence[float] | None) -> Euler:
    v = v or (0.0, 0.0, 0.0)
    return Euler([math.radians(a) for a in v], "XYZ")


def r(v: Any, nd: int = 4) -> Any:
    if isinstance(v, float):
        return round(v, nd)
    return [r(x, nd) for x in v] if hasattr(v, "__len__") and not isinstance(v, str) else v


def deg(v: Sequence[float]) -> list[float]:
    return [round(math.degrees(a), 2) for a in v]


def scene() -> bpy.types.Scene:
    return bpy.context.scene


def sync() -> None:
    bpy.context.view_layer.update()


def ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        scene().collection.children.link(col)
    return col


def link(ob: bpy.types.Object, collection: str | None = None) -> None:
    target = ensure_collection(collection) if collection else bpy.context.collection or scene().collection
    target.objects.link(ob)


def unique_name(base: str, pool: Any) -> str:
    if base not in pool:
        return base
    i = 1
    while f"{base}.{i:03d}" in pool:
        i += 1
    return f"{base}.{i:03d}"


def set_rna(target: Any, props: dict[str, Any], what: str) -> list[str]:
    done: list[str] = []
    rna_props = target.bl_rna.properties
    for key, value in props.items():
        if key not in rna_props or key == "rna_type":
            options = sorted(p.identifier for p in rna_props if not p.is_readonly and p.identifier != "rna_type")
            close = difflib.get_close_matches(key, options, n=3)
            hint = f" Did you mean {', '.join(close)}?" if close else f" Settable: {', '.join(options[:40])}"
            raise ToolError(f"{what} has no property {key!r}.{hint}")
        prop = rna_props[key]
        if prop.is_readonly:
            raise ToolError(f"{what}.{key} is read-only.")
        if prop.type == "POINTER" and isinstance(value, str):
            value = _resolve_pointer(prop, value)
        elif prop.type == "FLOAT" and getattr(prop, "subtype", "") in {"ANGLE", "EULER"}:
            value = math.radians(value) if isinstance(value, (int, float)) else [math.radians(v) for v in value]
        try:
            setattr(target, key, value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ToolError(f"Can't set {what}.{key} = {value!r}: {exc}") from exc
        done.append(key)
    return done


def _resolve_pointer(prop: Any, name: str) -> Any:
    fixed = prop.fixed_type.identifier if prop.fixed_type else ""
    pools = {
        "Object": bpy.data.objects,
        "Material": bpy.data.materials,
        "Image": bpy.data.images,
        "Collection": bpy.data.collections,
        "Texture": bpy.data.textures,
        "Curve": bpy.data.curves,
        "Mesh": bpy.data.meshes,
        "Action": bpy.data.actions,
        "NodeTree": bpy.data.node_groups,
    }
    pool = pools.get(fixed)
    if pool is None:
        raise ToolError(f"Can't set a {fixed} reference by name.")
    item = pool.get(name)
    if item is None:
        raise _missing(fixed.lower(), name, pool.keys())
    return item


def ui_context(area_type: str = "VIEW_3D") -> dict[str, Any] | None:
    wm = bpy.context.window_manager
    for window in getattr(wm, "windows", []):
        screen = window.screen
        for area in screen.areas:
            if area.type == area_type:
                region = next((rg for rg in area.regions if rg.type == "WINDOW"), None)
                if region is not None:
                    return {"window": window, "screen": screen, "area": area, "region": region}
    return None


def require_ui(area_type: str = "VIEW_3D") -> dict[str, Any]:
    ctx = ui_context(area_type)
    if ctx is None:
        raise ToolError("This needs an open 3D Viewport (Blender is running without one).")
    return ctx


def select_only(objects: Sequence[bpy.types.Object], active: bpy.types.Object | None = None) -> None:
    view_layer = bpy.context.view_layer
    for ob in view_layer.objects:
        if ob.select_get():
            ob.select_set(False)
    for ob in objects:
        ob.select_set(True)
    view_layer.objects.active = active or (objects[0] if objects else None)


def ensure_object_mode() -> None:
    ob = bpy.context.view_layer.objects.active
    if ob is not None and ob.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


@contextmanager
def edit_mode(ob: bpy.types.Object, mode: str = "EDIT") -> Iterator[None]:
    ensure_object_mode()
    select_only([ob], ob)
    with bpy.context.temp_override(**(ui_context() or {})):
        bpy.ops.object.mode_set(mode=mode)
        try:
            yield
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")


@contextmanager
def op_context(active: bpy.types.Object | None = None, selected: Sequence[bpy.types.Object] = ()) -> Iterator[None]:
    ensure_object_mode()
    if active is not None or selected:
        select_only(list(selected) or [active], active)
    sync()
    with bpy.context.temp_override(**(ui_context() or {})):
        yield


SELECTION = obj(
    {
        "by": string("How to pick mesh elements.", enum=["all", "indices", "normal", "position", "material", "vertex_group", "sharp", "boundary"]),
        "indices": array(integer(minimum=0), "Element indices, for by=indices."),
        "direction": vec3("World-space direction, for by=normal: faces facing within angle_deg of it."),
        "angle_deg": number("Angle tolerance for by=normal, or the minimum edge angle for by=sharp.", default=30),
        "min": vec3("World-space box corner, for by=position (elements whose center is inside)."),
        "max": vec3("World-space box corner, for by=position."),
        "material_index": integer("Material slot index, for by=material.", minimum=0),
        "vertex_group": string("Vertex group name, for by=vertex_group."),
    },
    ["by"],
    description="Which elements to act on. Defaults to all.",
)


def _normal_world(ob: bpy.types.Object) -> Matrix:
    return ob.matrix_world.to_3x3().inverted_safe().transposed()


def _in_box(p: Vector, spec: dict[str, Any]) -> bool:
    lo = spec.get("min") or (-math.inf,) * 3
    hi = spec.get("max") or (math.inf,) * 3
    return all(min(lo[i], hi[i]) <= p[i] <= max(lo[i], hi[i]) for i in range(3))


def pick_faces(bm: bmesh.types.BMesh, ob: bpy.types.Object, spec: dict[str, Any] | None) -> list[bmesh.types.BMFace]:
    spec = spec or {"by": "all"}
    by = spec["by"]
    faces = list(bm.faces)
    mw = ob.matrix_world
    if by == "all":
        return faces
    if by == "indices":
        bm.faces.ensure_lookup_table()
        idx = set(spec.get("indices") or [])
        return [f for f in faces if f.index in idx]
    if by == "normal":
        direction = Vector(spec.get("direction") or (0, 0, 1)).normalized()
        limit = math.radians(spec.get("angle_deg", 30))
        nm = _normal_world(ob)
        return [f for f in faces if (nm @ f.normal).normalized().angle(direction, math.pi) <= limit]
    if by == "position":
        return [f for f in faces if _in_box(mw @ f.calc_center_median(), spec)]
    if by == "material":
        return [f for f in faces if f.material_index == spec.get("material_index", 0)]
    if by == "vertex_group":
        verts = set(_group_verts(bm, ob, spec))
        return [f for f in faces if all(v in verts for v in f.verts)]
    if by in ("sharp", "boundary"):
        edges = set(pick_edges(bm, ob, spec))
        return [f for f in faces if any(e in edges for e in f.edges)]
    raise ToolError(f"Unknown selection {by!r}.")


def pick_edges(bm: bmesh.types.BMesh, ob: bpy.types.Object, spec: dict[str, Any] | None) -> list[bmesh.types.BMEdge]:
    spec = spec or {"by": "all"}
    by = spec["by"]
    edges = list(bm.edges)
    mw = ob.matrix_world
    if by == "all":
        return edges
    if by == "indices":
        idx = set(spec.get("indices") or [])
        return [e for e in edges if e.index in idx]
    if by == "sharp":
        limit = math.radians(spec.get("angle_deg", 30))
        return [e for e in edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) >= limit]
    if by == "boundary":
        return [e for e in edges if e.is_boundary]
    if by == "position":
        return [e for e in edges if _in_box(mw @ ((e.verts[0].co + e.verts[1].co) / 2), spec)]
    faces = set(pick_faces(bm, ob, spec))
    return [e for e in edges if any(f in faces for f in e.link_faces)]


def pick_verts(bm: bmesh.types.BMesh, ob: bpy.types.Object, spec: dict[str, Any] | None) -> list[bmesh.types.BMVert]:
    spec = spec or {"by": "all"}
    by = spec["by"]
    verts = list(bm.verts)
    if by == "all":
        return verts
    if by == "indices":
        idx = set(spec.get("indices") or [])
        return [v for v in verts if v.index in idx]
    if by == "position":
        mw = ob.matrix_world
        return [v for v in verts if _in_box(mw @ v.co, spec)]
    if by == "vertex_group":
        return _group_verts(bm, ob, spec)
    if by in ("sharp", "boundary"):
        return list({v for e in pick_edges(bm, ob, spec) for v in e.verts})
    return list({v for f in pick_faces(bm, ob, spec) for v in f.verts})


def _group_verts(bm: bmesh.types.BMesh, ob: bpy.types.Object, spec: dict[str, Any]) -> list[bmesh.types.BMVert]:
    name = spec.get("vertex_group") or ""
    group = ob.vertex_groups.get(name)
    if group is None:
        raise _missing(f"vertex group on {ob.name!r}", name, ob.vertex_groups.keys())
    deform = bm.verts.layers.deform.active
    if deform is None:
        return []
    return [v for v in bm.verts if v[deform].get(group.index, 0.0) > 0.0]


@contextmanager
def bmesh_of(ob: bpy.types.Object) -> Iterator[bmesh.types.BMesh]:
    if ob.type != "MESH":
        raise ToolError(f"{ob.name!r} is not a mesh.")
    ensure_object_mode()
    sync()
    bm = bmesh.new()
    try:
        bm.from_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        yield bm
        bm.normal_update()
        bm.to_mesh(ob.data)
        ob.data.update()
    finally:
        bm.free()


def require_selection(items: Sequence[Any], what: str) -> None:
    if not items:
        raise ToolError(f"The selection matched no {what}. Check the selection spec (get_scene_info shows bounds).")


def describe(ob: bpy.types.Object, detail: bool = False) -> dict[str, Any]:
    sync()
    info: dict[str, Any] = {
        "name": ob.name,
        "type": ob.type,
        "location": r(ob.location, 3),
        "rotation_deg": deg(ob.rotation_euler),
        "scale": r(ob.scale, 3),
        "dimensions": r(ob.dimensions, 3),
    }
    if ob.parent:
        info["parent"] = ob.parent.name + (f":{ob.parent_bone}" if ob.parent_bone else "")
    if ob.hide_get() or ob.hide_viewport:
        info["hidden"] = True
    if ob.type == "MESH":
        me = ob.data
        info["mesh"] = {"verts": len(me.vertices), "faces": len(me.polygons), "uv_maps": len(me.uv_layers)}
        if detail:
            info["vertex_groups"] = [g.name for g in ob.vertex_groups]
            if me.shape_keys:
                info["shape_keys"] = [k.name for k in me.shape_keys.key_blocks]
    if ob.material_slots:
        info["materials"] = [s.material.name if s.material else None for s in ob.material_slots]
    if ob.modifiers:
        info["modifiers"] = [f"{m.name} ({m.type})" for m in ob.modifiers]
    if ob.constraints:
        info["constraints"] = [f"{c.name} ({c.type})" for c in ob.constraints]
    if ob.type == "ARMATURE":
        bones = ob.data.bones
        info["bones"] = len(bones)
        if detail:
            info["bone_names"] = [b.name for b in bones][:200]
    elif ob.type == "LIGHT":
        info["light"] = {"type": ob.data.type, "energy": r(ob.data.energy, 2), "color": r(ob.data.color, 3)}
    elif ob.type == "CAMERA":
        info["camera"] = {"lens_mm": r(ob.data.lens, 2), "active": scene().camera == ob}
    anim = ob.animation_data
    if anim and anim.action:
        info["action"] = anim.action.name
    if anim and anim.nla_tracks:
        info["nla_tracks"] = [t.name for t in anim.nla_tracks]
    if detail:
        info["collections"] = [c.name for c in ob.users_collection]
        bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        info["world_bounds"] = {"min": r([min(p[i] for p in bb) for i in range(3)], 3), "max": r([max(p[i] for p in bb) for i in range(3)], 3)}
    return info


def world_bounds(objects: Sequence[bpy.types.Object]) -> tuple[Vector, Vector]:
    sync()
    pts = [ob.matrix_world @ Vector(c) for ob in objects for c in ob.bound_box]
    if not pts:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return lo, hi
