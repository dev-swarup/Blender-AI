from __future__ import annotations

from typing import Any

import bpy

from .common import get_object, op_context, set_rna
from .registry import ToolError, free_object, obj, string, tool


def _modifier(ob: bpy.types.Object, name: str) -> bpy.types.Modifier:
    mod = ob.modifiers.get(name)
    if mod is None:
        have = ", ".join(m.name for m in ob.modifiers) or "none"
        raise ToolError(f"{ob.name!r} has no modifier {name!r} (it has: {have}).")
    return mod


def _settings(mod: bpy.types.Modifier) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prop in mod.bl_rna.properties:
        if prop.is_readonly or prop.identifier in {"rna_type", "name", "type"} or prop.type == "COLLECTION":
            continue
        value = getattr(mod, prop.identifier)
        if prop.type == "POINTER":
            value = getattr(value, "name", None)
        elif hasattr(value, "__len__") and not isinstance(value, str):
            value = list(value)
        if isinstance(value, float):
            value = round(value, 4)
        out[prop.identifier] = value
    return out


def apply_modifier_now(ob: bpy.types.Object, name: str) -> None:
    with op_context(ob, [ob]):
        bpy.ops.object.modifier_apply(modifier=name)


@tool(
    obj(
        {
            "name": string("Object to add the modifier to."),
            "type": string(
                "Modifier type, e.g. BEVEL, SUBSURF, MIRROR, SOLIDIFY, ARRAY, BOOLEAN, WEIGHTED_NORMAL, SCREW, "
                "DISPLACE, DECIMATE, REMESH, SHRINKWRAP, CURVE, SIMPLE_DEFORM, LATTICE, TRIANGULATE, WIREFRAME."
            ),
            "settings": free_object(
                'Modifier properties by name, e.g. BEVEL {"width": 0.02, "segments": 3, "limit_method": "ANGLE"}, '
                'MIRROR {"use_axis": [true, false, false], "use_clip": true}, ARRAY {"count": 5, "relative_offset_displace": [1.1, 0, 0]}. '
                "Object references (mirror_object, offset_object...) take object names; angles are in degrees."
            ),
            "modifier_name": string("Name for the modifier."),
        },
        ["name", "type"],
    ),
    group="modifiers",
)
def add_modifier(name: str, type: str, settings: dict[str, Any] | None = None, modifier_name: str | None = None) -> dict[str, Any]:
    """Add a non-destructive modifier and set its properties. Returns all of its current settings."""
    ob = get_object(name)
    valid = bpy.types.Modifier.bl_rna.properties["type"].enum_items.keys()
    if type not in valid:
        raise ToolError(f"Unknown modifier type {type!r}. Options: {', '.join(valid)}")
    try:
        mod = ob.modifiers.new(modifier_name or type.title(), type)
    except (TypeError, RuntimeError) as exc:
        raise ToolError(f"Can't add {type} to {ob.type.lower()} object {ob.name!r}: {exc}") from exc
    if mod is None:
        raise ToolError(f"{type} isn't supported on {ob.type.lower()} objects.")
    if settings:
        try:
            set_rna(mod, settings, f"{type} modifier")
        except ToolError:
            ob.modifiers.remove(mod)
            raise
    return {"object": ob.name, "modifier": mod.name, "settings": _settings(mod)}


@tool(
    obj(
        {"name": string("Object."), "modifier": string("Modifier name."), "settings": free_object("Properties to change, by name (angles in degrees).")},
        ["name", "modifier", "settings"],
    ),
    group="modifiers",
)
def set_modifier_prop(name: str, modifier: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Change settings of an existing modifier."""
    ob = get_object(name)
    mod = _modifier(ob, modifier)
    set_rna(mod, settings, f"Modifier {mod.name!r}")
    return {"object": ob.name, "modifier": mod.name, "settings": _settings(mod)}


@tool(obj({"name": string("Object."), "modifier": string("Modifier name to apply.")}, ["name", "modifier"]), group="modifiers", destructive=True)
def apply_modifier(name: str, modifier: str) -> str:
    """Apply a modifier, baking its result into the mesh. Only do this when later steps need real geometry."""
    ob = get_object(name)
    _modifier(ob, modifier)
    apply_modifier_now(ob, modifier)
    return f"Applied {modifier} on {ob.name}; it now has {len(ob.data.vertices) if ob.type == 'MESH' else '?'} vertices"


@tool(obj({"name": string("Object."), "modifier": string("Modifier name to remove.")}, ["name", "modifier"]), group="modifiers", destructive=True)
def remove_modifier(name: str, modifier: str) -> str:
    """Remove a modifier from an object."""
    ob = get_object(name)
    ob.modifiers.remove(_modifier(ob, modifier))
    return f"Removed {modifier} from {ob.name}"
