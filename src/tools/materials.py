from __future__ import annotations

import difflib
from typing import Any

import bpy

from .common import SELECTION, bmesh_of, get_material, get_objects, pick_faces, set_rna
from .registry import ToolError, boolean, color, free_object, integer, names, number, obj, scalar, string, tool


def principled(mat: bpy.types.Material) -> bpy.types.ShaderNode:
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    node = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if node is None:
        node = nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None) or nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(node.outputs["BSDF"], out.inputs["Surface"])
    return node


def _input(node: bpy.types.ShaderNode, name: str) -> bpy.types.NodeSocket:
    sock = node.inputs.get(name)
    if sock is None:
        lowered = {s.name.lower(): s for s in node.inputs if s.enabled}
        sock = lowered.get(name.lower())
    if sock is None:
        options = [s.name for s in node.inputs if s.enabled]
        close = difflib.get_close_matches(name, options, n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else f" Inputs: {', '.join(options)}"
        raise ToolError(f"Principled BSDF has no input {name!r}.{hint}")
    return sock


def set_inputs(node: bpy.types.ShaderNode, values: dict[str, Any]) -> list[str]:
    done = []
    for key, value in values.items():
        sock = _input(node, key)
        if sock.is_linked:
            for link in list(sock.links):
                node.id_data.links.remove(link)
        current = sock.default_value
        if hasattr(current, "__len__") and not isinstance(value, (list, tuple)):
            value = [value] * len(current)
        if hasattr(current, "__len__") and len(value) == 3 and len(current) == 4:
            value = [*value, 1.0]
        try:
            sock.default_value = value
        except (TypeError, ValueError) as exc:
            raise ToolError(f"Can't set {sock.name} to {value!r}: {exc}") from exc
        done.append(sock.name)
    return done


@tool(
    obj(
        {
            "name": string("Material name (reused if it exists)."),
            "base_color": color("Base color, linear RGB 0..1."),
            "metallic": number("0 = dielectric, 1 = metal.", minimum=0, maximum=1),
            "roughness": number("0 = mirror, 1 = matte.", minimum=0, maximum=1),
            "emission_color": color("Emission color."),
            "emission_strength": number("Emission strength.", minimum=0),
            "alpha": number("Opacity.", minimum=0, maximum=1),
            "transmission": number("Glass-like transmission.", minimum=0, maximum=1),
            "ior": number("Index of refraction.", minimum=1, maximum=3),
            "assign_to": names("Objects to assign it to (first slot)."),
        },
        ["name"],
    ),
    group="materials",
)
def create_material(
    name: str,
    base_color: list[float] | None = None,
    metallic: float | None = None,
    roughness: float | None = None,
    emission_color: list[float] | None = None,
    emission_strength: float | None = None,
    alpha: float | None = None,
    transmission: float | None = None,
    ior: float | None = None,
    assign_to: list[str] | None = None,
) -> dict[str, Any]:
    """Create (or update) a Principled BSDF material with the common inputs, optionally assigning it."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    node = principled(mat)
    values: dict[str, Any] = {}
    if base_color is not None:
        values["Base Color"] = base_color
        mat.diffuse_color = [*base_color, 1.0]
    if metallic is not None:
        values["Metallic"] = metallic
    if roughness is not None:
        values["Roughness"] = roughness
    if emission_color is not None:
        values["Emission Color"] = emission_color
    if emission_strength is not None:
        values["Emission Strength"] = emission_strength
    if alpha is not None:
        values["Alpha"] = alpha
        if alpha < 1.0 and hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"
    if transmission is not None:
        values["Transmission Weight"] = transmission
    if ior is not None:
        values["IOR"] = ior
    set_inputs(node, values)
    if assign_to:
        assign_material(assign_to, mat.name)
    return {"material": mat.name, "set": list(values)}


@tool(
    obj(
        {
            "objects": names("Objects to assign to."),
            "material": string("Material name."),
            "slot": integer("Material slot index to fill (default 0; a new slot is added if needed).", minimum=0),
            "faces": {**SELECTION, "description": "Only these faces get the material (mesh objects, adds a slot)."},
            "replace_all": boolean("Clear other slots first.", default=False),
        },
        ["objects", "material"],
    ),
    group="materials",
)
def assign_material(objects: list[str], material: str, slot: int | None = None, faces: dict[str, Any] | None = None, replace_all: bool = False) -> str:
    """Assign a material to objects, to a specific slot, or only to some faces of a mesh."""
    mat = get_material(material)
    objs = get_objects(objects)
    for ob in objs:
        if not hasattr(ob.data, "materials"):
            raise ToolError(f"{ob.name!r} can't hold materials.")
        mats = ob.data.materials
        if replace_all:
            mats.clear()
        if faces is not None:
            if ob.type != "MESH":
                raise ToolError("Face assignment needs a mesh.")
            index = next((i for i, m in enumerate(mats) if m == mat), None)
            if index is None:
                if not mats:
                    mats.append(None)
                mats.append(mat)
                index = len(mats) - 1
            with bmesh_of(ob) as bm:
                for f in pick_faces(bm, ob, faces):
                    f.material_index = index
        else:
            i = slot or 0
            while len(mats) <= i:
                mats.append(None)
            mats[i] = mat
    return f"Assigned {mat.name} to {', '.join(o.name for o in objs)}"


@tool(
    obj(
        {
            "material": string("Material name."),
            "inputs": {
                "type": "object",
                "additionalProperties": scalar(),
                "description": 'Principled BSDF inputs by socket name, e.g. {"Base Color": [0.8,0.1,0.1], "Roughness": 0.3, "Coat Weight": 0.5, "Subsurface Weight": 0.2}.',
            },
            "settings": free_object('Material properties, e.g. {"use_backface_culling": true}.'),
        },
        ["material"],
    ),
    group="materials",
)
def set_principled(material: str, inputs: dict[str, Any] | None = None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Set any Principled BSDF input by its socket name (unlinking it if a texture was connected), and other
    material settings. Unknown names return the list of valid inputs."""
    mat = get_material(material)
    node = principled(mat)
    done = set_inputs(node, inputs or {})
    if settings:
        set_rna(mat, settings, f"Material {mat.name!r}")
    return {"material": mat.name, "inputs_set": done}
