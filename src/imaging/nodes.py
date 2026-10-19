from __future__ import annotations

import os
from typing import Any

import bpy
import numpy as np

PREFIX = "AI "

_SPACES = {"color": ("sRGB",), "data": ("Non-Color", "Generic Data", "Raw", "Linear")}
MAP_SPACE = {"base_color": "color", "roughness": "data", "metallic": "data", "normal": "data", "height": "data", "ao": "data"}


def set_colorspace(image: bpy.types.Image, kind: str) -> None:
    for name in _SPACES[kind]:
        try:
            image.colorspace_settings.name = name
            return
        except TypeError:
            continue


def load_map(path: str, name: str, kind: str, pack: bool = False) -> bpy.types.Image:
    old = bpy.data.images.get(name)
    if old is not None and old.users == 0:
        bpy.data.images.remove(old)
    image = bpy.data.images.load(path, check_existing=False)
    image.name = name
    set_colorspace(image, MAP_SPACE.get(kind, "data"))
    if pack:
        image.pack()
    return image


def image_to_array(image: bpy.types.Image) -> np.ndarray:
    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError(f"image {image.name!r} has no pixels (is the file missing?)")
    buf = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    return np.ascontiguousarray(buf.reshape(h, w, 4)[::-1])


def read_image_file(path: str) -> np.ndarray:
    image = bpy.data.images.load(path, check_existing=False)
    try:
        return image_to_array(image)
    finally:
        bpy.data.images.remove(image)


def build_pbr_material(
    mat: bpy.types.Material,
    maps: dict[str, bpy.types.Image],
    *,
    tiling: float = 1.0,
    normal_strength: float = 1.0,
    use_ao: bool = True,
    displacement: float = 0.0,
) -> list[str]:
    mat.use_nodes = True
    tree = mat.node_tree
    nodes, links = tree.nodes, tree.links
    for node in [n for n in nodes if n.name.startswith(PREFIX)]:
        nodes.remove(node)

    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
    bsdf.location, out.location = (200, 300), (550, 300)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def node(kind: str, label: str, x: float, y: float) -> Any:
        n = nodes.new(kind)
        n.name = n.label = PREFIX + label
        n.location = (x, y)
        return n

    coord = node("ShaderNodeTexCoord", "Coordinates", -1100, 300)
    mapping = node("ShaderNodeMapping", "Mapping", -900, 300)
    mapping.inputs["Scale"].default_value = (tiling, tiling, tiling)
    links.new(coord.outputs["UV"], mapping.inputs["Vector"])

    wired: list[str] = []
    textures: dict[str, Any] = {}
    y = 650
    for kind in ("base_color", "ao", "metallic", "roughness", "normal", "height"):
        image = maps.get(kind)
        if image is None:
            continue
        tex = node("ShaderNodeTexImage", kind.replace("_", " ").title(), -600, y)
        tex.image = image
        tex.interpolation = "Linear"
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        textures[kind] = tex
        y -= 280

    if "base_color" in textures:
        color_out = textures["base_color"].outputs["Color"]
        if use_ao and "ao" in textures:
            mix = node("ShaderNodeMix", "AO Multiply", -250, 600)
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            a = next(s for s in mix.inputs if s.name == "A" and s.type == "RGBA")
            b = next(s for s in mix.inputs if s.name == "B" and s.type == "RGBA")
            links.new(color_out, a)
            links.new(textures["ao"].outputs["Color"], b)
            color_out = next(s for s in mix.outputs if s.type == "RGBA")
            wired.append("ao")
        links.new(color_out, bsdf.inputs["Base Color"])
        wired.append("base_color")
    if "metallic" in textures:
        links.new(textures["metallic"].outputs["Color"], bsdf.inputs["Metallic"])
        wired.append("metallic")
    if "roughness" in textures:
        links.new(textures["roughness"].outputs["Color"], bsdf.inputs["Roughness"])
        wired.append("roughness")
    if "normal" in textures:
        nmap = node("ShaderNodeNormalMap", "Normal Map", -250, 0)
        nmap.inputs["Strength"].default_value = normal_strength
        links.new(textures["normal"].outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        wired.append("normal")
    if "height" in textures and displacement > 0:
        disp = node("ShaderNodeDisplacement", "Displacement", 200, -250)
        disp.inputs["Scale"].default_value = displacement
        disp.inputs["Midlevel"].default_value = 0.5
        links.new(textures["height"].outputs["Color"], disp.inputs["Height"])
        links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
        if hasattr(mat, "displacement_method"):
            mat.displacement_method = "BOTH"
        wired.append("height")
    return wired


def safe_filename(text: str, limit: int = 40) -> str:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug or "texture")[:limit]


def unique_path(folder: str, stem: str, ext: str = ".png") -> str:
    path = os.path.join(folder, stem + ext)
    i = 1
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{i}{ext}")
        i += 1
    return path
