from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any

import bpy
import numpy as np

from .. import state
from ..imaging import client as image_api
from ..imaging import nodes, pbr
from ..imaging.png import downscale, encode_png
from .common import get_image, get_objects
from .materials import principled
from .registry import AsyncJob, ImagePart, ToolError, ToolKind, ToolOutput, boolean, names, number, obj, string, tool


def _prefs() -> Any:
    from ..preferences import get_prefs

    prefs = get_prefs()
    if prefs is None:
        raise ToolError("Blender AI preferences are not available.")
    return prefs


def _params(prefs: Any, normal_strength: float, metallic: float, seamless: bool) -> pbr.PBRParams:
    return replace(pbr.PBRParams(), seamless=seamless, normal_strength=normal_strength, normal_convention=prefs.normal_convention, metallic=metallic)


def _reference_bytes(reference: str) -> bytes:
    if reference == "attached":
        path = state.get().ui.attachment_path
        if not path:
            raise ToolError("No image is attached to the prompt bar.")
        return _png_from_path(path)
    if reference in bpy.data.images:
        return encode_png(nodes.image_to_array(get_image(reference))[:, :, :3])
    path = bpy.path.abspath(reference)
    if os.path.isfile(path):
        return _png_from_path(path)
    raise ToolError(f"Reference {reference!r} is not 'attached', an image name, or an existing file.")


def _png_from_path(path: str) -> bytes:
    if path.lower().endswith(".png"):
        with open(path, "rb") as f:
            return f.read()
    return encode_png(nodes.read_image_file(path)[:, :, :3])


def _reference_array(reference: str) -> np.ndarray:
    if reference == "attached":
        path = state.get().ui.attachment_path
        if not path:
            raise ToolError("No image is attached to the prompt bar.")
        return nodes.read_image_file(path)
    if reference in bpy.data.images:
        return nodes.image_to_array(get_image(reference))
    path = bpy.path.abspath(reference)
    if os.path.isfile(path):
        return nodes.read_image_file(path)
    raise ToolError(f"Image {reference!r} is not 'attached', an image name, or an existing file.")


def _derive_job(pixels: np.ndarray, params: pbr.PBRParams, *, stem: str, material: str, objects: list[str] | None, tiling: float, source_note: str) -> AsyncJob:
    prefs = _prefs()
    folder = prefs.texture_dir()
    pack = prefs.pack_textures

    def work() -> dict[str, Any]:
        maps = pbr.derive(pixels, params)
        paths = {}
        for kind, data in maps.items():
            path = nodes.unique_path(folder, f"{stem}_{kind}")
            with open(path, "wb") as f:
                f.write(encode_png(data))
            paths[kind] = path
        preview = encode_png(downscale(maps["base_color"], 384))
        return {"paths": paths, "preview": preview}

    def finish(result: dict[str, Any]) -> ToolOutput:
        images = {kind: nodes.load_map(path, f"{stem}_{kind}", kind, pack) for kind, path in result["paths"].items()}
        mat = bpy.data.materials.get(material) or bpy.data.materials.new(material)
        principled(mat)
        wired = nodes.build_pbr_material(mat, images, tiling=tiling, normal_strength=params.normal_strength)
        assigned = _assign(mat, objects)
        text = f"Material {mat.name!r} now uses a PBR set ({', '.join(wired)}) {source_note}. Maps saved in {folder}. Tiling {tiling}."
        if assigned:
            text += f" Assigned to {', '.join(assigned)}."
        if objects and any(not getattr(o.data, "uv_layers", True) for o in get_objects(objects)):
            text += " Warning: some objects have no UVs; unwrap them so the textures map correctly."
        return ToolOutput(text, [ImagePart(result["preview"])])

    return AsyncJob("deriving PBR maps", work, finish)


def _assign(mat: bpy.types.Material, objects: list[str] | None) -> list[str]:
    done = []
    for ob in get_objects(objects or []):
        if not hasattr(ob.data, "materials"):
            continue
        if ob.data.materials:
            ob.data.materials[0] = mat
        else:
            ob.data.materials.append(mat)
        done.append(ob.name)
    return done


@tool(
    obj(
        {
            "prompt": string("The surface to make, e.g. 'weathered oak planks with nail holes'."),
            "material": string("Material to build or update (created if needed)."),
            "objects": names("Objects to assign the material to."),
            "reference": string("Optional reference image: 'attached' (the prompt bar attachment), an image name, or a file path."),
            "tiling": number("How many times the texture repeats across the UVs.", minimum=0.01, maximum=100, default=1.0),
            "normal_strength": number("Bump strength of the derived normal map.", minimum=0.0, maximum=10.0, default=1.0),
            "metallic": number("Metallic value for the whole surface.", minimum=0.0, maximum=1.0, default=0.0),
            "seamless": boolean("Blend the edges so the texture tiles without seams.", default=True),
        },
        ["prompt", "material"],
    ),
    group="textures",
    kind=ToolKind.ASYNC,
)
def generate_texture(
    prompt: str,
    material: str,
    objects: list[str] | None = None,
    reference: str | None = None,
    tiling: float = 1.0,
    normal_strength: float = 1.0,
    metallic: float = 0.0,
    seamless: bool = True,
) -> AsyncJob:
    """Generate a photographic, tileable texture with the image model, derive normal, roughness, height,
    AO and metallic maps from it, and wire them into a material. Takes 10-60 seconds. Objects need UVs."""
    prefs = _prefs()
    cfg = prefs.image_config()
    if cfg is None:
        raise ToolError(
            "Image generation isn't set up: with the Anthropic provider the user needs a separate image API in "
            "settings. Use reference_to_pbr with an image they have, or create_material instead."
        )
    if not cfg.model:
        raise ToolError("No image model is selected; the user can pick one on the prompt bar (image button).")
    ref = _reference_bytes(reference) if reference else None
    full_prompt = image_api.texture_prompt(prompt)
    stem = f"{nodes.safe_filename(prompt, 28)}_{time.strftime('%Y%m%d_%H%M%S')}"
    folder = prefs.texture_dir()
    params = _params(prefs, normal_strength, metallic, seamless)
    get_objects(objects or [])

    def work() -> bytes:
        return image_api.generate(cfg, full_prompt, ref)

    def finish(data: bytes) -> AsyncJob:
        source = nodes.unique_path(folder, f"{stem}_source")
        with open(source, "wb") as f:
            f.write(data)
        pixels = nodes.read_image_file(source)
        return _derive_job(
            pixels, params, stem=stem, material=material, objects=objects, tiling=tiling, source_note=f"generated by {cfg.model} from {prompt!r}"
        )

    return AsyncJob("generating image", work, finish)


@tool(
    obj(
        {
            "image": string("Source image: 'attached', an image name in the file, or a file path."),
            "material": string("Material to build or update."),
            "objects": names("Objects to assign the material to."),
            "tiling": number("Texture repeats across the UVs.", minimum=0.01, maximum=100, default=1.0),
            "normal_strength": number("Bump strength.", minimum=0.0, maximum=10.0, default=1.0),
            "metallic": number("Metallic value.", minimum=0.0, maximum=1.0, default=0.0),
            "seamless": boolean("Blend the edges to tile.", default=True),
        },
        ["image", "material"],
    ),
    group="textures",
    kind=ToolKind.ASYNC,
)
def reference_to_pbr(
    image: str, material: str, objects: list[str] | None = None, tiling: float = 1.0, normal_strength: float = 1.0, metallic: float = 0.0, seamless: bool = True
) -> AsyncJob:
    """Turn an image the user already has (attached, in the file, or on disk) into a PBR material, locally
    and without any API call."""
    prefs = _prefs()
    pixels = _reference_array(image)
    stem = f"{nodes.safe_filename(os.path.splitext(os.path.basename(image))[0], 28)}_{time.strftime('%Y%m%d_%H%M%S')}"
    get_objects(objects or [])
    return _derive_job(
        pixels,
        _params(prefs, normal_strength, metallic, seamless),
        stem=stem,
        material=material,
        objects=objects,
        tiling=tiling,
        source_note=f"derived from {image!r}",
    )


@tool(
    obj(
        {
            "material": string("Material to build or update."),
            "base_color": string("Base color map: file path or image name."),
            "roughness": string("Roughness map."),
            "metallic": string("Metallic map."),
            "normal": string("Normal map (OpenGL, or set directx)."),
            "height": string("Height map (used for displacement when displacement > 0)."),
            "ao": string("Ambient occlusion map."),
            "directx_normal": boolean("The normal map is DirectX-style (green flipped).", default=False),
            "objects": names("Objects to assign the material to."),
            "tiling": number("Texture repeats.", minimum=0.01, maximum=100, default=1.0),
            "normal_strength": number("Normal strength.", minimum=0.0, maximum=10.0, default=1.0),
            "displacement": number("Displacement scale in meters (0 disables).", minimum=0.0, default=0.0),
        },
        ["material"],
    ),
    group="textures",
)
def apply_pbr_set(
    material: str,
    base_color: str | None = None,
    roughness: str | None = None,
    metallic: str | None = None,
    normal: str | None = None,
    height: str | None = None,
    ao: str | None = None,
    directx_normal: bool = False,
    objects: list[str] | None = None,
    tiling: float = 1.0,
    normal_strength: float = 1.0,
    displacement: float = 0.0,
) -> str:
    """Build a material from existing PBR map files or images (for example a downloaded texture set)."""
    prefs = _prefs()
    given = {"base_color": base_color, "roughness": roughness, "metallic": metallic, "normal": normal, "height": height, "ao": ao}
    images: dict[str, bpy.types.Image] = {}
    for kind, ref in given.items():
        if not ref:
            continue
        if ref in bpy.data.images:
            img = get_image(ref)
            nodes.set_colorspace(img, nodes.MAP_SPACE[kind])
        else:
            path = bpy.path.abspath(ref)
            if not os.path.isfile(path):
                raise ToolError(f"{kind}: no image named {ref!r} and no file at {path}.")
            if kind == "normal" and directx_normal:
                pixels = nodes.read_image_file(path)
                pixels[:, :, 1] = 1.0 - pixels[:, :, 1]
                path = nodes.unique_path(prefs.texture_dir(), nodes.safe_filename(os.path.basename(path)) + "_opengl")
                with open(path, "wb") as f:
                    f.write(encode_png(pixels[:, :, :3]))
            img = nodes.load_map(path, f"{material}_{kind}", kind, prefs.pack_textures)
        images[kind] = img
    if not images:
        raise ToolError("Give at least one map.")
    mat = bpy.data.materials.get(material) or bpy.data.materials.new(material)
    principled(mat)
    wired = nodes.build_pbr_material(mat, images, tiling=tiling, normal_strength=normal_strength, displacement=displacement)
    assigned = _assign(mat, objects)
    return f"Material {mat.name!r} wired with {', '.join(wired)}" + (f"; assigned to {', '.join(assigned)}" if assigned else "")
