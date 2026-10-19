from __future__ import annotations

import os
import tempfile
from typing import Any

import bpy

from ..agent import context
from .common import get_object, get_objects, scene
from .registry import ImagePart, ToolError, ToolOutput, array, boolean, integer, names, number, obj, string, tool

ENGINES = {"eevee": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"), "cycles": ("CYCLES",), "workbench": ("BLENDER_WORKBENCH",)}


def _engine_id(name: str) -> str:
    render = scene().render
    current = render.engine
    try:
        for candidate in ENGINES[name]:
            try:
                render.engine = candidate
            except TypeError:
                continue
            return candidate
    finally:
        render.engine = current
    raise ToolError(f"Render engine {name} isn't available.")


@tool(
    obj(
        {
            "engine": string("Render engine.", enum=list(ENGINES)),
            "resolution": array({"type": "integer", "minimum": 16, "maximum": 16384}, "Width and height in pixels.", min_items=2, max_items=2),
            "percentage": integer("Resolution scale.", minimum=1, maximum=100),
            "samples": integer("Render samples (Cycles or EEVEE).", minimum=1, maximum=16384),
            "film_transparent": boolean("Transparent background."),
            "view_transform": string("Color management view, e.g. AgX, Filmic, Standard."),
            "look": string("Color management look, e.g. 'AgX - Punchy'."),
            "exposure": number("Exposure in stops.", minimum=-10, maximum=10),
            "output_path": string("Output path for renders and animations."),
        }
    ),
    group="render",
)
def set_render(
    engine: str | None = None,
    resolution: list[int] | None = None,
    percentage: int | None = None,
    samples: int | None = None,
    film_transparent: bool | None = None,
    view_transform: str | None = None,
    look: str | None = None,
    exposure: float | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Change render engine, resolution, samples, transparency, color management and output path."""
    sc = scene()
    r = sc.render
    if engine:
        r.engine = _engine_id(engine)
    if resolution:
        r.resolution_x, r.resolution_y = resolution
    if percentage:
        r.resolution_percentage = percentage
    if samples:
        if r.engine == "CYCLES":
            sc.cycles.samples = samples
        elif hasattr(sc.eevee, "taa_render_samples"):
            sc.eevee.taa_render_samples = samples
    if film_transparent is not None:
        r.film_transparent = film_transparent
    vs = sc.view_settings
    try:
        if view_transform:
            vs.view_transform = view_transform
        if look:
            vs.look = look
    except TypeError as exc:
        raise ToolError(f"Color management: {exc}") from exc
    if exposure is not None:
        vs.exposure = exposure
    if output_path:
        r.filepath = output_path
    return {
        "engine": r.engine,
        "resolution": [r.resolution_x, r.resolution_y, r.resolution_percentage],
        "film_transparent": r.film_transparent,
        "view_transform": vs.view_transform,
        "look": vs.look,
    }


@tool(
    obj({"names": names("Objects to frame (default: everything)."), "through_camera": boolean("Look through the active camera instead.", default=False)}),
    group="render",
    mutates=False,
)
def frame_view(names: list[str] | None = None, through_camera: bool = False) -> str:
    """Point the user's 3D Viewport at some objects (or through the scene camera)."""
    if through_camera:
        if scene().camera is None:
            raise ToolError("The scene has no active camera.")
        found = False
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.region_3d.view_perspective = "CAMERA"
                    area.tag_redraw()
                    found = True
        if not found:
            raise ToolError("No 3D Viewport is open.")
        return "Viewing through the camera"
    objs = get_objects(names) if names else [o for o in scene().objects if o.visible_get()]
    if not context.frame_objects(objs):
        raise ToolError("No 3D Viewport is open (or every viewport is looking through a camera).")
    return f"Framed {len(objs)} object(s)"


@tool(obj({"width": integer("Image width in pixels.", minimum=128, maximum=1568, default=768)}), group="render", mutates=False)
def capture_viewport(width: int = 768) -> ToolOutput:
    """Take a screenshot of the 3D Viewport as the user sees it, to check your work visually."""
    image = context.capture_viewport(width)
    if image is None:
        raise ToolError("No 3D Viewport is open.")
    return ToolOutput("Viewport screenshot attached.", [image])


@tool(
    obj(
        {
            "width": integer("Preview width in pixels.", minimum=128, maximum=1568, default=768),
            "engine": string("Engine for the preview; 'current' keeps the scene's.", enum=["current", *ENGINES], default="current"),
            "samples": integer("Samples for the preview (low is fine).", minimum=1, maximum=4096),
            "camera": string("Render from this camera instead of the active one."),
        }
    ),
    group="render",
    mutates=False,
)
def render_preview(width: int = 768, engine: str = "current", samples: int | None = None, camera: str | None = None) -> ToolOutput:
    """Render a still from the scene camera at preview size and return the image. Use to check lighting and
    composition in Scene mode. Blocks Blender while it renders."""
    sc = scene()
    r = sc.render
    cam = get_object(camera, ("CAMERA",)) if camera else sc.camera
    if cam is None:
        raise ToolError("The scene has no active camera; add one with add_camera.")
    saved: dict[str, Any] = {
        "engine": r.engine,
        "pct": r.resolution_percentage,
        "path": r.filepath,
        "format": r.image_settings.file_format,
        "camera": sc.camera,
    }
    saved_samples = None
    out = os.path.join(tempfile.gettempdir(), f"blender_ai_preview_{os.getpid()}.png")
    try:
        if engine != "current":
            r.engine = _engine_id(engine)
        if r.engine == "CYCLES":
            saved_samples = ("cycles", sc.cycles.samples)
            sc.cycles.samples = samples or 32
        elif hasattr(sc.eevee, "taa_render_samples") and r.engine != "BLENDER_WORKBENCH":
            saved_samples = ("eevee", sc.eevee.taa_render_samples)
            sc.eevee.taa_render_samples = samples or 16
        r.resolution_percentage = max(1, min(100, round(width / max(1, r.resolution_x) * 100)))
        r.filepath = out
        r.image_settings.file_format = "PNG"
        sc.camera = cam
        bpy.ops.render.render(write_still=True)
        with open(out, "rb") as f:
            data = f.read()
    finally:
        r.engine = saved["engine"]
        r.resolution_percentage = saved["pct"]
        r.filepath = saved["path"]
        r.image_settings.file_format = saved["format"]
        sc.camera = saved["camera"]
        if saved_samples:
            target = sc.cycles if saved_samples[0] == "cycles" else sc.eevee
            setattr(target, "samples" if saved_samples[0] == "cycles" else "taa_render_samples", saved_samples[1])
    return ToolOutput(f"Preview render from {cam.name} ({r.engine}).", [ImagePart(data)])


@tool(
    obj(
        {
            "names": names("Objects to orbit around (default: everything visible)."),
            "views": integer("Number of views.", minimum=2, maximum=9, default=4),
            "cell": integer("Pixel size of each view.", minimum=128, maximum=768, default=384),
        }
    ),
    group="render",
    mutates=False,
)
def render_turntable(names: list[str] | None = None, views: int = 4, cell: int = 384) -> ToolOutput:
    """Render the asset from several angles around it (viewport shading) as one contact sheet, to check it
    from every side before finishing."""
    objs = get_objects(names) if names else [o for o in scene().objects if o.visible_get() and o.type in {"MESH", "CURVE", "FONT", "ARMATURE"}]
    image = context.turntable(objs, views, cell)
    if image is None:
        raise ToolError("No 3D Viewport is open.")
    return ToolOutput(f"Turntable of {len(objs)} object(s), {views} views (front, right, back, left...).", [image])
