from __future__ import annotations

import math
from collections.abc import Sequence

import bpy
import numpy as np
from mathutils import Matrix, Vector

from .. import state
from ..imaging.png import encode_png
from ..tools.common import world_bounds
from ..tools.registry import ImagePart

MAX_SNAPSHOT_OBJECTS = 60


def _fmt(v: Sequence[float]) -> str:
    return "(" + ", ".join(f"{x:.2f}".rstrip("0").rstrip(".") if abs(x) >= 0.005 else "0" for x in v) + ")"


def scene_snapshot() -> str:
    sc = bpy.context.scene
    objs = list(sc.objects)
    fps = sc.render.fps / sc.render.fps_base
    active = bpy.context.view_layer.objects.active
    camera = sc.camera.name if sc.camera else "none"
    frames = f"frames {sc.frame_start}-{sc.frame_end} at {fps:g} fps (current {sc.frame_current})"
    lines = [f"Blender {bpy.app.version_string}. Scene {sc.name!r}: {len(objs)} objects, {frames}, engine {sc.render.engine}, camera {camera}."]
    selected = [ob.name for ob in bpy.context.selected_objects]
    if selected:
        lines.append("Selected: " + ", ".join(selected[:20]) + (f" (active: {active.name})" if active else ""))
    if objs:
        lines.append("Objects (name, type, location, dimensions, extras):")
    for ob in objs[:MAX_SNAPSHOT_OBJECTS]:
        extras = []
        if ob.parent:
            extras.append(f"parent={ob.parent.name}")
        if ob.material_slots:
            mats = [s.material.name for s in ob.material_slots if s.material]
            if mats:
                extras.append("mats=" + "/".join(mats[:4]))
        if ob.modifiers:
            extras.append("mods=" + "/".join(m.type.lower() for m in ob.modifiers[:5]))
        if ob.type == "MESH":
            extras.append(f"faces={len(ob.data.polygons)}")
        if ob.animation_data and ob.animation_data.action:
            extras.append(f"action={ob.animation_data.action.name}")
        if ob.hide_get():
            extras.append("hidden")
        lines.append(f"- {ob.name} {ob.type.lower()} {_fmt(ob.location)} {_fmt(ob.dimensions)}" + (" " + " ".join(extras) if extras else ""))
    if len(objs) > MAX_SNAPSHOT_OBJECTS:
        lines.append(f"... and {len(objs) - MAX_SNAPSHOT_OBJECTS} more (use get_scene_info).")
    return "\n".join(lines)


def _view3d() -> tuple[bpy.types.Area, bpy.types.SpaceView3D, bpy.types.Region] | None:
    if bpy.app.background:
        return None
    best = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is None:
                continue
            if best is None or area.width * area.height > best[0].width * best[0].height:
                best = (area, area.spaces.active, region)
    return best


def _buffer_array(buffer: object) -> np.ndarray:
    arr = np.asarray(buffer)
    if not arr.flags.c_contiguous:
        row_major = tuple(arr.itemsize * math.prod(arr.shape[i + 1 :]) for i in range(arr.ndim))
        arr = np.lib.stride_tricks.as_strided(arr, strides=row_major, writeable=False)
    return arr


def _draw(space: bpy.types.SpaceView3D, region: bpy.types.Region, view: Matrix, projection: Matrix, width: int, height: int) -> np.ndarray:
    import gpu

    sc = bpy.context.scene
    st = state.get()
    offscreen = gpu.types.GPUOffScreen(width, height)
    try:
        st.capturing = True
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        offscreen.draw_view3d(sc, bpy.context.view_layer, space, region, view, projection, do_color_management=True)
        pixels = _buffer_array(offscreen.texture_color.read()).astype(np.float32)
    finally:
        st.capturing = False
        offscreen.free()
    if pixels.max(initial=0.0) > 1.0:
        pixels /= 255.0
    pixels = pixels.reshape(height, width, 4)[::-1]
    alpha = pixels[:, :, 3:4]
    background = np.array([0.23, 0.23, 0.23], dtype=np.float32)
    return pixels[:, :, :3] * alpha + background * (1.0 - alpha)


def capture_viewport(width: int = 768) -> ImagePart | None:
    found = _view3d()
    if found is None:
        return None
    _area, space, region = found
    rv3d = space.region_3d
    height = max(64, round(width * region.height / max(1, region.width)))
    rgb = _draw(space, region, rv3d.view_matrix, rv3d.window_matrix, width, height)
    return ImagePart(encode_png(rgb))


def turntable(objects: Sequence[bpy.types.Object], views: int = 4, cell: int = 512) -> ImagePart | None:
    found = _view3d()
    if found is None:
        return None
    _area, space, region = found
    lo, hi = world_bounds(objects) if objects else world_bounds(list(bpy.context.scene.objects))
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.05)
    cam_data = bpy.data.cameras.new(".blender_ai_turntable")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(40)
    cam_data.clip_start = radius * 0.01
    cam_data.clip_end = radius * 100
    cam = bpy.data.objects.new(".blender_ai_turntable", cam_data)
    distance = radius / math.sin(cam_data.angle / 2) * 1.1
    elevation = math.radians(20)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tiles = []
    try:
        for i in range(views):
            azimuth = math.radians(-90 + 360 * i / views)
            eye = center + Vector(
                (math.cos(azimuth) * math.cos(elevation) * distance, math.sin(azimuth) * math.cos(elevation) * distance, math.sin(elevation) * distance)
            )
            rotation = (center - eye).to_track_quat("-Z", "Y").to_matrix().to_4x4()
            cam.matrix_world = Matrix.Translation(eye) @ rotation
            projection = cam.calc_matrix_camera(depsgraph, x=cell, y=cell)
            tiles.append(_draw(space, region, cam.matrix_world.inverted(), projection, cell, cell))
    finally:
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(cam_data)
    cols = 2 if views <= 4 else 3
    rows = math.ceil(views / cols)
    sheet = np.full((rows * cell, cols * cell, 3), 0.23, dtype=np.float32)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = tile
    return ImagePart(encode_png(sheet))


def frame_objects(objects: Sequence[bpy.types.Object], margin: float = 2.2) -> bool:
    if not objects:
        return False
    lo, hi = world_bounds(objects)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.05)
    framed = False
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            rv3d = area.spaces.active.region_3d
            if rv3d.view_perspective == "CAMERA":
                continue
            rv3d.view_location = center
            rv3d.view_distance = radius * 2.0 * margin
            area.tag_redraw()
            framed = True
    return framed
