from __future__ import annotations

import math
from collections.abc import Sequence

import blf
import gpu
from gpu_extras.batch import batch_for_shader

from .layout import Rect
from .theme import RGBA

FONT = 0
Point = tuple[float, float]


def _shader() -> gpu.types.GPUShader:
    gpu.state.blend_set("ALPHA")
    return gpu.shader.from_builtin("UNIFORM_COLOR")


def _fan(points: Sequence[Point]) -> list[Point]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    tris: list[Point] = []
    for i in range(len(points)):
        tris += [(cx, cy), points[i], points[(i + 1) % len(points)]]
    return tris


def polygon(points: Sequence[Point], color: RGBA) -> None:
    if len(points) < 3:
        return
    shader = _shader()
    batch = batch_for_shader(shader, "TRIS", {"pos": _fan(points)})
    shader.uniform_float("color", color)
    batch.draw(shader)


def triangles(tris: Sequence[Point], color: RGBA) -> None:
    shader = _shader()
    batch = batch_for_shader(shader, "TRIS", {"pos": list(tris)})
    shader.uniform_float("color", color)
    batch.draw(shader)


def outline(points: Sequence[Point], color: RGBA, width: float = 1.0) -> None:
    gpu.state.blend_set("ALPHA")
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    pts = [*points, points[0]]
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": [(x, y, 0.0) for x, y in pts]})
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set("ALPHA")


def rounded(rect: Rect, radius: float, segments: int = 6) -> list[Point]:
    r = max(0.0, min(radius, rect.w / 2, rect.h / 2))
    if r < 0.5:
        return [(rect.x, rect.y), (rect.right, rect.y), (rect.right, rect.top), (rect.x, rect.top)]
    corners = (
        (rect.right - r, rect.y + r, -math.pi / 2),
        (rect.right - r, rect.top - r, 0.0),
        (rect.x + r, rect.top - r, math.pi / 2),
        (rect.x + r, rect.y + r, math.pi),
    )
    pts: list[Point] = []
    for cx, cy, start in corners:
        for i in range(segments + 1):
            a = start + (math.pi / 2) * i / segments
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def rounded_rect(rect: Rect, radius: float, fill: RGBA, border: RGBA | None = None, width: float = 1.0) -> None:
    pts = rounded(rect, radius)
    polygon(pts, fill)
    if border is not None and border[3] > 0.01:
        outline(pts, border, width)


def soft_shadow(rect: Rect, radius: float, alpha: float, spread: float) -> None:
    steps = max(2, round(spread))
    if alpha <= 0.0:
        return
    for i in range(steps, 0, -1):
        t = i / steps
        a = alpha * (1.0 - t) ** 2 * 2.2 / steps
        grown = Rect(rect.x - i, rect.y - i * 1.15, rect.w + 2 * i, rect.h + 2 * i)
        outline(rounded(grown, radius + i), (0.0, 0.0, 0.0, a), 1.5)


def circle(cx: float, cy: float, r: float, color: RGBA, segments: int = 24) -> None:
    polygon([(cx + r * math.cos(2 * math.pi * i / segments), cy + r * math.sin(2 * math.pi * i / segments)) for i in range(segments)], color)


def text_width(s: str, size: float) -> float:
    blf.size(FONT, size)
    return blf.dimensions(FONT, s)[0]


def text(x: float, y: float, s: str, size: float, color: RGBA, clip: Rect | None = None) -> None:
    blf.size(FONT, size)
    blf.color(FONT, *color)
    if clip is not None:
        blf.enable(FONT, blf.CLIPPING)
        blf.clipping(FONT, clip.x, clip.y, clip.right, clip.top)
    blf.position(FONT, round(x), round(y), 0)
    blf.draw(FONT, s)
    if clip is not None:
        blf.disable(FONT, blf.CLIPPING)
    gpu.state.blend_set("ALPHA")


def ellipsize(s: str, size: float, width: float) -> str:
    if text_width(s, size) <= width:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_width(s[:mid] + "…", size) <= width:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo] + "…"


def baseline(rect: Rect, size: float) -> float:
    blf.size(FONT, size)
    cap = blf.dimensions(FONT, "M")[1]
    return rect.y + (rect.h - cap) / 2


def glyph_send(rect: Rect, color: RGBA) -> None:
    s = rect.w * 0.26
    cx, cy = rect.cx, rect.cy
    triangles([(cx - s, cy - s * 0.15), (cx + s, cy - s * 0.15), (cx, cy + s * 1.05)], color)
    w = s * 0.32
    polygon([(cx - w, cy - s * 1.05), (cx + w, cy - s * 1.05), (cx + w, cy - s * 0.1), (cx - w, cy - s * 0.1)], color)


def glyph_stop(rect: Rect, color: RGBA) -> None:
    s = rect.w * 0.2
    rounded_rect(Rect(rect.cx - s, rect.cy - s, 2 * s, 2 * s), s * 0.3, color)


def glyph_gear(rect: Rect, color: RGBA, hole: RGBA) -> None:
    cx, cy = rect.cx, rect.cy
    outer, inner = rect.w * 0.3, rect.w * 0.22
    teeth = 8
    pts: list[Point] = []
    for i in range(teeth * 4):
        a = 2 * math.pi * i / (teeth * 4)
        r = outer if (i % 4) in (0, 1) else inner
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    polygon(pts, color)
    circle(cx, cy, rect.w * 0.1, hole, 16)


def glyph_image(rect: Rect, color: RGBA, active: bool) -> None:
    s = rect.w * 0.28
    frame = Rect(rect.cx - s, rect.cy - s * 0.8, 2 * s, 1.6 * s)
    outline(rounded(frame, s * 0.15, 3), color, 1.2)
    triangles(
        [
            (frame.x + s * 0.25, frame.y + s * 0.25),
            (frame.x + s * 0.95, frame.y + s * 0.25),
            (frame.x + s * 0.6, frame.y + s * 0.95),
            (frame.x + s * 0.75, frame.y + s * 0.25),
            (frame.right - s * 0.25, frame.y + s * 0.25),
            (frame.right - s * 0.6, frame.y + s * 0.75),
        ],
        color,
    )
    circle(frame.right - s * 0.5, frame.top - s * 0.45, s * 0.16, color, 10)
    if active:
        circle(rect.right - rect.w * 0.18, rect.top - rect.h * 0.18, rect.w * 0.09, color, 10)


def glyph_plus(rect: Rect, color: RGBA) -> None:
    s, w = rect.w * 0.24, max(1.0, rect.w * 0.045)
    cx, cy = rect.cx, rect.cy
    polygon([(cx - s, cy - w), (cx + s, cy - w), (cx + s, cy + w), (cx - s, cy + w)], color)
    polygon([(cx - w, cy - s), (cx + w, cy - s), (cx + w, cy + s), (cx - w, cy + s)], color)


def glyph_sparkle(cx: float, cy: float, r: float, color: RGBA) -> None:
    k = r * 0.3
    polygon([(cx, cy + r), (cx + k, cy + k), (cx + r, cy), (cx + k, cy - k), (cx, cy - r), (cx - k, cy - k), (cx - r, cy), (cx - k, cy + k)], color)


def glyph_ring(cx: float, cy: float, r: float, color: RGBA) -> None:
    outline(rounded(Rect(cx - r, cy - r, 2 * r, 2 * r), r, 5), color, max(1.0, r * 0.28))


def glyph_dial(cx: float, cy: float, r: float, color: RGBA) -> None:
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in (math.pi * (i / 10) for i in range(11))]
    gpu.state.blend_set("ALPHA")
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": [(x, y - r * 0.35, 0.0) for x, y in pts]})
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", max(1.0, r * 0.28))
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set("ALPHA")
    a = math.radians(55)
    tip = (cx + r * 0.8 * math.cos(a), cy - r * 0.35 + r * 0.8 * math.sin(a))
    w = r * 0.14
    base = (cx, cy - r * 0.35)
    nx, ny = -math.sin(a) * w, math.cos(a) * w
    polygon([(base[0] + nx, base[1] + ny), (tip[0], tip[1]), (base[0] - nx, base[1] - ny)], color)
    circle(base[0], base[1], w * 1.4, color, 10)


def glyph_chevron(cx: float, cy: float, s: float, color: RGBA) -> None:
    triangles([(cx - s, cy + s * 0.4), (cx + s, cy + s * 0.4), (cx, cy - s * 0.6)], color)


def spinner(rect: Rect, t: float, color: RGBA) -> None:
    cx, cy = rect.cx, rect.cy
    r = rect.w * 0.3
    for i in range(8):
        a = -t * 2 * math.pi + 2 * math.pi * i / 8
        alpha = color[3] * (0.25 + 0.75 * ((i + 1) / 8))
        circle(cx + r * math.cos(a), cy + r * math.sin(a), rect.w * 0.06, (color[0], color[1], color[2], alpha), 8)


def status_dot(cx: float, cy: float, r: float, color: RGBA) -> None:
    circle(cx, cy, r, color, 12)


def texture_quad(rect: Rect, texture: gpu.types.GPUTexture, color: RGBA) -> None:
    shader = gpu.shader.from_builtin("IMAGE_COLOR")
    batch = batch_for_shader(
        shader,
        "TRIS",
        {
            "pos": [(rect.x, rect.y), (rect.right, rect.y), (rect.right, rect.top), (rect.x, rect.y), (rect.right, rect.top), (rect.x, rect.top)],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)],
        },
    )
    shader.uniform_sampler("image", texture)
    shader.uniform_float("color", color)
    gpu.state.blend_set("ALPHA")
    batch.draw(shader)
