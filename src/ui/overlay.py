from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import bpy
import gpu

from .. import models, state
from ..agent import roles
from ..agent.messages import LogEntry, LogKind, Phase, Status
from . import draw, icons, layout, theme
from . import dropdown as dropdown_mod
from .layout import BarLayout, Rect
from .textedit import Span, caret_line, wrap

COLLAPSED_ROWS = 4


@dataclass
class Row:
    entry_id: int
    text: str
    color: theme.RGBA
    dot: theme.RGBA | None = None
    spinning: bool = False
    suffix: str = ""


def _prefs() -> Any:
    from ..preferences import get_prefs

    return get_prefs()


def _status_color(entry: LogEntry, pal: theme.Palette) -> theme.RGBA:
    return {Status.RUNNING: pal.accent, Status.WAITING: pal.warning, Status.ERROR: pal.error, Status.CANCELLED: pal.text_dim}.get(
        entry.status, theme.mix(pal.log_text, pal.accent, 0.5)
    )


def _wrapped(text: str, size: float, width: float, limit: int) -> list[str]:
    spans = wrap(text, width, lambda s: draw.text_width(s, size))
    lines = [text[s.start : s.end] for s in spans if text[s.start : s.end].strip()]
    if len(lines) > limit:
        lines = [*lines[: limit - 1], draw.ellipsize(" ".join(lines[limit - 1 :]), size, width)]
    return lines


def _who(entry: LogEntry) -> str:
    return entry.agent.split(" · ", 1)[-1] + " › " if entry.agent else ""


def log_rows(entries: list[LogEntry], pal: theme.Palette, size: float, width: float, limit: int) -> list[Row]:
    rows: list[Row] = []
    last_assistant = next((e.id for e in reversed(entries) if e.kind is LogKind.ASSISTANT and not e.agent), None)
    for e in entries[-(limit * 3) :]:
        first = (e.text or "").strip().splitlines()[0] if (e.text or "").strip() else ""
        who = _who(e)
        if e.kind is LogKind.USER:
            rows.append(Row(e.id, "› " + first, pal.text_dim))
        elif e.kind is LogKind.ASSISTANT:
            if not e.text.strip():
                continue
            expanded = e.expanded or e.id == last_assistant
            lines = _wrapped(e.text.strip(), size, width, 4) if expanded else [who + first]
            rows.extend(Row(e.id, line, pal.log_text if not who else pal.text_dim) for line in lines)
        elif e.kind is LogKind.REASONING:
            arrow = "▾" if e.expanded else "▸"
            rows.append(Row(e.id, f"{arrow} {who}Reasoning", pal.text_dim, spinning=e.status is Status.RUNNING))
            if e.expanded and e.detail.strip():
                rows.extend(Row(e.id, line, pal.text_dim) for line in _wrapped(e.detail.strip(), size, width - size, 8))
        elif e.kind is LogKind.TOOL:
            mark = "! " if e.destructive else ""
            rows.append(
                Row(
                    e.id,
                    who + mark + e.text,
                    pal.warning if e.destructive and e.status is Status.OK else pal.log_text,
                    dot=_status_color(e, pal),
                    spinning=e.status is Status.RUNNING,
                    suffix=e.detail.replace("\n", " ") if not e.expanded else "",
                )
            )
            if e.expanded and e.detail:
                rows.extend(Row(e.id, line, pal.text_dim) for line in _wrapped(e.detail, size, width - size, 8))
        elif e.kind is LogKind.ERROR:
            rows.extend(Row(e.id, line, pal.error) for line in _wrapped(e.text, size, width, 3))
        elif e.kind is LogKind.AGENT:
            done = e.status is not Status.RUNNING
            rows.append(
                Row(
                    e.id,
                    who + ("Sub-agent " if not done else "") + e.text,
                    pal.log_text,
                    dot=_status_color(e, pal),
                    spinning=not done,
                    suffix=(e.detail.strip().splitlines() or [""])[0] if not e.expanded else "",
                )
            )
            if e.expanded and e.detail:
                rows.extend(Row(e.id, line, pal.text_dim) for line in _wrapped(e.detail, size, width - size, 10))
        elif e.kind is LogKind.APPROVAL:
            rows.append(Row(e.id, who + "Run this Python? Click to read the code.", pal.warning, dot=pal.warning))
            if e.expanded:
                rows.extend(Row(e.id, line, pal.log_text) for line in e.detail.splitlines()[:10])
        elif e.kind is LogKind.QUESTION:
            color = pal.accent_text if e.status is Status.WAITING else pal.log_text
            rows.append(Row(e.id, "? " + first, color, dot=pal.accent))
            if e.detail:
                rows.append(Row(e.id, e.detail, pal.text_dim))
        else:
            rows.append(Row(e.id, first, pal.text_dim))
    return rows[-limit:]


def _insets(area: bpy.types.Area) -> tuple[float, float]:
    if not bpy.context.preferences.system.use_region_overlap:
        return 0.0, 0.0
    left = right = 0.0
    for region in area.regions:
        if region.width <= 1:
            continue
        if region.type == "TOOLS":
            left = max(left, region.width)
        elif region.type == "UI":
            right = max(right, region.width)
    return left, right


@dataclass
class Labels:
    model: str
    reasoning: str
    image: str | None
    send: str


def _labels(prefs: Any, session: Any) -> Labels:
    st = state.get()
    model = models.short_label(prefs.model(), st.chat_models.models) or "Choose model"
    image = None
    if prefs.image_endpoint() is not None:
        image = models.short_label(prefs.image_model, st.image_models.models) or "Image model"
    if session is not None and session.phase is Phase.QUESTION:
        send = "Answer"
    elif session is not None and session.busy:
        send = "Stop"
    else:
        send = "Generate"
    return Labels(model, "Reasoning: " + models.LEVEL_LABELS.get(prefs.reasoning, "Auto"), image, send)


def _pill_w(label: str, m: layout.Metrics, icon: bool) -> float:
    icon_w = m.small_px * 1.25 if icon else 0.0
    return draw.text_width(label, m.small_px) + m.pad * 2.4 + icon_w + m.small_px * 0.9 + 3 * m.scale


def _tab_widths(m: layout.Metrics) -> list[tuple[str, float, float]]:
    out = []
    icon = m.unit * 0.8
    for mode in roles.ORDER:
        label = roles.info(mode).label
        icon_only = icon + m.unit * 0.7
        out.append((mode.value, icon_only + draw.text_width(label, m.small_px) + m.unit * 0.35, icon_only))
    return out


def _visible_lines(spans: list[Span], caret: int, focused: bool) -> tuple[int, int]:
    if len(spans) <= layout.MAX_TEXT_LINES:
        return 0, len(spans)
    line, _ = caret_line(spans, caret)
    first = max(0, min(line - layout.MAX_TEXT_LINES + 1, len(spans) - layout.MAX_TEXT_LINES)) if focused else len(spans) - layout.MAX_TEXT_LINES
    return first, first + layout.MAX_TEXT_LINES


def draw_bar() -> None:
    st = state.get()
    if st.capturing or not st.registered:
        return
    prefs = _prefs()
    ctx = bpy.context
    area, region = ctx.area, ctx.region
    if prefs is None or not prefs.show_bar or area is None or region is None or region.type != "WINDOW":
        return
    if not st.ready:
        _draw_loading(area, region)
        return
    session = st.session
    ui = st.ui
    m = theme.metrics()
    pal = theme.palette()
    labels = _labels(prefs, session)
    pills = (
        min(_pill_w(labels.model, m, True), m.unit * 8),
        _pill_w(labels.reasoning, m, True),
        min(_pill_w(labels.image, m, True), m.unit * 7) if labels.image else None,
    )
    send_w = draw.text_width(labels.send, m.font_px) + m.font_px * 1.6 + m.unit * 1.4
    left, right = _insets(area)
    busy = session is not None and session.busy
    rows_limit = layout.MAX_LOG_ROWS if ui.expanded_log or ui.focused or busy else COLLAPSED_ROWS
    approval = session is not None and session.phase is Phase.APPROVAL
    common: dict[str, Any] = {
        "left_inset": left,
        "right_inset": right,
        "tab_widths": _tab_widths(m),
        "active_tab": prefs.mode,
        "pill_widths": pills,
        "send_width": send_w,
        "approval": approval,
    }

    base = layout.compute(region.width, region.height, m, **common)
    if base is None:
        st.ui.layouts.pop(region.as_pointer(), None)
        return
    text_w = base.field.w - 2 * m.pad
    buffer = ui.buffer
    spans = wrap(buffer.text, text_w, lambda s: draw.text_width(s, m.font_px)) if buffer.text else [Span(0, 0)]
    rows = log_rows(session.log if session else [], pal, m.small_px, base.card.w - 2 * m.pad - m.unit, rows_limit)
    bar = layout.compute(region.width, region.height, m, text_lines=len(spans), log_rows=len(rows), **common)
    if bar is None:
        return
    shown = rows[len(rows) - len(bar.log_rows) :] if bar.log_rows else []
    bar.log_rows = [(rect, row.entry_id) for (rect, _), row in zip(bar.log_rows, shown, strict=True)]
    st.ui.layouts[region.as_pointer()] = bar

    gpu.state.blend_set("ALPHA")
    try:
        _draw_log(bar, shown, m, pal)
        _draw_tabs(bar, m, pal, prefs.mode)
        _draw_card(bar, m, pal, labels, spans, busy)
        _draw_dropdown(bar, region, m, pal)
    finally:
        gpu.state.blend_set("NONE")


def _shadow(rect: Rect, radius: float, m: layout.Metrics, pal: theme.Palette) -> None:
    draw.soft_shadow(rect, radius, pal.shadow[3], m.unit * 0.35)


def _draw_loading(area: Any, region: Any) -> None:
    st = state.get()
    m = theme.metrics()
    pal = theme.palette()
    left, right = _insets(area)
    bar = layout.compute(region.width, region.height, m, left_inset=left, right_inset=right, pill_widths=(None,) * 3)
    if bar is None:
        return
    gpu.state.blend_set("ALPHA")
    try:
        card = bar.card
        radius = m.radius * 1.6
        _shadow(card, radius, m, pal)
        draw.rounded_rect(card, radius, pal.bar, pal.bar_outline)
        row = Rect(card.x + m.pad * 2, card.cy - m.line_h / 2, card.w - m.pad * 4, m.line_h)
        if st.load_error:
            text, color = f"Blender AI failed to load: {st.load_error}", pal.error
        else:
            draw.spinner(Rect(row.x, row.cy - m.font_px, m.font_px * 2, m.font_px * 2), time.monotonic(), pal.accent)
            text, color = "Loading Blender AI…", pal.card_text_dim
        x = row.x + (0 if st.load_error else m.font_px * 2.4)
        draw.text(x, draw.baseline(row, m.font_px), draw.ellipsize(text, m.font_px, row.right - x), m.font_px, color)
    finally:
        gpu.state.blend_set("NONE")


def _draw_log(bar: BarLayout, rows: list[Row], m: layout.Metrics, pal: theme.Palette) -> None:
    if bar.log_panel is None:
        return
    panel = bar.log_panel
    _shadow(panel, m.radius, m, pal)
    draw.rounded_rect(panel, m.radius, pal.log_bg, pal.bar_outline)
    t = time.monotonic()
    for (rect, _), row in zip(bar.log_rows, rows, strict=True):
        x = rect.x
        if row.spinning:
            draw.spinner(Rect(x, rect.y, rect.h, rect.h), t, row.dot or pal.accent)
            x += rect.h
        elif row.dot is not None:
            draw.status_dot(x + rect.h * 0.4, rect.cy, rect.h * 0.16, row.dot)
            x += rect.h * 0.85
        y = draw.baseline(rect, m.small_px)
        avail = rect.right - x
        label = draw.ellipsize(row.text, m.small_px, avail)
        draw.text(x, y, label, m.small_px, row.color)
        if row.suffix:
            used = draw.text_width(label, m.small_px) + m.pad
            rest = avail - used
            if rest > m.unit * 2:
                draw.text(x + used, y, draw.ellipsize(row.suffix, m.small_px, rest), m.small_px, pal.text_dim)
    if bar.approve and bar.deny:
        for rect, label, fill, fg in ((bar.approve, "Run", pal.accent, pal.accent_text), (bar.deny, "Deny", pal.button, pal.button_text)):
            draw.rounded_rect(rect, m.radius * 0.8, fill, pal.bar_outline)
            draw.text(rect.cx - draw.text_width(label, m.small_px) / 2, draw.baseline(rect, m.small_px), label, m.small_px, fg)


def _draw_tabs(bar: BarLayout, m: layout.Metrics, pal: theme.Palette, active: str) -> None:
    if bar.tab_bar is None:
        return
    strip = bar.tab_bar
    radius = strip.h / 2 * min(1.0, m.radius / (m.unit * 0.4))
    _shadow(strip, radius, m, pal)
    draw.rounded_rect(strip, radius, pal.tab_bg, pal.bar_outline)
    for tab in bar.tabs:
        selected = tab.mode == active
        if selected:
            draw.rounded_rect(tab.rect, radius * 0.8, pal.tab_active)
        fg = pal.tab_active_text if selected else pal.tab_text
        info = roles.info(roles.Mode(tab.mode))
        size = round(min(tab.rect.h * 0.62, m.unit * 0.8))
        x = tab.rect.x + (m.unit * 0.35 if tab.label else (tab.rect.w - size) / 2)
        texture = icons.texture(info.icon)
        if texture is not None:
            draw.texture_quad(Rect(round(x), round(tab.rect.cy - size / 2), size, size), texture, fg)
        if tab.label:
            draw.text(x + size + m.unit * 0.3, draw.baseline(tab.rect, m.small_px), info.label, m.small_px, fg)


def _pill_fill(name: str, pal: theme.Palette) -> theme.RGBA:
    ui = state.get().ui
    if ui.dropdown is not None and ui.dropdown.target == name:
        return pal.pill_hover
    return pal.pill_hover if ui.hover == name else pal.pill


def _draw_pill(rect: Rect | None, label: str | None, m: layout.Metrics, pal: theme.Palette, icon: str, name: str) -> None:
    if rect is None or label is None:
        return
    draw.rounded_rect(rect, m.radius, _pill_fill(name, pal), pal.bar_outline)
    s = m.small_px * 0.42
    ix = rect.x + m.pad * 1.0 + s
    fg = pal.pill_text
    if icon == "ring":
        draw.glyph_ring(ix, rect.cy, s, theme.rgba(fg, 0.85))
    elif icon == "dial":
        draw.glyph_dial(ix, rect.cy, s * 1.1, theme.rgba(fg, 0.85))
    elif icon == "image":
        box = s * 3.2
        draw.glyph_image(Rect(ix - box / 2, rect.cy - box / 2, box, box), theme.rgba(fg, 0.85), False)
    chev = m.small_px * 0.9
    tx = ix + s + m.pad * 0.8
    draw.text(tx, draw.baseline(rect, m.small_px), draw.ellipsize(label, m.small_px, rect.right - tx - chev - m.pad * 0.6), m.small_px, fg)
    draw.glyph_chevron(rect.right - m.pad - chev * 0.35, rect.cy, chev * 0.28, theme.rgba(fg, 0.7))


def _draw_card(bar: BarLayout, m: layout.Metrics, pal: theme.Palette, labels: Labels, spans: list[Span], busy: bool) -> None:
    st = state.get()
    ui = st.ui
    session = st.session
    focused = ui.focused
    card = bar.card
    radius = m.radius * 1.6
    _shadow(card, radius, m, pal)
    draw.rounded_rect(card, radius, pal.bar, pal.focus_outline if focused else pal.bar_outline)

    field = bar.field
    text = ui.buffer.text
    caret_w = max(1.0, m.scale * 1.5)
    if text:
        first, last = _visible_lines(spans, ui.buffer.caret, focused)
        rows = _line_rects(field, last - first, m)
        for span, row in zip(spans[first:last], rows, strict=True):
            draw.text(field.x + m.pad, draw.baseline(row, m.font_px), text[span.start : span.end], m.font_px, pal.card_text)
        if focused and ui.caret_visible:
            line, col = caret_line(spans, ui.buffer.caret)
            if first <= line < last:
                row = rows[line - first]
                span = spans[line]
                cx = field.x + m.pad + draw.text_width(text[span.start : span.start + col], m.font_px)
                draw.rounded_rect(_caret(row, cx, caret_w, m), 0, pal.caret)
    else:
        row = _line_rects(field, 1, m)[0]
        placeholder = draw.ellipsize(_placeholder(session, focused), m.font_px, field.w - 2 * m.pad)
        draw.text(field.x + m.pad, draw.baseline(row, m.font_px), placeholder, m.font_px, pal.card_text_dim)
        if focused and ui.caret_visible:
            draw.rounded_rect(_caret(row, field.x + m.pad, caret_w, m), 0, pal.caret)

    attached = ui.attachment_path is not None
    attach_fill = pal.pill_active if attached else _pill_fill("attach", pal)
    draw.rounded_rect(bar.attach, m.radius, attach_fill, pal.bar_outline)
    if attached:
        draw.glyph_image(bar.attach, pal.pill_active_text, True)
    else:
        draw.glyph_plus(bar.attach, pal.pill_text)
    _draw_pill(bar.model_pill, labels.model, m, pal, "ring", "model")
    _draw_pill(bar.reasoning_pill, labels.reasoning, m, pal, "dial", "reasoning")
    _draw_pill(bar.image_pill, labels.image, m, pal, "image", "image")
    gear_fill = _pill_fill("gear", pal)
    draw.rounded_rect(bar.gear, m.radius, gear_fill, pal.bar_outline)
    draw.glyph_gear(bar.gear, pal.pill_text, gear_fill)

    send = bar.send
    question = session is not None and session.phase is Phase.QUESTION
    ready = bool(text.strip()) or attached
    active = busy or ready
    fill = pal.accent if active else pal.button
    fg = pal.accent_text if active else theme.rgba(pal.button_text, 0.45)
    draw.rounded_rect(send, radius, fill)
    label_w = draw.text_width(labels.send, m.font_px)
    mark = m.font_px * 0.55
    total = label_w + m.font_px * 0.5 + mark * 2
    x = send.cx - total / 2
    y = draw.baseline(send, m.font_px)
    draw.text(x, y, labels.send, m.font_px, fg)
    mx = x + label_w + m.font_px * 0.5 + mark
    if busy and not question:
        draw.spinner(Rect(mx - mark * 1.6, send.cy - mark * 1.6, mark * 3.2, mark * 3.2), time.monotonic(), fg)
        draw.glyph_stop(Rect(mx - mark * 1.6, send.cy - mark * 1.6, mark * 3.2, mark * 3.2), fg)
    else:
        draw.glyph_sparkle(mx, send.cy, mark, fg)


def dropdown_width(target: str, m: layout.Metrics) -> float:
    return m.unit * (17 if target in ("model", "image") else 9)


def _draw_dropdown(bar: BarLayout, region: Any, m: layout.Metrics, pal: theme.Palette) -> None:
    ui = state.get().ui
    dd = ui.dropdown
    if dd is None or ui.dropdown_region != region.as_pointer():
        return
    anchor = {"model": bar.model_pill, "image": bar.image_pill, "reasoning": bar.reasoning_pill}.get(dd.target)
    if anchor is None:
        ui.dropdown = None
        return
    lay = dropdown_mod.compute(dd, anchor, region.width, region.height, m, dropdown_width(dd.target, m))
    ui.dropdown_layout = lay
    panel = lay.panel
    _shadow(panel, m.radius, m, pal)
    draw.rounded_rect(panel, m.radius, pal.menu_bg, pal.menu_outline)
    items = dd.visible()
    if lay.search is not None:
        _draw_search(lay.search, dd, m, pal)
    if not items:
        rect = Rect(panel.x + m.pad, panel.y + round(m.unit * 0.25), panel.w - 2 * m.pad, round(m.unit * 1.1))
        draw.text(rect.x + m.pad, draw.baseline(rect, m.small_px), "No matches", m.small_px, pal.menu_text_dim)
    check_w = m.small_px * 1.4
    for rect, index in lay.rows:
        item = items[index]
        hovered = index == dd.hover and item.enabled
        if hovered:
            draw.rounded_rect(rect, m.radius * 0.7, pal.menu_item_sel)
        fg = pal.menu_item_text_sel if hovered else (pal.menu_text if item.enabled else pal.menu_text_dim)
        y = draw.baseline(rect, m.small_px)
        if item.checked:
            draw.text(rect.x + m.pad * 0.6, y, "✓", m.small_px, fg)
        x = rect.x + check_w
        hint_w = draw.text_width(item.hint, m.small_px) + m.pad if item.hint else 0.0
        draw.text(x, y, draw.ellipsize(item.label, m.small_px, rect.right - x - hint_w - m.pad), m.small_px, fg)
        if item.hint:
            draw.text(rect.right - hint_w, y, item.hint, m.small_px, theme.rgba(fg, 0.6))
    if lay.more_above or lay.more_below:
        _draw_scrollbar(lay, dd, len(items), m, pal)


def _draw_search(field: Rect, dd: dropdown_mod.Dropdown, m: layout.Metrics, pal: theme.Palette) -> None:
    draw.rounded_rect(field, m.radius * 0.8, pal.field, pal.menu_outline)
    y = draw.baseline(field, m.small_px)
    x = field.x + m.pad
    if dd.query:
        shown = draw.ellipsize(dd.query, m.small_px, field.w - 2 * m.pad)
        draw.text(x, y, shown, m.small_px, pal.field_text)
        x += draw.text_width(shown, m.small_px)
    else:
        count = sum(1 for i in dd.items if i.enabled and not i.id.startswith("__"))
        hint = f"Type to search {count} models…"
        draw.text(x, y, draw.ellipsize(hint, m.small_px, field.w - 2 * m.pad), m.small_px, pal.menu_text_dim)
    if state.get().ui.caret_visible:
        caret = Rect(x + 1, field.cy - m.small_px * 0.6, max(1.0, m.scale * 1.5), m.small_px * 1.2)
        draw.rounded_rect(caret, 0, pal.caret)


def _draw_scrollbar(lay: dropdown_mod.DropdownLayout, dd: dropdown_mod.Dropdown, total: int, m: layout.Metrics, pal: theme.Palette) -> None:
    panel = lay.panel
    top_limit = lay.search.y - m.pad * 0.5 if lay.search is not None else panel.top - m.pad
    track = Rect(panel.right - m.scale * 4, panel.y + m.pad, m.scale * 2, top_limit - panel.y - m.pad)
    h = max(m.unit * 0.6, track.h * lay.row_count / max(1, total))
    top = track.top - (track.h - h) * dd.scroll / max(1, total - lay.row_count)
    draw.rounded_rect(Rect(track.x, top - h, track.w, h), track.w / 2, theme.rgba(pal.menu_text, 0.35))


def _line_rects(field: Rect, count: int, m: layout.Metrics) -> list[Rect]:
    top = field.top - round(m.unit * 0.1)
    return [Rect(field.x, top - (i + 1) * m.line_h, field.w, m.line_h) for i in range(max(1, count))]


def _caret(row: Rect, x: float, width: float, m: layout.Metrics) -> Rect:
    h = m.font_px * 1.25
    return Rect(x, row.cy - h / 2, width, h)


def _placeholder(session: Any, focused: bool) -> str:
    if session is not None and session.phase is Phase.QUESTION:
        return "Type your answer and press Enter"
    if session is not None and session.phase is Phase.APPROVAL:
        return "Waiting for your answer in the dialog: run this Python code?"
    if session is not None and session.busy:
        return "Working… press Esc or Stop to interrupt"
    if focused:
        return "Describe what to make · Enter to generate · Shift+Enter new line · Tab switches tabs"
    return "Describe what you want to create…   (Ctrl+Space)"


def region_at(context: Any, mouse_x: int, mouse_y: int) -> tuple[Any, Any] | None:
    window = context.window
    if window is None:
        return None
    for area in window.screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type == "WINDOW" and region.x <= mouse_x < region.x + region.width and region.y <= mouse_y < region.y + region.height:
                return area, region
    return None


def hit_test(context: Any, event: Any) -> tuple[str | None, Any, Any]:
    found = region_at(context, event.mouse_x, event.mouse_y)
    if found is None:
        return None, None, None
    area, region = found
    bar = state.get().ui.layouts.get(region.as_pointer())
    return (None, area, region) if bar is None else (bar.hit(event.mouse_x - region.x, event.mouse_y - region.y), area, region)


def register() -> None:
    st = state.get()
    handle = bpy.types.SpaceView3D.draw_handler_add(draw_bar, (), "WINDOW", "POST_PIXEL")
    st.draw_handles.append((bpy.types.SpaceView3D, handle))


def unregister() -> None:
    st = state.get()
    for space, handle in st.draw_handles:
        space.draw_handler_remove(handle, "WINDOW")
    st.draw_handles.clear()
