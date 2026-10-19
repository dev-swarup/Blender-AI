from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

MAX_TEXT_LINES = 6
MIN_TEXT_LINES = 2


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.top

    def inset(self, d: float) -> Rect:
        return Rect(self.x + d, self.y + d, max(0.0, self.w - 2 * d), max(0.0, self.h - 2 * d))


@dataclass(frozen=True)
class Metrics:
    scale: float
    unit: float
    font_px: float
    small_px: float
    line_h: float
    radius: float
    pad: float

    @classmethod
    def from_scale(cls, scale: float, font_points: float = 11.0, roundness: float = 0.4) -> Metrics:
        scale = max(0.5, scale)
        unit = 20.0 * scale
        font_px = font_points * scale
        return cls(
            scale=scale,
            unit=unit,
            font_px=font_px,
            small_px=max(8.0, font_px * 0.9),
            line_h=round(font_px * 1.45),
            radius=max(2.0, min(0.5, roundness) * unit * 0.9),
            pad=round(unit * 0.3),
        )


@dataclass(frozen=True)
class Tab:
    rect: Rect
    mode: str
    label: bool


@dataclass
class BarLayout:
    card: Rect
    field: Rect
    send: Rect
    attach: Rect
    gear: Rect
    model_pill: Rect | None = None
    reasoning_pill: Rect | None = None
    image_pill: Rect | None = None
    tab_bar: Rect | None = None
    tabs: list[Tab] = field(default_factory=list)
    log_panel: Rect | None = None
    log_header: Rect | None = None
    log_view: Rect | None = None
    log_latest: Rect | None = None
    log_rows: list[tuple[Rect, int]] = field(default_factory=list)
    approve: Rect | None = None
    deny: Rect | None = None
    lines: int = MIN_TEXT_LINES

    def hit(self, px: float, py: float) -> str | None:
        for name, rect in (
            ("approve", self.approve),
            ("deny", self.deny),
            ("log_latest", self.log_latest),
            ("send", self.send),
            ("attach", self.attach),
            ("gear", self.gear),
            ("model", self.model_pill),
            ("reasoning", self.reasoning_pill),
            ("image", self.image_pill),
            ("field", self.field),
        ):
            if rect is not None and rect.contains(px, py):
                return name
        for tab in self.tabs:
            if tab.rect.contains(px, py):
                return f"tab:{tab.mode}"
        for rect, entry_id in self.log_rows:
            if rect.contains(px, py):
                return f"log:{entry_id}"
        if self.log_panel is not None and self.log_panel.contains(px, py):
            return "log"
        if self.card.contains(px, py):
            return "field"
        return "tabs" if self.tab_bar is not None and self.tab_bar.contains(px, py) else None


def _tabs(card: Rect, m: Metrics, widths: Sequence[tuple[str, float, float]], active: str) -> tuple[Rect, list[Tab]]:
    u = m.unit
    h = round(u * 1.5)
    inner = round(u * 0.2)
    gap = round(u * 0.15)
    for mode_labels in ("all", "active", "none"):
        sizes = []
        for mode, full, icon_only in widths:
            show = mode_labels == "all" or (mode_labels == "active" and mode == active)
            sizes.append((mode, full if show else icon_only, show))
        total = sum(w for _, w, _ in sizes) + gap * (len(sizes) - 1) + 2 * inner
        if total <= card.w or mode_labels == "none":
            break
    bar = Rect(round(card.cx - total / 2), card.top + round(u * 0.4), round(total), h)
    tabs: list[Tab] = []
    x = bar.x + inner
    for mode, w, show in sizes:
        tabs.append(Tab(Rect(round(x), bar.y + inner, round(w), h - 2 * inner), mode, show))
        x += w + gap
    return bar, tabs


def compute(
    region_w: float,
    region_h: float,
    m: Metrics,
    *,
    left_inset: float = 0.0,
    right_inset: float = 0.0,
    bottom_inset: float = 0.0,
    tab_widths: Sequence[tuple[str, float, float]] = (),
    active_tab: str = "",
    pill_widths: tuple[float | None, float | None, float | None] = (90.0, 90.0, 90.0),
    send_width: float | None = None,
    text_lines: int = 1,
    log_height: float = 0.0,
    log_header: bool = False,
    approval: bool = False,
) -> BarLayout | None:
    u = m.unit
    avail_w = region_w - left_inset - right_inset - 2 * u
    if avail_w < 14 * u or region_h < 8 * u:
        return None

    p = round(u * 0.45)
    ctl_h = round(u * 1.25)
    gap = round(u * 0.3)
    sep = round(p * 0.7)
    lines = max(MIN_TEXT_LINES, min(MAX_TEXT_LINES, text_lines))
    text_h = lines * m.line_h + round(u * 0.2)
    base_h = p + ctl_h + sep + text_h + p
    card_w = round(min(avail_w, 36 * u))
    cx = left_inset + (region_w - left_inset - right_inset) / 2
    x, y = round(cx - card_w / 2), round(bottom_inset + u * 1.0)

    header_h = round(m.small_px * 2.0) if log_header else 0
    action_h = round(u * 1.1) if approval else 0
    tabs_h = round(u * 1.9) if tab_widths else 0
    room = region_h - y - base_h - tabs_h - u - sep - header_h - action_h
    view_h = round(max(0.0, min(log_height, room)))
    if view_h < m.line_h and not approval:
        view_h = 0
    section = sep + action_h + view_h + header_h if view_h or approval else 0
    card = Rect(x, y, card_w, base_h + section)

    send_w = round(max(send_width or 0.0, u * 3.5))
    row_y = card.y + p
    send = Rect(card.right - p - send_w, row_y, send_w, ctl_h)
    gear = Rect(send.x - gap - ctl_h, row_y, ctl_h, ctl_h)
    attach = Rect(card.x + p, row_y, ctl_h, ctl_h)
    field_rect = Rect(card.x + p, row_y + ctl_h + sep, card.w - 2 * p, text_h)

    wanted = list(pill_widths)
    while True:
        px = attach.right + gap
        rects: list[Rect | None] = []
        fits = True
        for w in wanted:
            if w is None:
                rects.append(None)
                continue
            rect = Rect(round(px), row_y, round(w), ctl_h)
            if rect.right > gear.x - gap:
                fits = False
            rects.append(rect)
            px = rect.right + gap
        if fits:
            break
        drop = next((i for i in (2, 1, 0) if wanted[i] is not None), None)
        if drop is None:
            rects = [None, None, None]
            break
        wanted[drop] = None

    layout = BarLayout(
        card=card, field=field_rect, send=send, attach=attach, gear=gear, model_pill=rects[0], reasoning_pill=rects[1], image_pill=rects[2], lines=lines
    )
    if tab_widths:
        layout.tab_bar, layout.tabs = _tabs(card, m, tab_widths, active_tab)
    if not section:
        return layout
    panel = Rect(field_rect.x, field_rect.top + sep, field_rect.w, action_h + view_h + header_h)
    layout.log_panel = panel
    layout.log_view = Rect(panel.x, panel.y + action_h, panel.w, view_h)
    if header_h:
        layout.log_header = Rect(panel.x, layout.log_view.top, panel.w, header_h)
    if approval:
        btn_w = round(u * 3.2)
        layout.deny = Rect(panel.right - btn_w, panel.y + 2, btn_w, action_h - 4)
        layout.approve = Rect(layout.deny.x - round(m.pad * 0.5) - btn_w, panel.y + 2, btn_w, action_h - 4)
    return layout
