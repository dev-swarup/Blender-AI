from __future__ import annotations

from dataclasses import dataclass, field

from .layout import Metrics, Rect

MAX_ROWS = 10


@dataclass(frozen=True)
class Item:
    id: str
    label: str
    hint: str = ""
    checked: bool = False
    enabled: bool = True


@dataclass
class Dropdown:
    target: str
    items: list[Item] = field(default_factory=list)
    searchable: bool = False
    query: str = ""
    hover: int = -1
    scroll: int = 0
    pressed: int | None = None

    def visible(self) -> list[Item]:
        words = self.query.lower().split()
        if not words:
            return list(self.items)
        return [i for i in self.items if all(w in f"{i.label} {i.id} {i.hint}".lower() for w in words)]

    def start_at_checked(self, rows: int) -> None:
        items = self.visible()
        self.hover = next((n for n, i in enumerate(items) if i.checked), 0 if items else -1)
        self.reveal(rows)

    def reveal(self, rows: int) -> None:
        if self.hover < 0:
            return
        if self.hover < self.scroll:
            self.scroll = self.hover
        elif self.hover >= self.scroll + rows:
            self.scroll = self.hover - rows + 1
        self.clamp(rows)

    def clamp(self, rows: int) -> None:
        self.scroll = max(0, min(self.scroll, len(self.visible()) - rows))

    def move(self, delta: int, rows: int) -> None:
        items = self.visible()
        enabled = [n for n, i in enumerate(items) if i.enabled]
        if not enabled:
            return
        if self.hover not in enabled:
            self.hover = enabled[0] if delta > 0 else enabled[-1]
        else:
            pos = (enabled.index(self.hover) + delta) % len(enabled)
            self.hover = enabled[pos]
        self.reveal(rows)

    def scroll_by(self, delta: int, rows: int) -> None:
        self.scroll += delta
        self.clamp(rows)

    def type(self, text: str) -> None:
        self.query += text
        self._after_filter()

    def backspace(self) -> None:
        self.query = self.query[:-1]
        self._after_filter()

    def _after_filter(self) -> None:
        items = self.visible()
        self.scroll = 0
        self.hover = next((n for n, i in enumerate(items) if i.enabled), -1)

    def chosen(self) -> Item | None:
        items = self.visible()
        return items[self.hover] if 0 <= self.hover < len(items) and items[self.hover].enabled else None


@dataclass
class DropdownLayout:
    panel: Rect
    search: Rect | None
    rows: list[tuple[Rect, int]]
    row_count: int
    more_above: bool = False
    more_below: bool = False

    def row_at(self, px: float, py: float) -> int | None:
        for rect, index in self.rows:
            if rect.contains(px, py):
                return index
        return None

    def contains(self, px: float, py: float) -> bool:
        return self.panel.contains(px, py)


def compute(dd: Dropdown, anchor: Rect, region_w: float, region_h: float, m: Metrics, width: float) -> DropdownLayout:
    u = m.unit
    pad = round(u * 0.25)
    row_h = round(u * 1.1)
    gap = round(u * 0.25)
    search_h = round(u * 1.3) if dd.searchable else 0
    items = dd.visible()
    room = region_h - (anchor.top + gap) - u * 0.5 - 2 * pad - search_h
    fit = max(1, int(room // row_h))
    rows = max(1, min(MAX_ROWS, fit, len(items) or 1))
    dd.clamp(rows)
    w = round(max(anchor.w, width))
    x = round(min(max(anchor.x, u * 0.5), region_w - u * 0.5 - w))
    h = 2 * pad + search_h + rows * row_h
    panel = Rect(x, anchor.top + gap, w, h)
    search = Rect(x + pad, panel.top - pad - search_h, w - 2 * pad, search_h - round(u * 0.15)) if search_h else None
    out: list[tuple[Rect, int]] = []
    y = panel.top - pad - search_h - row_h
    for index in range(dd.scroll, min(len(items), dd.scroll + rows)):
        out.append((Rect(x + pad, y, w - 2 * pad, row_h), index))
        y -= row_h
    return DropdownLayout(panel, search, out, rows, more_above=dd.scroll > 0, more_below=dd.scroll + rows < len(items))
