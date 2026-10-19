from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..agent.messages import LogEntry, LogKind, Status, Usage
from .textedit import wrap

GAP = 0.45
EXPANDED_LINES = 24
TONES = {Status.RUNNING: "running", Status.OK: "ok", Status.ERROR: "error", Status.CANCELLED: "cancelled", Status.WAITING: "warning"}

Measure = Callable[[str], float]


@dataclass(frozen=True)
class Line:
    entry_id: int = -1
    text: str = ""
    tone: str = "text"
    marker: str = ""
    marker_tone: str = ""
    suffix: str = ""
    indent: int = 0
    band: bool = False
    height: float = 1.0


@dataclass
class Scroll:
    offset: float = 0.0
    content: float = 0.0

    @property
    def following(self) -> bool:
        return self.offset < 1.0

    def sync(self, content: float, view: float) -> None:
        if not self.following and content > self.content:
            self.offset += content - self.content
        self.content = content
        self.offset = max(0.0, min(self.offset, content - view))

    def scroll(self, delta: float, view: float) -> None:
        self.offset = max(0.0, min(self.offset + delta, self.content - view))

    def to_latest(self) -> None:
        self.offset = 0.0


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def usage_parts(usage: Usage) -> list[str]:
    if not (usage.input_tokens or usage.output_tokens):
        return []
    parts = [f"{fmt_tokens(usage.input_tokens)} in", f"{fmt_tokens(usage.output_tokens)} out"]
    if usage.cache_read_tokens:
        parts.append(f"{fmt_tokens(usage.cache_read_tokens)} cached")
    return parts


def _wrapped(text: str, measure: Measure, width: float, limit: int = 0) -> list[str]:
    lines = [text[s.start : s.end].rstrip() for s in wrap(text.strip(), max(width, 1.0), measure)]
    while lines and not lines[-1]:
        lines.pop()
    return [*lines[: limit - 1], "…"] if limit and len(lines) > limit else lines


def _first(text: str) -> str:
    return next((line.strip() for line in text.strip().splitlines() if line.strip()), "")


def _block(entry: LogEntry, text: str, measure: Measure, width: float, indent: int, **first: str | bool) -> list[Line]:
    lines = _wrapped(text, measure, width) or [""]
    tone = str(first.pop("tone", "text"))
    band = bool(first.pop("band", False))
    out = [Line(entry.id, lines[0], tone, indent=indent, band=band, **first)]
    return out + [Line(entry.id, line, tone, indent=indent, band=band) for line in lines[1:]]


def _detail(entry: LogEntry, text: str, measure: Measure, width: float, indent: int, tone: str = "dim") -> list[Line]:
    return [Line(entry.id, line, tone, indent=indent) for line in _wrapped(text, measure, width, EXPANDED_LINES)]


def _tree(entry: LogEntry, text: str, indent: int, tone: str = "dim", suffix: str = "") -> Line:
    return Line(entry.id, text, tone, marker="tree", indent=indent, suffix=suffix)


def _entry(entry: LogEntry, children: dict[str, list[LogEntry]], measure: Measure, width: float, indent_w: float, indent: int) -> list[Line]:
    w = width - indent * indent_w
    tone = TONES.get(entry.status, "ok")
    mark = "spin" if entry.status is Status.RUNNING else "dot"
    kind = entry.kind
    if kind is LogKind.USER:
        return _block(entry, entry.text, measure, w, indent, tone="user", marker="user", band=True)
    if kind is LogKind.ASSISTANT:
        return _block(entry, entry.text, measure, w, indent, marker="dot", marker_tone="text") if entry.text.strip() else []
    if kind is LogKind.REASONING:
        if entry.status is Status.RUNNING:
            return [Line(entry.id, "Thinking…", "dim", marker="spin", marker_tone="running", indent=indent)]
        seconds = round(entry.updated - entry.started)
        out = [Line(entry.id, f"Thought for {seconds}s" if seconds >= 1 else "Thought", "dim", marker="thought", indent=indent)]
        return out + (_detail(entry, entry.detail, measure, w, indent) if entry.expanded and entry.detail.strip() else [])
    if kind is LogKind.TOOL:
        name = ("! " if entry.destructive else "") + entry.text
        out = [Line(entry.id, name, "warning" if entry.destructive else "text", marker=mark, marker_tone=tone, indent=indent, suffix=_first(entry.detail))]
        if entry.expanded:
            out += _detail(entry, entry.detail, measure, w, indent) + _detail(entry, entry.result, measure, w, indent, "error" if tone == "error" else "dim")
        elif entry.result:
            out.append(_tree(entry, _first(entry.result), indent, "error" if tone == "error" else "dim"))
        elif entry.status is Status.CANCELLED:
            out.append(_tree(entry, "Cancelled", indent))
        return out
    if kind is LogKind.AGENT:
        steps = [e for e in children.get(entry.text, []) if e.kind is LogKind.TOOL]
        count = f"{len(steps)} step{'s' * (len(steps) != 1)}"
        out = [Line(entry.id, entry.text, "text", marker=mark, marker_tone=tone, indent=indent, suffix="sub-agent")]
        if entry.status is Status.RUNNING:
            out.append(_tree(entry, steps[-1].text if steps else _first(entry.detail) or "Starting…", indent, suffix=count))
        else:
            out.append(_tree(entry, _first(entry.result) or "Done", indent, "error" if tone == "error" else "dim", count))
        if entry.expanded:
            for child in children.get(entry.text, []):
                out += _entry(child, children, measure, width, indent_w, indent + 1)
        return out
    if kind is LogKind.ERROR:
        return _block(entry, entry.text, measure, w, indent, tone="error", marker="dot", marker_tone="error")
    if kind is LogKind.APPROVAL:
        out = [Line(entry.id, "Run this Python?", "warning", marker="dot", marker_tone="warning", indent=indent)]
        code = [Line(entry.id, line, "dim", indent=indent) for line in entry.detail.splitlines()[:EXPANDED_LINES]]
        return out + (code if entry.expanded else [_tree(entry, "Click to read the code", indent)])
    if kind is LogKind.QUESTION:
        out = _block(entry, entry.text, measure, w, indent, marker="dot", marker_tone="accent")
        return out + ([_tree(entry, _first(entry.detail), indent)] if entry.detail.strip() else [])
    return _block(entry, entry.text, measure, w, indent, tone="dim")


def build(entries: Sequence[LogEntry], measure: Measure, width: float, indent_w: float) -> list[Line]:
    children: dict[str, list[LogEntry]] = {}
    for entry in entries:
        if entry.agent:
            children.setdefault(entry.agent, []).append(entry)
    lines: list[Line] = []
    for entry in entries:
        if entry.agent:
            continue
        block = _entry(entry, children, measure, width, indent_w, 0)
        if block:
            if lines:
                lines.append(Line(height=GAP))
            lines += block
    return lines


def content_height(lines: Sequence[Line], row_h: float) -> float:
    return sum(line.height for line in lines) * row_h
