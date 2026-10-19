from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


def _is_space(ch: str) -> bool:
    return ch.isspace()


class TextBuffer:
    def __init__(self, text: str = "", history_limit: int = 50) -> None:
        self.text = text
        self.caret = len(text)
        self.history: list[str] = []
        self.history_limit = history_limit
        self._hist_index: int | None = None
        self._draft = ""

    def insert(self, s: str) -> None:
        s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        s = "".join(ch for ch in s if ch == "\n" or ch.isprintable())
        if not s:
            return
        self.text = self.text[: self.caret] + s + self.text[self.caret :]
        self.caret += len(s)
        self._hist_index = None

    def backspace(self, word: bool = False) -> None:
        if self.caret == 0:
            return
        start = self._word_left(self.caret) if word else self.caret - 1
        self.text = self.text[:start] + self.text[self.caret :]
        self.caret = start
        self._hist_index = None

    def delete(self, word: bool = False) -> None:
        if self.caret >= len(self.text):
            return
        end = self._word_right(self.caret) if word else self.caret + 1
        self.text = self.text[: self.caret] + self.text[end:]
        self._hist_index = None

    def clear(self) -> None:
        self.text = ""
        self.caret = 0
        self._hist_index = None

    def take(self) -> str:
        text = self.text
        if text.strip() and (not self.history or self.history[-1] != text):
            self.history.append(text)
            del self.history[: -self.history_limit]
        self.clear()
        return text

    def left(self, word: bool = False) -> None:
        self.caret = self._word_left(self.caret) if word else max(0, self.caret - 1)

    def right(self, word: bool = False) -> None:
        self.caret = self._word_right(self.caret) if word else min(len(self.text), self.caret + 1)

    def home(self) -> None:
        self.caret = self.text.rfind("\n", 0, self.caret) + 1

    def end(self) -> None:
        nl = self.text.find("\n", self.caret)
        self.caret = len(self.text) if nl < 0 else nl

    def up(self) -> None:
        line_start = self.text.rfind("\n", 0, self.caret) + 1
        if line_start == 0:
            self.history_prev()
            return
        column = self.caret - line_start
        prev_start = self.text.rfind("\n", 0, line_start - 1) + 1
        self.caret = min(prev_start + column, line_start - 1)

    def down(self) -> None:
        nl = self.text.find("\n", self.caret)
        if nl < 0:
            self.history_next()
            return
        line_start = self.text.rfind("\n", 0, self.caret) + 1
        column = self.caret - line_start
        next_end = self.text.find("\n", nl + 1)
        next_end = len(self.text) if next_end < 0 else next_end
        self.caret = min(nl + 1 + column, next_end)

    def history_prev(self) -> None:
        if not self.history:
            return
        if self._hist_index is None:
            self._draft = self.text
            self._hist_index = len(self.history) - 1
        elif self._hist_index > 0:
            self._hist_index -= 1
        self._set_from_history()

    def history_next(self) -> None:
        if self._hist_index is None:
            return
        if self._hist_index < len(self.history) - 1:
            self._hist_index += 1
            self._set_from_history()
        else:
            self._hist_index = None
            self.text = self._draft
            self.caret = len(self.text)

    def _set_from_history(self) -> None:
        assert self._hist_index is not None
        self.text = self.history[self._hist_index]
        self.caret = len(self.text)

    def _word_left(self, i: int) -> int:
        while i > 0 and _is_space(self.text[i - 1]):
            i -= 1
        while i > 0 and not _is_space(self.text[i - 1]):
            i -= 1
        return i

    def _word_right(self, i: int) -> int:
        n = len(self.text)
        while i < n and _is_space(self.text[i]):
            i += 1
        while i < n and not _is_space(self.text[i]):
            i += 1
        return i


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def wrap(text: str, width: float, measure: Callable[[str], float]) -> list[Span]:
    spans: list[Span] = []
    para_start = 0
    for paragraph in text.split("\n"):
        spans.extend(_wrap_paragraph(paragraph, para_start, width, measure))
        para_start += len(paragraph) + 1
    return spans


def _wrap_paragraph(p: str, offset: int, width: float, measure: Callable[[str], float]) -> list[Span]:
    if not p:
        return [Span(offset, offset)]
    spans: list[Span] = []
    start = 0
    n = len(p)
    while start < n:
        if measure(p[start:]) <= width:
            spans.append(Span(offset + start, offset + n))
            break
        end = start + 1
        last_break = -1
        while end <= n and measure(p[start:end]) <= width:
            if p[end - 1] == " ":
                last_break = end
            end += 1
        cut = last_break if last_break > start else max(start + 1, end - 1)
        spans.append(Span(offset + start, offset + cut))
        start = cut
    return spans


def caret_line(spans: list[Span], caret: int) -> tuple[int, int]:
    for i, span in enumerate(spans):
        nxt = spans[i + 1].start if i + 1 < len(spans) else None
        if span.start <= caret <= span.end and (nxt is None or caret < nxt or nxt > span.end):
            return i, caret - span.start
    last = spans[-1] if spans else Span(0, 0)
    return len(spans) - 1 if spans else 0, max(0, caret - last.start)
