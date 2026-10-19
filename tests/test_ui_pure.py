from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from blender_ai.agent.messages import LogEntry, LogKind, Status, Usage
from blender_ai.imaging.png import downscale, encode_png, png_size
from blender_ai.ui import dropdown, layout, transcript
from blender_ai.ui.textedit import Span, TextBuffer, caret_line, wrap


def test_editing_and_words() -> None:
    buf = TextBuffer()
    buf.insert("make a red  cube")
    buf.backspace(word=True)
    assert buf.text == "make a red  "
    buf.left(word=True)
    assert buf.caret == len("make a ")
    buf.insert("big ")
    assert buf.text == "make a big red  "
    buf.home()
    buf.delete(word=True)
    assert buf.text == " a big red  "
    buf.insert("\tx\r\ny")
    assert buf.text.startswith("    x\ny")


def test_history_and_lines() -> None:
    buf = TextBuffer()
    for text in ("first", "second"):
        buf.insert(text)
        assert buf.take() == text
    buf.insert("draft")
    buf.up()
    assert buf.text == "second"
    buf.up()
    assert buf.text == "first"
    buf.down()
    buf.down()
    assert buf.text == "draft"
    buf.clear()
    buf.insert("line one\nline two")
    buf.up()
    assert buf.caret == len("line one")
    buf.down()
    assert buf.caret == len(buf.text)


def test_wrap_and_caret_mapping() -> None:
    measure = len
    text = "abc def ghi\njk"
    spans = wrap(text, 6, measure)
    assert [text[s.start : s.end] for s in spans] == ["abc ", "def ", "ghi", "jk"]
    assert caret_line(spans, 4) == (1, 0)
    assert caret_line(spans, 11) == (2, 3)
    assert caret_line(spans, 12) == (3, 0)
    long_word = wrap("abcdefghij", 4, measure)
    assert [s.end - s.start for s in long_word] == [4, 4, 2]
    assert wrap("", 10, measure) == [Span(0, 0)]


TABS = [("MODEL", 110.0, 30.0), ("SCENE", 90.0, 30.0), ("ANIMATE", 120.0, 30.0), ("MATERIAL", 110.0, 30.0)]


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
@pytest.mark.parametrize(("w", "h"), [(800, 600), (1920, 1080), (520, 500)])
def test_bar_centered_and_inside(scale: float, w: int, h: int) -> None:
    m = layout.Metrics.from_scale(scale)
    bar = layout.compute(w, h, m, left_inset=40, right_inset=200, log_height=60, log_header=True, tab_widths=TABS, active_tab="MODEL")
    if bar is None:
        assert w - 240 - 2 * m.unit < 14 * m.unit or h < 8 * m.unit
        return
    visible_center = 40 + (w - 240) / 2
    card = bar.card
    assert abs(card.cx - visible_center) <= 1
    assert card.x >= 40 and card.right <= w - 200
    for rect in (bar.model_pill, bar.reasoning_pill, bar.image_pill, bar.field, bar.attach, bar.gear, bar.send):
        if rect is not None:
            assert card.contains(rect.x, rect.y) and card.contains(rect.right, rect.top)
    assert bar.gear.right <= bar.send.x and bar.send.right <= bar.field.right
    assert bar.send.h == bar.gear.h and bar.send.y == bar.gear.y, "Submit sits in the control row, as tall as the gear"
    assert bar.field.y >= bar.send.top, "the text field spans the card above the controls"
    pills = [r for r in (bar.model_pill, bar.reasoning_pill, bar.image_pill) if r is not None]
    for left, right in zip([bar.attach, *pills], [*pills, bar.gear], strict=True):
        assert left.right <= right.x, "pills don't overlap"
    assert bar.tab_bar is not None and bar.tab_bar.y >= card.top and bar.tab_bar.top <= h
    assert abs(bar.tab_bar.cx - card.cx) <= 1 and bar.tab_bar.w <= card.w
    if bar.log_panel is not None:
        assert card.contains(bar.log_panel.x, bar.log_panel.y) and card.contains(bar.log_panel.right, bar.log_panel.top), "the transcript is inside the bar"
        assert bar.log_panel.y >= bar.field.top, "the transcript sits above the text field"
        assert bar.log_view is not None and bar.log_header is not None and bar.log_view.top <= bar.log_header.y


def test_narrow_bar_drops_pills_image_first_and_tab_labels() -> None:
    m = layout.Metrics.from_scale(1.0)
    wide = layout.compute(1400, 800, m, pill_widths=(140, 130, 120), tab_widths=TABS, active_tab="SCENE")
    narrow = layout.compute(470, 800, m, pill_widths=(140, 130, 120), tab_widths=TABS, active_tab="SCENE")
    assert wide and wide.model_pill and wide.reasoning_pill and wide.image_pill
    assert all(t.label for t in wide.tabs)
    assert narrow and narrow.image_pill is None and narrow.model_pill is not None
    assert [t.label for t in narrow.tabs] == [False, True, False, False], "only the active tab keeps its label"


def test_layout_shrinks_log_and_hides_when_tiny() -> None:
    m = layout.Metrics.from_scale(1.0)
    assert layout.compute(150, 400, m) is None
    short = layout.compute(900, 260, m, log_height=400, log_header=True, tab_widths=TABS, active_tab="MODEL")
    assert short is not None and short.log_view is not None and short.log_view.h < 400 and short.tab_bar.top <= 260
    tall = layout.compute(900, 900, m, log_height=200, log_header=True, approval=True)
    assert tall is not None and tall.log_view.h == 200 and tall.approve is not None
    assert tall.log_view.y >= tall.approve.top >= tall.approve.y >= tall.field.top, "Run / Deny sit between the transcript and the text field"
    empty = layout.compute(900, 900, m)
    assert empty.log_panel is None and tall.card.y == empty.card.y and tall.card.h > empty.card.h, "the bar grows upwards with the transcript"


def test_hit_testing() -> None:
    m = layout.Metrics.from_scale(1.0)
    bar = layout.compute(1000, 800, m, pill_widths=(90, 110, 90), log_height=80, log_header=True, tab_widths=TABS, active_tab="MODEL")
    assert bar is not None
    assert bar.hit(bar.send.cx, bar.send.cy) == "send"
    assert bar.hit(bar.model_pill.cx, bar.model_pill.cy) == "model"
    assert bar.hit(bar.reasoning_pill.cx, bar.reasoning_pill.cy) == "reasoning"
    assert bar.hit(bar.image_pill.cx, bar.image_pill.cy) == "image"
    assert bar.hit(bar.attach.cx, bar.attach.cy) == "attach"
    assert bar.hit(bar.field.cx, bar.field.cy) == "field"
    tab = bar.tabs[2]
    assert bar.hit(tab.rect.cx, tab.rect.cy) == "tab:ANIMATE"
    assert bar.hit(5, 5) is None
    view = bar.log_view
    bar.log_rows = [(layout.Rect(view.x, view.y, view.w, 20), 42)]
    assert bar.hit(view.cx, view.y + 10) == "log:42"
    bar.log_latest = layout.Rect(view.right - 60, view.y, 60, 20)
    assert bar.hit(view.right - 30, view.y + 10) == "log_latest"
    assert bar.hit(view.cx, view.top - 5) == "log"


def test_text_area_grows_upwards() -> None:
    m = layout.Metrics.from_scale(1.0)
    one = layout.compute(1000, 800, m, text_lines=1)
    two = layout.compute(1000, 800, m, text_lines=2)
    four = layout.compute(1000, 800, m, text_lines=4)
    many = layout.compute(1000, 800, m, text_lines=40)
    assert one and two and four and many
    assert one.card.h == two.card.h, "the text area always shows at least two lines"
    assert four.card.h > one.card.h and four.card.y == one.card.y
    assert many.lines == layout.MAX_TEXT_LINES


def _decode(data: bytes) -> np.ndarray:
    w, h = png_size(data)
    pos, raw = 8, b""
    channels = {0: 1, 2: 3, 4: 2, 6: 4}
    color_type = data[25]
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        if tag == b"IDAT":
            raw += data[pos + 8 : pos + 8 + length]
        pos += 12 + length
    c = channels[color_type]
    rows = np.frombuffer(zlib.decompress(raw), dtype=np.uint8).reshape(h, w * c + 1)
    assert set(rows[:, 0]) == {2}
    return np.cumsum(rows[:, 1:].astype(np.uint16), axis=0).astype(np.uint8).reshape(h, w, c)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_png_round_trip(channels: int) -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(17, 23, channels), dtype=np.uint8)
    data = encode_png(img if channels > 1 else img[:, :, 0])
    assert png_size(data) == (23, 17)
    assert np.array_equal(_decode(data), img)


def test_downscale() -> None:
    img = np.ones((400, 300, 3), dtype=np.float32)
    small = downscale(img, 128)
    assert max(small.shape[:2]) <= 128 and np.allclose(small, 1.0)


def _models_dropdown(count: int = 30) -> dropdown.Dropdown:
    items = [dropdown.Item(f"vendor/model-{n}", f"Model {n}", "free" if n % 3 == 0 else "", n == 12) for n in range(count)]
    items.append(dropdown.Item("__loading__", "Loading models…", enabled=False))
    return dropdown.Dropdown("model", items, searchable=True)


def test_dropdown_opens_above_its_pill_and_scrolls_to_the_checked_item() -> None:
    m = layout.Metrics.from_scale(1.0)
    dd = _models_dropdown()
    dd.start_at_checked(dropdown.MAX_ROWS)
    anchor = layout.Rect(300, 40, 120, 25)
    lay = dropdown.compute(dd, anchor, 1400, 800, m, 340)
    assert lay.panel.y > anchor.top and lay.panel.w == 340 and lay.search is not None
    assert lay.row_count == dropdown.MAX_ROWS and lay.more_below
    assert dd.hover == 12 and any(index == 12 for _, index in lay.rows), "the current model is in view"
    rect, index = lay.rows[0]
    assert lay.row_at(rect.cx, rect.cy) == index and lay.contains(rect.cx, rect.cy)
    low = dropdown.compute(dd, anchor, 1400, 260, m, 340)
    assert low.row_count < dropdown.MAX_ROWS and low.panel.top <= 260


def test_dropdown_filtering_keys_and_disabled_rows() -> None:
    dd = _models_dropdown()
    dd.type("model-2")
    assert [i.id for i in dd.visible()][:3] == ["vendor/model-2", "vendor/model-20", "vendor/model-21"]
    assert dd.hover == 0 and dd.chosen().id == "vendor/model-2"
    dd.move(-1, 5)
    assert dd.chosen().id == dd.visible()[-1].id, "up from the first row wraps to the last enabled row"
    dd.backspace()
    dd.backspace()
    assert dd.query == "model" and len(dd.visible()) == 31
    dd.type(" free")
    assert all(i.hint == "free" for i in dd.visible())
    loading = dropdown.Dropdown("model", [dropdown.Item("__loading__", "Loading models…", enabled=False)])
    loading.start_at_checked(5)
    assert loading.chosen() is None


def test_dropdown_scroll_is_clamped() -> None:
    dd = _models_dropdown(12)
    dd.scroll_by(50, 5)
    assert dd.scroll == len(dd.visible()) - 5
    dd.scroll_by(-50, 5)
    assert dd.scroll == 0


def entry(i: int, kind: LogKind, text: str, **kw: object) -> LogEntry:
    return LogEntry(i, kind, text, **kw)


def test_transcript_reads_like_a_chat() -> None:
    log = [
        entry(1, LogKind.USER, "make a red cube and a ball"),
        entry(2, LogKind.REASONING, "Reasoning", detail="plan the cube", started=10.0, updated=14.2),
        entry(3, LogKind.ASSISTANT, "I'll add both now."),
        entry(4, LogKind.TOOL, "create_primitive", detail="kind=cube  name=Crate", result="Created Crate", tool="create_primitive"),
        entry(5, LogKind.TOOL, "run_python", detail="code=...", result="NameError: x", status=Status.ERROR, tool="run_python"),
        entry(6, LogKind.TOOL, "frame_view", status=Status.RUNNING, tool="frame_view"),
        entry(7, LogKind.ASSISTANT, "Done: a red cube named Crate."),
    ]
    lines = transcript.build(log, len, 60, 4)
    text = [(line.marker, line.text, line.suffix) for line in lines if line.height >= 1]
    assert text[0] == ("user", "make a red cube and a ball", "")
    assert lines[0].band and lines[0].tone == "user"
    assert ("thought", "Thought for 4s", "") in text
    assert ("dot", "I'll add both now.", "") in text
    assert ("dot", "create_primitive", "kind=cube  name=Crate") in text
    assert ("tree", "Created Crate", "") in text
    assert any(line.marker == "tree" and line.tone == "error" and line.text == "NameError: x" for line in lines)
    assert ("spin", "frame_view", "") in text, "a running tool spins and has no result line yet"
    assert sum(1 for line in lines if line.height < 1) == len(log) - 1, "a small gap between blocks"


def test_transcript_wraps_and_expands() -> None:
    long = "word " * 30
    lines = transcript.build([entry(1, LogKind.ASSISTANT, long)], len, 40, 4)
    assert len(lines) > 3 and lines[0].marker == "dot" and all(line.marker == "" for line in lines[1:])
    assert all(len(line.text) <= 40 for line in lines)
    tool = entry(2, LogKind.TOOL, "run_python", detail="code=line1\nline2", result="ok", expanded=True)
    expanded = [line.text for line in transcript.build([tool], len, 40, 4)]
    assert "line2" in expanded and "ok" in expanded
    thinking = entry(3, LogKind.REASONING, "Reasoning", detail="first idea", status=Status.RUNNING)
    assert [line.text for line in transcript.build([thinking], len, 40, 4)] == ["Thinking…"]


def test_transcript_folds_sub_agents() -> None:
    log = [
        entry(1, LogKind.USER, "build a village"),
        entry(2, LogKind.AGENT, "Model Builder · House", detail="Build a house", status=Status.RUNNING),
        entry(3, LogKind.TOOL, "create_primitive", detail="kind=cube", result="ok", agent="Model Builder · House"),
        entry(4, LogKind.TOOL, "add_modifier", detail="type=BEVEL", status=Status.RUNNING, agent="Model Builder · House"),
    ]
    lines = transcript.build(log, len, 80, 4)
    assert [line.text for line in lines if line.height >= 1] == ["build a village", "Model Builder · House", "add_modifier"]
    assert lines[-1].suffix == "2 steps" and lines[-1].marker == "tree"
    log[1].status, log[1].result, log[1].expanded = Status.OK, "Built House in collection House", True
    lines = transcript.build(log, len, 80, 4)
    assert [line.text for line in lines if line.height >= 1][2] == "Built House in collection House"
    assert any(line.text == "create_primitive" and line.indent == 1 for line in lines), "expanding shows the steps"


def test_scroll_follows_until_the_user_scrolls_up() -> None:
    s = transcript.Scroll()
    s.sync(100, 200)
    assert s.offset == 0 and s.following
    s.sync(500, 200)
    assert s.offset == 0, "new lines keep the view on the latest"
    s.scroll(120, 200)
    assert s.offset == 120 and not s.following
    s.sync(560, 200)
    assert s.offset == 180, "the same lines stay in view while new ones arrive"
    s.scroll(10_000, 200)
    assert s.offset == 360
    s.sync(250, 200)
    assert s.offset == 50, "clamped when the content shrinks"
    s.to_latest()
    assert s.following


def test_token_labels() -> None:
    assert transcript.fmt_tokens(950) == "950" and transcript.fmt_tokens(12_400) == "12.4k" and transcript.fmt_tokens(2_500_000) == "2.5M"
    usage = Usage()
    assert transcript.usage_parts(usage) == []
    usage.add({"input_tokens": 250_400, "output_tokens": 17_500, "cached_tokens": 177_800})
    assert transcript.usage_parts(usage) == ["250.4k in", "17.5k out", "177.8k cached"]
