from __future__ import annotations

import contextlib
import time
from typing import Any

import bpy
from bpy.props import BoolProperty

from .. import models, state
from ..agent import roles
from ..agent.messages import Phase
from . import overlay
from .dropdown import Dropdown, Item

PASS = {"PASS_THROUGH"}
EAT = {"RUNNING_MODAL"}
NAVIGATION = {
    "MIDDLEMOUSE",
    "RIGHTMOUSE",
    "WHEELUPMOUSE",
    "WHEELDOWNMOUSE",
    "WHEELINMOUSE",
    "WHEELOUTMOUSE",
    "TRACKPADPAN",
    "TRACKPADZOOM",
    "MOUSEROTATE",
    "MOUSESMARTZOOM",
    "NDOF_MOTION",
    "MOUSEMOVE",
    "INBETWEEN_MOUSEMOVE",
    "WINDOW_DEACTIVATE",
    "TIMER",
    "TIMER_REPORT",
    "TIMERREGION",
}


def _prefs() -> Any:
    from ..preferences import get_prefs

    return get_prefs()


def _redraw(context: Any) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def submit(context: Any) -> None:
    st = state.get()
    session = st.session
    if session is None:
        return
    text = st.ui.buffer.text
    attachment = None
    if st.ui.attachment_path and session.phase is Phase.IDLE:
        from ..imaging.nodes import read_image_file
        from ..imaging.png import downscale, encode_png
        from ..tools.registry import ImagePart

        try:
            pixels = read_image_file(st.ui.attachment_path)
            attachment = ImagePart(encode_png(downscale(pixels[:, :, :3], 1568)))
        except (RuntimeError, ValueError) as exc:
            session.notify(f"Couldn't read the attached image: {exc}", error=True)
    if not text.strip() and attachment is None:
        return
    st.ui.buffer.take()
    session.submit(text, attachment)
    _redraw(context)


REFRESH = "__refresh__"
ON_RELEASE = ("gear", "attach")
REASONING_HINTS = {"auto": "model default", "low": "fastest", "medium": "balanced", "high": "most careful"}


def _list_items(listed: Any, chosen: list[Any], current: str, empty: str) -> list[Item]:
    ids = {m.id for m in chosen}
    items: list[Item] = []
    if current and current not in ids:
        items.append(Item(current, current, "current", True))
    items += [Item(m.id, m.label or m.id, "free" if m.free else "", m.id == current) for m in chosen]
    if listed.loading:
        items.append(Item("__loading__", "Loading models…", enabled=False))
    elif listed.error:
        items.append(Item("__error__", listed.error, enabled=False))
    elif not chosen:
        items.append(Item("__empty__", empty, enabled=False))
    items.append(Item(REFRESH, "↻ Reload the list from the API"))
    return items


def dropdown_items(target: str) -> list[Item]:
    st = state.get()
    prefs = _prefs()
    if target == "model":
        return _list_items(st.chat_models, models.chat_models(st.chat_models.models), prefs.model(), "No models yet. Add your API key in settings (gear).")
    if target == "image":
        return _list_items(st.image_models, models.image_models(st.image_models.models), prefs.image_model, "This API lists no image models.")
    return [Item(level, models.LEVEL_LABELS[level], REASONING_HINTS.get(level, ""), level == prefs.reasoning) for level in models.REASONING_LEVELS]


def open_dropdown(context: Any, target: str, region: Any) -> None:
    st = state.get()
    ui = st.ui
    if ui.dropdown is not None and ui.dropdown.target == target:
        close_dropdown(context)
        return
    if target == "image" and _prefs().image_endpoint() is None:
        if st.session is not None:
            st.session.notify("Image models need an image API: add one under Separate image API in settings.")
        return
    if target in ("model", "image"):
        from .operators import refresh_model_lists

        listed = st.chat_models if target == "model" else st.image_models
        if not listed.models and not listed.loading:
            refresh_model_lists(force=True)
    items = dropdown_items(target)
    dd = Dropdown(target, items, searchable=target in ("model", "image") and len(items) > 8)
    dd.start_at_checked(8)
    ui.dropdown = dd
    ui.dropdown_region = region.as_pointer() if region is not None else 0
    ui.dropdown_layout = None
    if not st.modal_running:
        bpy.ops.blender_ai.prompt_input("INVOKE_DEFAULT", focus=False)
    _redraw(context)


def close_dropdown(context: Any) -> None:
    ui = state.get().ui
    ui.dropdown = None
    ui.dropdown_layout = None
    _redraw(context)


def choose(context: Any, item: Item | None) -> None:
    ui = state.get().ui
    dd = ui.dropdown
    close_dropdown(context)
    if dd is None or item is None or not item.enabled:
        return
    prefs = _prefs()
    if item.id == REFRESH:
        from .operators import refresh_model_lists

        refresh_model_lists(force=True)
    elif dd.target == "model":
        prefs.set_model(item.id)
    elif dd.target == "image":
        prefs.image_model = item.id
    elif dd.target == "reasoning":
        prefs.reasoning = item.id
    _redraw(context)


def release_click(context: Any, hit: str | None) -> bool:
    st = state.get()
    pressed, st.ui.pressed = st.ui.pressed, None
    if pressed is None or hit != pressed:
        return False
    if pressed == "gear":
        st.ui.focused = False
        bpy.ops.wm.call_panel(name="BLENDER_AI_PT_quick_settings", keep_open=True)
    elif pressed == "attach":
        st.ui.focused = False
        bpy.ops.blender_ai.attach_image("INVOKE_DEFAULT")
    _redraw(context)
    return True


def handle_click(context: Any, hit: str, region: Any = None) -> bool:
    st = state.get()
    session = st.session
    prefs = _prefs()
    if hit in ("field", "tabs"):
        return hit == "field"
    if hit.startswith("tab:"):
        prefs.mode = hit.split(":", 1)[1]
    elif hit in ("model", "image", "reasoning"):
        open_dropdown(context, hit, region)
        return False
    elif hit in ON_RELEASE:
        if hit == "attach" and st.ui.attachment_path:
            st.ui.attachment_path = None
        else:
            st.ui.pressed = hit
    elif hit == "send":
        if session is not None and session.busy and session.phase is not Phase.QUESTION:
            session.stop()
        else:
            submit(context)
    elif hit == "approve" and session is not None:
        session.approve(True)
    elif hit == "deny" and session is not None:
        session.approve(False)
    elif hit.startswith("log:") and session is not None:
        entry_id = int(hit.split(":", 1)[1])
        for entry in session.log:
            if entry.id == entry_id:
                entry.expanded = not entry.expanded
                break
    elif hit == "log":
        st.ui.expanded_log = not st.ui.expanded_log
    _redraw(context)
    return False


class BLENDER_AI_OT_prompt_input(bpy.types.Operator):
    bl_description = "Type into the Blender AI prompt bar"

    bl_idname = "blender_ai.prompt_input"
    bl_label = "Blender AI Prompt"
    bl_options = {"INTERNAL"}

    focus: BoolProperty(default=True, options={"SKIP_SAVE"})

    _timer: Any = None
    _blink = 0.0

    def invoke(self, context: Any, event: Any) -> set[str]:
        st = state.get()
        if self.focus:
            st.ui.focused = True
        if st.modal_running:
            return {"CANCELLED"}
        st.modal_running = True
        st.ui.caret_visible = True
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        _redraw(context)
        return EAT

    def _end(self, context: Any, passthrough: bool = False) -> set[str]:
        st = state.get()
        st.modal_running = False
        st.ui.focused = False
        st.ui.hover = None
        st.ui.dropdown = None
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _redraw(context)
        return {"FINISHED", "PASS_THROUGH"} if passthrough else {"FINISHED"}

    def cancel(self, context: Any) -> None:
        self._end(context)

    def modal(self, context: Any, event: Any) -> set[str]:
        st = state.get()
        if not st.registered or not (st.ui.focused or st.ui.dropdown is not None):
            return self._end(context, passthrough=True)
        if event.type == "TIMER":
            now = time.monotonic()
            if now - self._blink >= 0.45:
                self._blink = now
                st.ui.caret_visible = not st.ui.caret_visible
                _redraw(context)
            return PASS
        if st.ui.dropdown is not None:
            handled = self._dropdown_event(context, event)
            if handled is not None:
                return handled
        if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            hit, _area, _region = overlay.hit_test(context, event)
            hover = hit if hit in ("model", "image", "reasoning", "gear", "attach") else None
            if hover != st.ui.hover:
                st.ui.hover = hover
                _redraw(context)
            return PASS
        if event.type in NAVIGATION:
            return PASS
        if event.type == "LEFTMOUSE":
            hit, _area, region = overlay.hit_test(context, event)
            if event.value == "RELEASE":
                return EAT if release_click(context, hit) else PASS
            if event.value != "PRESS":
                return PASS
            if hit is None:
                return self._end(context, passthrough=True)
            if handle_click(context, hit, region):
                st.ui.focused = True
            return EAT
        if not st.ui.focused:
            return PASS
        if event.value not in {"PRESS", "CLICK"} and event.type != "TEXTINPUT":
            return EAT if event.type in {"RET", "NUMPAD_ENTER"} else PASS
        return self._key(context, event)

    def _dropdown_event(self, context: Any, event: Any) -> set[str] | None:
        st = state.get()
        ui = st.ui
        dd = ui.dropdown
        lay = ui.dropdown_layout
        t = event.type
        found = overlay.region_at(context, event.mouse_x, event.mouse_y)
        inside = None
        if found is not None and lay is not None and found[1].as_pointer() == ui.dropdown_region:
            px, py = event.mouse_x - found[1].x, event.mouse_y - found[1].y
            inside = (px, py) if lay.contains(px, py) else None
        rows = lay.row_count if lay is not None else 8
        if t in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
            if inside is not None:
                index = lay.row_at(*inside)
                if index is not None and index != dd.hover:
                    dd.hover = index
                    _redraw(context)
            return PASS
        if t in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and inside is not None:
            dd.scroll_by(-2 if t == "WHEELUPMOUSE" else 2, rows)
            _redraw(context)
            return EAT
        if t == "LEFTMOUSE":
            if inside is not None:
                index = lay.row_at(*inside)
                if event.value == "PRESS":
                    dd.pressed = index
                elif event.value == "RELEASE":
                    if index is not None and index == dd.pressed:
                        dd.hover = index
                        choose(context, dd.chosen())
                    dd.pressed = None
                return EAT
            if event.value == "PRESS":
                hit, _area, region = overlay.hit_test(context, event)
                target = dd.target
                close_dropdown(context)
                if hit is not None and hit != target and handle_click(context, hit, region):
                    ui.focused = True
                return EAT
            return EAT if event.value == "RELEASE" else PASS
        if event.value not in {"PRESS", "CLICK"} and t != "TEXTINPUT":
            return EAT if t in {"RET", "NUMPAD_ENTER", "ESC"} else None
        if t == "ESC":
            close_dropdown(context)
            return EAT
        if t in {"UP_ARROW", "DOWN_ARROW"}:
            dd.move(-1 if t == "UP_ARROW" else 1, rows)
            _redraw(context)
            return EAT
        if t in {"RET", "NUMPAD_ENTER"}:
            choose(context, dd.chosen())
            return EAT
        if t in NAVIGATION:
            return PASS
        ctrl = event.ctrl or event.oskey
        if dd.searchable and not ctrl and not event.alt:
            if t == "BACK_SPACE":
                dd.backspace()
                _redraw(context)
                return EAT
            if event.unicode and event.unicode.isprintable():
                dd.type(event.unicode)
                _redraw(context)
                return EAT
        return EAT if not ctrl else PASS

    def _key(self, context: Any, event: Any) -> set[str]:
        st = state.get()
        buf = st.ui.buffer
        ctrl = event.ctrl or event.oskey
        t = event.type
        st.ui.caret_visible = True
        if t == "ESC" or (t == "SPACE" and ctrl):
            return self._end(context)
        if t in {"RET", "NUMPAD_ENTER"}:
            if event.shift:
                buf.insert("\n")
            else:
                submit(context)
        elif t == "BACK_SPACE":
            buf.backspace(word=ctrl)
        elif t == "DEL":
            buf.delete(word=ctrl)
        elif t == "LEFT_ARROW":
            buf.left(word=ctrl)
        elif t == "RIGHT_ARROW":
            buf.right(word=ctrl)
        elif t == "HOME":
            buf.home()
        elif t == "END":
            buf.end()
        elif t == "UP_ARROW":
            buf.up()
        elif t == "DOWN_ARROW":
            buf.down()
        elif t == "TAB" and not ctrl:
            prefs = _prefs()
            prefs.mode = roles.toggled(prefs.current_mode()).value
        elif ctrl and t == "V":
            buf.insert(context.window_manager.clipboard or "")
        elif ctrl and t in {"C", "X"}:
            context.window_manager.clipboard = buf.text
            if t == "X":
                buf.clear()
        elif ctrl and t == "A":
            buf.caret = len(buf.text)
        elif event.unicode and not ctrl and not event.alt and event.unicode.isprintable():
            buf.insert(event.unicode)
        elif ctrl or event.alt or (t.startswith("F") and t[1:].isdigit()):
            return PASS
        else:
            return EAT
        _redraw(context)
        return EAT


class BLENDER_AI_OT_focus_bar(bpy.types.Operator):
    bl_description = "Focus the Blender AI prompt bar (press again to leave it)"

    bl_idname = "blender_ai.focus_bar"
    bl_label = "Blender AI: Focus Prompt Bar"

    def invoke(self, context: Any, event: Any) -> set[str]:
        st = state.get()
        prefs = _prefs()
        if prefs and not prefs.show_bar:
            prefs.show_bar = True
        if st.modal_running:
            st.ui.focused = False
            _redraw(context)
            return {"FINISHED"}
        return bpy.ops.blender_ai.prompt_input("INVOKE_DEFAULT")


class BLENDER_AI_OT_bar_click(bpy.types.Operator):
    bl_description = "Handle a click on the Blender AI prompt bar"

    bl_idname = "blender_ai.bar_click"
    bl_label = "Blender AI: Bar Click"
    bl_options = {"INTERNAL"}

    def invoke(self, context: Any, event: Any) -> set[str]:
        prefs = _prefs()
        st = state.get()
        if prefs is None or not prefs.show_bar or st.modal_running:
            return PASS
        hit, _area, region = overlay.hit_test(context, event)
        if event.value == "RELEASE":
            return {"FINISHED"} if release_click(context, hit) else PASS
        if hit is None:
            return PASS
        if handle_click(context, hit, region):
            bpy.ops.blender_ai.prompt_input("INVOKE_DEFAULT")
        return {"FINISHED"}


class BLENDER_AI_OT_escape(bpy.types.Operator):
    bl_description = "Stop the running Blender AI request"

    bl_idname = "blender_ai.escape"
    bl_label = "Blender AI: Stop"
    bl_options = {"INTERNAL"}

    def invoke(self, context: Any, event: Any) -> set[str]:
        session = state.get().session
        if session is None or not session.busy:
            return PASS
        session.stop()
        _redraw(context)
        return {"FINISHED"}


class BLENDER_AI_OT_approve(bpy.types.Operator):
    bl_description = "Approve or deny the pending run_python call"

    bl_idname = "blender_ai.approve"
    bl_label = "Approve"
    bl_options = {"INTERNAL"}

    allow: BoolProperty(default=False)

    def execute(self, context: Any) -> set[str]:
        session = state.get().session
        if session is not None:
            session.approve(self.allow)
        _redraw(context)
        return {"FINISHED"}


CLASSES = (BLENDER_AI_OT_prompt_input, BLENDER_AI_OT_focus_bar, BLENDER_AI_OT_bar_click, BLENDER_AI_OT_escape, BLENDER_AI_OT_approve)


def register_keymaps() -> None:
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is None:
        return
    st = state.get()
    km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
    for idname, key, value, kwargs in (
        ("blender_ai.focus_bar", "SPACE", "PRESS", {"ctrl": True}),
        ("blender_ai.bar_click", "LEFTMOUSE", "PRESS", {}),
        ("blender_ai.bar_click", "LEFTMOUSE", "RELEASE", {}),
        ("blender_ai.escape", "ESC", "PRESS", {}),
    ):
        kmi = km.keymap_items.new(idname, key, value, **kwargs)
        st.keymaps.append((km, kmi))


def unregister_keymaps() -> None:
    st = state.get()
    for km, kmi in st.keymaps:
        with contextlib.suppress(ReferenceError, RuntimeError):
            km.keymap_items.remove(kmi)
    st.keymaps.clear()


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    register_keymaps()


def unregister() -> None:
    unregister_keymaps()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
