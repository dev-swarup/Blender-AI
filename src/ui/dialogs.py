from __future__ import annotations

from typing import Any

import bpy
from bpy.props import StringProperty

from .. import state
from ..agent.messages import Phase
from ..integrations import registry as helpers

PREVIEW_LINES = 18
HELPERS_DIALOG_ROWS = 7


def _session_pending(call_id: str = "") -> Any:
    session = state.get().session
    if session is None or session.phase is not Phase.APPROVAL or session.pending is None:
        return None, None
    return (None, None) if call_id and session.pending.id != call_id else (session, session.pending)


def _view3d_override() -> dict[str, Any] | None:
    wm = bpy.context.window_manager
    window = bpy.context.window or (wm.windows[0] if wm.windows else None)
    if window is None:
        return None
    areas = [a for a in window.screen.areas if a.type == "VIEW_3D"] or list(window.screen.areas)
    if not areas:
        return {"window": window}
    area = max(areas, key=lambda a: a.width * a.height)
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    return {"window": window, "area": area, "region": region} if region else {"window": window, "area": area}


def open_approval() -> None:
    _session, pending = _session_pending()
    if pending is None or bpy.app.background:
        return
    override = _view3d_override()
    if override is None:
        return
    with bpy.context.temp_override(**override):
        bpy.ops.blender_ai.approval_dialog("INVOKE_DEFAULT", call_id=pending.id)


def open_approval_later() -> None:
    if not bpy.app.background:
        bpy.app.timers.register(open_approval, first_interval=0.05)


class BLENDER_AI_OT_approval_dialog(bpy.types.Operator):
    bl_description = "Review the Python code an AI agent wants to run"
    bl_idname = "blender_ai.approval_dialog"
    bl_label = "Run Python code?"
    bl_options = {"INTERNAL"}

    call_id: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context: Any, event: Any) -> set[str]:
        _session, pending = _session_pending(self.call_id)
        if pending is None:
            return {"CANCELLED"}
        wm = context.window_manager
        width = 560
        try:
            return wm.invoke_props_dialog(self, width=width, title="Run Python code?", confirm_text="Yes, run it")
        except TypeError:
            return wm.invoke_props_dialog(self, width=width)

    def draw(self, context: Any) -> None:
        layout = self.layout
        session, pending = _session_pending(self.call_id)
        if pending is None:
            layout.label(text="This request has already been answered.", icon="INFO")
            return
        who = session.pending_agent or "The AI"
        layout.label(text=f"{who} wants to run Python code in your scene.", icon="SCRIPT")
        purpose = str(pending.input.get("purpose", "")).strip()
        if purpose:
            layout.label(text=purpose[:110])
        lines = str(pending.input.get("code", "")).replace("\t", "    ").splitlines()
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.85
        for line in lines[:PREVIEW_LINES]:
            col.label(text=line[:120] or " ")
        if len(lines) > PREVIEW_LINES:
            col.label(text=f"… {len(lines) - PREVIEW_LINES} more lines (open the Steps panel to read them all)")
        col = layout.column(align=True)
        col.label(text="Yes runs it now; Ctrl+Z undoes it.", icon="INFO")
        col.label(text="Cancel tells the AI you said no, and it carries on without it.")

    def execute(self, context: Any) -> set[str]:
        session, _pending = _session_pending(self.call_id)
        if session is not None:
            session.approve(True)
        return {"FINISHED"}

    def cancel(self, context: Any) -> None:
        session, _pending = _session_pending(self.call_id)
        if session is not None:
            session.approve(False)


def _prefs() -> Any:
    from ..preferences import get_prefs

    return get_prefs()


def open_helpers_prompt() -> None:
    prefs = _prefs()
    if bpy.app.background or prefs is None or prefs.helpers_prompted or not bpy.app.online_access or not helpers.missing():
        return
    override = _view3d_override()
    if override is not None:
        window = override["window"]
        unit = 20 * bpy.context.preferences.system.ui_scale
        window.cursor_warp(window.width // 2 - round(2.5 * unit), window.height // 2 + round(HELPERS_DIALOG_ROWS * unit / 2))
        with bpy.context.temp_override(**override):
            bpy.ops.blender_ai.install_helpers("INVOKE_DEFAULT")


def open_helpers_prompt_later() -> None:
    if not bpy.app.background:
        bpy.app.timers.register(open_helpers_prompt, first_interval=1.5)


def _helpers_finished(missing: list[str]) -> None:
    session = state.get().session
    if session is not None:
        text = f"Couldn't install {', '.join(missing)}; see the Info editor." if missing else "Helper add-ons installed and enabled."
        session.notify(text, error=bool(missing))
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


class BLENDER_AI_OT_install_helpers(bpy.types.Operator):
    bl_idname = "blender_ai.install_helpers"
    bl_label = "Install helper add-ons"
    bl_description = "Install and enable the add-ons Blender AI builds with: Rigify, MPFB, LoopTools and Extra Mesh Objects"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: Any) -> bool:
        return bool(helpers.missing())

    def invoke(self, context: Any, event: Any) -> set[str]:
        wm = context.window_manager
        try:
            return wm.invoke_props_dialog(self, width=520, title="Install helper add-ons?", confirm_text="Install")
        except TypeError:
            return wm.invoke_props_dialog(self, width=520)

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.label(text="Blender AI builds better models, rigs and characters with these add-ons:", icon="PACKAGE")
        col = layout.box().column(align=True)
        for helper in helpers.HELPERS.values():
            ready = helpers.enabled_module(helper.id) is not None
            source = "included with Blender" if helper.bundled else f"extensions.blender.org, {helper.size}"
            col.label(text=f"{helper.name}: {'enabled' if ready else source}", icon="CHECKBOX_HLT" if ready else "CHECKBOX_DEHLT")
        col = layout.column(align=True)
        col.label(text="Install downloads the missing ones from extensions.blender.org and enables them all.", icon="INFO")
        col.label(text="Cancel skips this; you can install them later in Blender AI's preferences.")

    def execute(self, context: Any) -> set[str]:
        from ..integrations import installer

        prefs = _prefs()
        if prefs is not None:
            prefs.helpers_prompted = True
        try:
            enabled, downloading = installer.install(interactive=context.window is not None)
        except installer.InstallError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if downloading:
            installer.watch(_helpers_finished)
            self.report({"INFO"}, (f"Enabled {', '.join(enabled)}. " if enabled else "") + f"Installing {', '.join(downloading)}…")
        else:
            _helpers_finished([h.name for h in helpers.missing()])
        return {"FINISHED"}

    def cancel(self, context: Any) -> None:
        prefs = _prefs()
        if prefs is not None:
            prefs.helpers_prompted = True


CLASSES = (BLENDER_AI_OT_approval_dialog, BLENDER_AI_OT_install_helpers)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
