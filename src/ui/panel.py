from __future__ import annotations

from typing import Any

import bpy

from .. import providers, state
from ..agent.messages import LogKind, Phase, Status
from . import icons

KIND_ICONS = {
    LogKind.USER: "USER",
    LogKind.ASSISTANT: "OUTLINER_DATA_GP_LAYER",
    LogKind.REASONING: "LIGHT",
    LogKind.TOOL: "TOOL_SETTINGS",
    LogKind.ERROR: "ERROR",
    LogKind.INFO: "INFO",
    LogKind.APPROVAL: "SCRIPT",
    LogKind.QUESTION: "QUESTION",
    LogKind.AGENT: "COMMUNITY",
}
STATUS_ICONS = {Status.RUNNING: "SORTTIME", Status.WAITING: "PAUSE", Status.ERROR: "CANCEL", Status.CANCELLED: "X"}


def _prefs() -> Any:
    from ..preferences import get_prefs

    return get_prefs()


class BLENDER_AI_MT_presets(bpy.types.Menu):
    bl_idname = "BLENDER_AI_MT_presets"
    bl_label = "Providers"

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.label(text="Fill in the API URL for:")
        for provider, label, url in providers.PRESETS:
            op = layout.operator("blender_ai.use_preset", text=label)
            op.provider, op.url = provider, url


class BLENDER_AI_MT_reasoning(bpy.types.Menu):
    bl_idname = "BLENDER_AI_MT_reasoning"
    bl_label = "Reasoning"

    def draw(self, context: Any) -> None:
        from ..preferences import get_prefs

        prefs = get_prefs()
        if prefs is not None:
            for level in ("auto", *prefs.reasoning_levels()):
                self.layout.prop_enum(prefs, "reasoning", level)


class BLENDER_AI_UL_mcp_servers(bpy.types.UIList):
    bl_idname = "BLENDER_AI_UL_mcp_servers"

    def draw_item(
        self, context: Any, layout: Any, data: Any, item: Any, icon: Any, active_data: Any, active_property: str, index: int = 0, flt_flag: int = 0
    ) -> None:
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.label(text=item.name or "server", icon="URL" if item.kind == "URL" else "CONSOLE")
        manager = state.get().mcp
        status = manager.status().get(item.name, "") if manager is not None else ""
        if status:
            sub = row.row()
            sub.alert = status.startswith("failed")
            sub.label(text=status.split(" (log:")[0][:40])


def _draw_tabs(layout: Any, prefs: Any) -> None:
    layout.row().prop(prefs, "mode", expand=True)


class BLENDER_AI_PT_quick_settings(bpy.types.Panel):
    bl_idname = "BLENDER_AI_PT_quick_settings"
    bl_label = "Blender AI"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 16

    def draw(self, context: Any) -> None:
        from ..preferences import draw_model_rows

        prefs = _prefs()
        layout = self.layout
        layout.label(text="Blender AI", icon_value=icons.icon("logo") or 0)
        layout.row().prop(prefs, "provider", expand=True)
        draw_model_rows(layout.column(), prefs)
        col = layout.column(align=True)
        col.prop(prefs, "auto_screenshot")
        col.prop(prefs, "confirm_python")
        col.prop(prefs, "pack_textures")
        col.prop(prefs, "show_bar")
        row = layout.row(align=True)
        row.operator("blender_ai.new_conversation", icon="ADD")
        row.operator("blender_ai.open_settings", icon="PREFERENCES")


class BLENDER_AI_PT_main(bpy.types.Panel):
    bl_idname = "BLENDER_AI_PT_main"
    bl_label = "Blender AI"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender AI"

    def draw_header(self, context: Any) -> None:
        icon_id = icons.icon("logo")
        if icon_id:
            self.layout.label(icon_value=icon_id)

    def draw(self, context: Any) -> None:
        from ..preferences import draw_model_rows

        prefs = _prefs()
        st = state.get()
        session = st.session
        layout = self.layout
        if prefs is None or session is None:
            layout.label(text="Blender AI isn't initialised.", icon="ERROR")
            return

        _draw_tabs(layout, prefs)
        draw_model_rows(layout.column(), prefs)

        box = layout.box()
        text = st.ui.buffer.text.strip()
        box.label(text=(text[:60] + "…") if len(text) > 60 else (text or "Prompt is empty"), icon="GREASEPENCIL")
        row = box.row(align=True)
        row.operator("blender_ai.focus_bar", text="Type in prompt bar", icon="CONSOLE")
        if session.busy:
            row.operator("blender_ai.stop", text="", icon="CANCEL")
        else:
            row.operator("blender_ai.send", text="", icon="PLAY")
        if st.ui.attachment_path:
            row = box.row(align=True)
            row.label(text=bpy.path.basename(st.ui.attachment_path), icon="IMAGE_DATA")
            row.operator("blender_ai.clear_attachment", text="", icon="X")
        else:
            box.operator("blender_ai.attach_image", icon="IMAGE_REFERENCE")

        if session.phase is Phase.APPROVAL and session.pending is not None:
            box = layout.box()
            box.alert = True
            box.label(text="Approve running this Python?", icon="SCRIPT")
            purpose = str(session.pending.input.get("purpose", ""))
            if purpose:
                box.label(text=purpose[:80])
            col = box.column(align=True)
            for line in str(session.pending.input.get("code", "")).splitlines()[:24]:
                col.label(text=line.replace("\t", "    ") or " ")
            row = box.row(align=True)
            row.operator("blender_ai.approval_dialog", text="Review…", icon="ZOOM_ALL").call_id = session.pending.id
            row.operator("blender_ai.approve", text="Run", icon="CHECKMARK").allow = True
            row.operator("blender_ai.approve", text="Deny", icon="X").allow = False
        elif session.phase is Phase.QUESTION and session.pending is not None:
            box = layout.box()
            box.label(text=str(session.pending.input.get("question", ""))[:120], icon="QUESTION")
            box.label(text="Answer in the prompt bar.")

        row = layout.row(align=True)
        row.operator("blender_ai.new_conversation", icon="ADD")
        row.operator("blender_ai.open_settings", text="", icon="PREFERENCES")
        if not bpy.app.online_access:
            layout.label(text="Enable online access in Preferences > System", icon="ERROR")


class BLENDER_AI_PT_log(bpy.types.Panel):
    bl_idname = "BLENDER_AI_PT_log"
    bl_label = "Steps"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender AI"
    bl_parent_id = "BLENDER_AI_PT_main"

    def draw(self, context: Any) -> None:
        session = state.get().session
        layout = self.layout
        if session is None or not session.log:
            layout.label(text="No steps yet.")
            return
        col = layout.column(align=True)
        for entry in session.log[-40:]:
            row = col.row(align=True)
            icon = STATUS_ICONS.get(entry.status, KIND_ICONS[entry.kind])
            label = entry.text.strip().splitlines()[0] if entry.text.strip() else entry.kind.value
            if entry.agent and entry.kind is not LogKind.AGENT:
                label = f"{entry.agent.split(' · ', 1)[-1]} › {label}"
            row.alert = entry.status is Status.ERROR or entry.destructive
            row.label(text=label[:70], icon=icon)
            body = "\n".join(part for part in (entry.detail.strip(), entry.result.strip()) if part)
            if body:
                op = row.operator("blender_ai.toggle_entry", text="", icon="DISCLOSURE_TRI_DOWN" if entry.expanded else "DISCLOSURE_TRI_RIGHT", emboss=False)
                op.entry_id = entry.id
                if entry.expanded:
                    for line in body.splitlines()[:16]:
                        col.label(text="    " + line[:90])
        usage = session.usage
        layout.label(text=f"{usage.turns} requests · {usage.input_tokens:,} in · {usage.output_tokens:,} out", icon="INFO")
        layout.operator("blender_ai.clear_log", icon="TRASH")


CLASSES = (BLENDER_AI_MT_presets, BLENDER_AI_MT_reasoning, BLENDER_AI_UL_mcp_servers, BLENDER_AI_PT_quick_settings, BLENDER_AI_PT_main, BLENDER_AI_PT_log)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
