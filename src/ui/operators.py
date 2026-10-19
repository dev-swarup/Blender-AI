from __future__ import annotations

import threading
from typing import Any

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from .. import models, providers, state
from ..agent.config import AgentConfig
from ..agent.mcp_bridge import format_pairs, parse_config_json
from ..imaging import client as image_api
from .input_modal import submit

REFRESH = "__refresh__"


def _prefs() -> Any:
    from ..preferences import get_prefs

    return get_prefs()


def _redraw() -> None:
    from ..agent.host import redraw

    redraw()


def _in_background(fn: Any, done: Any) -> None:
    bridge = state.get().bridge

    def work() -> None:
        try:
            result = fn()
        except Exception as exc:
            bridge.post(done, None, exc)
        else:
            bridge.post(done, result, None)

    threading.Thread(target=work, name="blender_ai-settings", daemon=True).start()


def _can_fetch(url: str, key: str) -> bool:
    local = any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    return bool(bpy.app.online_access) and bool(url) and (bool(key) or local)


def refresh_model_lists(force: bool = False) -> None:
    prefs = _prefs()
    st = state.get()
    if prefs is None:
        return
    chat = prefs.chat_endpoint()
    image = prefs.image_endpoint()
    same = image is not None and prefs.provider == models.OPENAI and image == (chat[1], chat[2])

    def fetch(target: state.ModelList, endpoint: tuple[Any, ...], provider: str, url: str, key: str, both: bool) -> None:
        if not force and target.endpoint == endpoint and (target.models or target.loading):
            return
        if not _can_fetch(url, key):
            target.endpoint, target.models, target.error = endpoint, [], ""
            return
        target.endpoint, target.loading, target.error = endpoint, True, ""

        def done(listed: list[models.ModelInfo] | None, exc: BaseException | None) -> None:
            if target.endpoint != endpoint:
                return
            target.loading = False
            target.models = listed or []
            target.error = f"Couldn't load models: {exc}" if exc is not None else ""
            if both:
                st.image_models.endpoint, st.image_models.models = ("image", url, key), target.models
                st.image_models.loading, st.image_models.error = False, target.error
            _apply_defaults()
            _redraw()

        _in_background(lambda: providers.list_models(provider, url, key), done)

    fetch(st.chat_models, chat, chat[0], chat[1], chat[2], same)
    if image is not None and not same:
        fetch(st.image_models, ("image", *image), models.OPENAI, image[0], image[1], False)
    elif image is None:
        st.image_models.endpoint, st.image_models.models = (), []
    if same:
        st.image_models.endpoint = ("image", chat[1], chat[2])
        st.image_models.models = st.chat_models.models
        st.image_models.loading, st.image_models.error = st.chat_models.loading, st.chat_models.error


def _apply_defaults() -> None:
    prefs = _prefs()
    st = state.get()
    if prefs is None:
        return
    if not prefs.model() and st.chat_models.models:
        prefs.set_model(models.default_chat(st.chat_models.models))
    if not prefs.image_model and st.image_models.models:
        prefs.image_model = models.default_image(st.image_models.models)


def _items(listed: list[models.ModelInfo], current: str, key: str) -> list[tuple[Any, ...]]:
    items: list[tuple[Any, ...]] = []
    ids = {m.id for m in listed}
    if current and current not in ids:
        items.append((current, current, "Current model (not in the endpoint's list)", "CHECKMARK", 0))
    for i, info in enumerate(listed, start=1):
        label = info.label if info.label == info.id else f"{info.label}  ({info.id})"
        if info.free:
            label += "  · free"
        items.append((info.id, label, info.id, "CHECKMARK" if info.id == current else "NONE", i))
    items.append((REFRESH, "↻ Reload the list from the API", "", "FILE_REFRESH", 99999))
    state.get().enum_cache[key] = items
    return items


def _chat_items(_self: Any, _context: Any) -> list[tuple[Any, ...]]:
    prefs = _prefs()
    st = state.get()
    return _items(models.chat_models(st.chat_models.models), prefs.model() if prefs else "", "chat")


def _image_items(_self: Any, _context: Any) -> list[tuple[Any, ...]]:
    prefs = _prefs()
    st = state.get()
    return _items(models.image_models(st.image_models.models), prefs.image_model if prefs else "", "image")


class BLENDER_AI_OT_choose_model(bpy.types.Operator):
    bl_description = "Pick the AI model (searchable list from your API's /models)"
    bl_idname = "blender_ai.choose_model"
    bl_label = "Choose Model"
    bl_property = "choice"

    choice: EnumProperty(items=_chat_items)

    def invoke(self, context: Any, event: Any) -> set[str]:
        st = state.get()
        if not st.chat_models.models and not st.chat_models.loading:
            refresh_model_lists(force=True)
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Any) -> set[str]:
        if self.choice == REFRESH:
            refresh_model_lists(force=True)
        else:
            _prefs().set_model(self.choice)
        _redraw()
        return {"FINISHED"}


class BLENDER_AI_OT_choose_image_model(bpy.types.Operator):
    bl_description = "Pick the image model used for generated textures"
    bl_idname = "blender_ai.choose_image_model"
    bl_label = "Choose Image Model"
    bl_property = "choice"

    choice: EnumProperty(items=_image_items)

    def invoke(self, context: Any, event: Any) -> set[str]:
        if _prefs().image_endpoint() is None:
            self.report({"WARNING"}, "Set a separate image API in Blender AI settings first")
            return {"CANCELLED"}
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Any) -> set[str]:
        if self.choice == REFRESH:
            refresh_model_lists(force=True)
        else:
            _prefs().image_model = self.choice
        _redraw()
        return {"FINISHED"}


class BLENDER_AI_OT_refresh_models(bpy.types.Operator):
    bl_description = "Reload the model lists from the API"
    bl_idname = "blender_ai.refresh_models"
    bl_label = "Refresh Models"

    def execute(self, context: Any) -> set[str]:
        if not bpy.app.online_access:
            self.report({"ERROR"}, "Online access is disabled in Preferences > System")
            return {"CANCELLED"}
        refresh_model_lists(force=True)
        return {"FINISHED"}


class BLENDER_AI_OT_use_preset(bpy.types.Operator):
    bl_description = "Use this provider's API URL"
    bl_idname = "blender_ai.use_preset"
    bl_label = "Use Provider"
    bl_options = {"INTERNAL"}

    provider: StringProperty()
    url: StringProperty()

    def execute(self, context: Any) -> set[str]:
        prefs = _prefs()
        prefs.provider = self.provider
        if self.provider == models.ANTHROPIC:
            prefs.anthropic_url = self.url
        else:
            prefs.openai_url = self.url
            prefs.openai_model = ""
        prefs.image_model = ""
        refresh_model_lists(force=True)
        return {"FINISHED"}


class BLENDER_AI_OT_test_llm(bpy.types.Operator):
    bl_description = "Check the URL and key, load the model list and send a one-word request"
    bl_idname = "blender_ai.test_llm"
    bl_label = "Test Connection"

    def execute(self, context: Any) -> set[str]:
        prefs = _prefs()
        st = state.get()
        if not bpy.app.online_access:
            st.connection["llm"] = "Online access is disabled in Preferences > System"
            return {"CANCELLED"}
        cfg: AgentConfig = prefs.agent_config()
        endpoint = prefs.chat_endpoint()
        st.connection["llm"] = "Testing…"

        def done(result: tuple[str, list[models.ModelInfo]] | None, exc: BaseException | None) -> None:
            if exc is not None:
                st.connection["llm"] = f"Failed: {exc}"
            elif result is not None:
                message, listed = result
                st.connection["llm"] = message
                if listed:
                    st.chat_models.endpoint, st.chat_models.models, st.chat_models.error = endpoint, listed, ""
                    if prefs.provider == models.OPENAI and not prefs.image_url.strip():
                        st.image_models.endpoint = ("image", endpoint[1], endpoint[2])
                        st.image_models.models = listed
                    _apply_defaults()
            _redraw()

        _in_background(lambda: providers.test_connection(cfg), done)
        return {"FINISHED"}


class BLENDER_AI_OT_test_image(bpy.types.Operator):
    bl_description = "Check the image API URL and key"
    bl_idname = "blender_ai.test_image"
    bl_label = "Test Image API"

    def execute(self, context: Any) -> set[str]:
        prefs = _prefs()
        st = state.get()
        if not bpy.app.online_access:
            st.connection["image"] = "Online access is disabled in Preferences > System"
            return {"CANCELLED"}
        cfg = prefs.image_config()
        if cfg is None:
            st.connection["image"] = "Enter an image API URL first."
            return {"CANCELLED"}
        endpoint = ("image", cfg.base_url, cfg.api_key)
        st.connection["image"] = "Testing…"

        def done(result: tuple[str, list[models.ModelInfo]] | None, exc: BaseException | None) -> None:
            if exc is not None:
                st.connection["image"] = f"Failed: {exc}"
            elif result is not None:
                st.connection["image"], listed = result
                st.image_models.endpoint, st.image_models.models = endpoint, listed
                _apply_defaults()
            _redraw()

        _in_background(lambda: image_api.test_connection(cfg), done)
        return {"FINISHED"}


def connect_mcp(reconnect: bool = False) -> None:
    prefs = _prefs()
    st = state.get()
    if prefs is None or st.mcp is None:
        return
    configs = prefs.mcp_configs()
    if not configs:
        st.mcp.sync([])
        st.connection["mcp"] = ""
        return
    if not bpy.app.online_access and any(c.url for c in configs):
        st.connection["mcp"] = "Online access is off, so URL servers can't connect."
    (st.mcp.reconnect if reconnect else st.mcp.sync)(configs)


class BLENDER_AI_OT_mcp_add(bpy.types.Operator):
    bl_description = "Add an MCP server"
    bl_idname = "blender_ai.mcp_add"
    bl_label = "Add MCP Server"

    def execute(self, context: Any) -> set[str]:
        prefs = _prefs()
        item = prefs.mcp_servers.add()
        item.name = f"server{len(prefs.mcp_servers)}"
        prefs.mcp_index = len(prefs.mcp_servers) - 1
        return {"FINISHED"}


class BLENDER_AI_OT_mcp_remove(bpy.types.Operator):
    bl_description = "Remove the selected MCP server"
    bl_idname = "blender_ai.mcp_remove"
    bl_label = "Remove MCP Server"

    def execute(self, context: Any) -> set[str]:
        prefs = _prefs()
        if 0 <= prefs.mcp_index < len(prefs.mcp_servers):
            prefs.mcp_servers.remove(prefs.mcp_index)
            prefs.mcp_index = max(0, prefs.mcp_index - 1)
            connect_mcp()
        return {"FINISHED"}


class BLENDER_AI_OT_mcp_import(bpy.types.Operator):
    bl_description = (
        'Add servers from an MCP JSON config on the clipboard (the "mcpServers" snippet that MCP servers publish for Claude Desktop, Cursor or VS Code)'
    )

    bl_idname = "blender_ai.mcp_import"
    bl_label = "Paste MCP JSON Config"

    def execute(self, context: Any) -> set[str]:
        prefs = _prefs()
        try:
            configs = parse_config_json(context.window_manager.clipboard or "")
        except ValueError as exc:
            self.report({"ERROR"}, f"Couldn't read MCP config from the clipboard: {exc}")
            return {"CANCELLED"}
        names = {item.name for item in prefs.mcp_servers}
        for config in configs:
            existing = next((i for i in prefs.mcp_servers if i.name == config.name), None)
            item = existing or prefs.mcp_servers.add()
            item.name = config.name if config.name not in names or existing else f"{config.name}2"
            item.kind = "URL" if config.url else "COMMAND"
            item.command = config.command
            item.url = config.url
            item.env = format_pairs(config.env)
            item.headers = format_pairs(config.headers, ": ")
            item.enabled = True
        prefs.mcp_index = len(prefs.mcp_servers) - 1
        connect_mcp(reconnect=True)
        self.report({"INFO"}, f"Added {len(configs)} MCP server(s)")
        return {"FINISHED"}


class BLENDER_AI_OT_mcp_connect(bpy.types.Operator):
    bl_description = "Connect (or reconnect) the enabled MCP servers. New tools apply to new conversations"
    bl_idname = "blender_ai.mcp_connect"
    bl_label = "Connect MCP Servers"

    def execute(self, context: Any) -> set[str]:
        connect_mcp(reconnect=True)
        return {"FINISHED"}


class BLENDER_AI_OT_attach_image(bpy.types.Operator):
    bl_description = "Attach a reference image to the next prompt (texture tools can use it as 'attached')"
    bl_idname = "blender_ai.attach_image"
    bl_label = "Attach Image"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_image: BoolProperty(default=True, options={"HIDDEN"})
    filter_folder: BoolProperty(default=True, options={"HIDDEN"})

    def invoke(self, context: Any, event: Any) -> set[str]:
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Any) -> set[str]:
        state.get().ui.attachment_path = bpy.path.abspath(self.filepath)
        _redraw()
        return {"FINISHED"}


class BLENDER_AI_OT_clear_attachment(bpy.types.Operator):
    bl_description = "Remove the attached image"
    bl_idname = "blender_ai.clear_attachment"
    bl_label = "Remove Attachment"

    def execute(self, context: Any) -> set[str]:
        state.get().ui.attachment_path = None
        _redraw()
        return {"FINISHED"}


class BLENDER_AI_OT_send(bpy.types.Operator):
    bl_description = "Send the prompt bar text"
    bl_idname = "blender_ai.send"
    bl_label = "Generate"

    def execute(self, context: Any) -> set[str]:
        submit(context)
        return {"FINISHED"}


class BLENDER_AI_OT_stop(bpy.types.Operator):
    bl_description = "Stop the running request"
    bl_idname = "blender_ai.stop"
    bl_label = "Stop"

    def execute(self, context: Any) -> set[str]:
        session = state.get().session
        if session is not None:
            session.stop()
        _redraw()
        return {"FINISHED"}


class BLENDER_AI_OT_new_conversation(bpy.types.Operator):
    bl_description = "Forget the conversation so far (the scene is kept)"
    bl_idname = "blender_ai.new_conversation"
    bl_label = "New Conversation"

    def execute(self, context: Any) -> set[str]:
        session = state.get().session
        if session is not None:
            session.new_thread()
        _redraw()
        return {"FINISHED"}


class BLENDER_AI_OT_clear_log(bpy.types.Operator):
    bl_description = "Clear the step log"
    bl_idname = "blender_ai.clear_log"
    bl_label = "Clear Log"

    def execute(self, context: Any) -> set[str]:
        session = state.get().session
        if session is not None:
            session.clear_log()
        return {"FINISHED"}


class BLENDER_AI_OT_toggle_entry(bpy.types.Operator):
    bl_description = "Show or hide details"
    bl_idname = "blender_ai.toggle_entry"
    bl_label = "Toggle Details"
    bl_options = {"INTERNAL"}

    entry_id: IntProperty()

    def execute(self, context: Any) -> set[str]:
        session = state.get().session
        for entry in session.log if session else []:
            if entry.id == self.entry_id:
                entry.expanded = not entry.expanded
        _redraw()
        return {"FINISHED"}


class BLENDER_AI_OT_open_settings(bpy.types.Operator):
    bl_description = "Open Blender AI in Preferences"
    bl_idname = "blender_ai.open_settings"
    bl_label = "All Settings…"

    def execute(self, context: Any) -> set[str]:
        from ..preferences import PACKAGE

        bpy.ops.preferences.addon_show(module=PACKAGE)
        return {"FINISHED"}


CLASSES = (
    BLENDER_AI_OT_choose_model,
    BLENDER_AI_OT_choose_image_model,
    BLENDER_AI_OT_refresh_models,
    BLENDER_AI_OT_use_preset,
    BLENDER_AI_OT_test_llm,
    BLENDER_AI_OT_test_image,
    BLENDER_AI_OT_mcp_add,
    BLENDER_AI_OT_mcp_remove,
    BLENDER_AI_OT_mcp_import,
    BLENDER_AI_OT_mcp_connect,
    BLENDER_AI_OT_attach_image,
    BLENDER_AI_OT_clear_attachment,
    BLENDER_AI_OT_send,
    BLENDER_AI_OT_stop,
    BLENDER_AI_OT_new_conversation,
    BLENDER_AI_OT_clear_log,
    BLENDER_AI_OT_toggle_entry,
    BLENDER_AI_OT_open_settings,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
