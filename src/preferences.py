from __future__ import annotations

import os
from typing import Any

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, PropertyGroup

from . import models, state
from .agent.config import AgentConfig
from .providers import ANTHROPIC_URL, OPENAI_URL
from .tools.registry import Mode

PACKAGE = __package__ or "blender_ai"

MODE_ITEMS = [
    ("MODEL", "3D Model", "Build one asset: modelling, UVs, materials, rigging", "MESH_CUBE", 0),
    ("SCENE", "Scene", "Compose a scene: layout, lights, camera, world", "SCENE_DATA", 1),
    ("ANIMATE", "Animation", "Rig and animate: keyframes, cycles, cameras, physics", "ARMATURE_DATA", 2),
    ("MATERIAL", "Material", "Materials and PBR textures", "MATERIAL", 3),
]


REASONING_ITEMS = [(level, models.LEVEL_LABELS[level], "") for level in models.REASONING_LEVELS]


def get_prefs() -> BlenderAIPreferences | None:
    addon = bpy.context.preferences.addons.get(PACKAGE)
    return addon.preferences if addon else None


def _redraw(_self: Any = None, context: Any = None) -> None:
    for window in getattr(bpy.context.window_manager, "windows", []):
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "PREFERENCES"}:
                area.tag_redraw()


def _endpoint_changed(self: Any, context: Any) -> None:
    from .ui import operators

    operators.refresh_model_lists(force=True)
    _redraw()


def _mode_changed(self: BlenderAIPreferences, context: Any) -> None:
    session = state.get().session
    if session is not None:
        session.set_mode(self.current_mode())
    _redraw()


def _default_cache_dir() -> str:
    try:
        return bpy.utils.extension_path_user(PACKAGE, path="textures", create=True)
    except (ValueError, AttributeError):
        return os.path.join(bpy.utils.user_resource("DATAFILES", path="blender_ai", create=True), "textures")


class BlenderAIMCPServer(PropertyGroup):
    name: StringProperty(name="Name", default="server")
    enabled: BoolProperty(name="Enabled", default=True)
    kind: EnumProperty(
        name="Type",
        items=[("COMMAND", "Command", "Start a server program on this computer (stdio)"), ("URL", "URL", "Connect to a server on the internet (HTTP)")],
        default="COMMAND",
    )
    command: StringProperty(name="Command", description="The command that starts the server, for example: uvx some-mcp-server")
    url: StringProperty(name="URL", description="The server's MCP URL")
    env: StringProperty(name="Keys / environment", description="Environment variables the server needs, like API keys: NAME=value; NAME2=value")
    headers: StringProperty(name="Headers", description="HTTP headers for URL servers: Authorization: Bearer xyz; Other: value")

    def config(self) -> Any:
        from .agent.mcp_bridge import ServerConfig, parse_pairs

        name = self.name.strip() or "server"
        if self.kind == "URL":
            return ServerConfig(name, url=self.url.strip(), headers=parse_pairs(self.headers, ":")) if self.url.strip() else None
        return ServerConfig(name, command=self.command.strip(), env=parse_pairs(self.env)) if self.command.strip() else None


class BlenderAIPreferences(AddonPreferences):
    bl_idname = PACKAGE

    provider: EnumProperty(
        name="AI provider",
        items=[
            (models.OPENAI, "OpenAI-compatible", "OpenAI, OpenRouter, Gemini, Groq, DeepSeek, Mistral, Ollama, LM Studio and any OpenAI-style API"),
            (models.ANTHROPIC, "Anthropic", "Claude through the Anthropic API"),
        ],
        default=models.OPENAI,
        update=_endpoint_changed,
    )
    openai_url: StringProperty(name="API URL", default=OPENAI_URL, update=_endpoint_changed)
    openai_key: StringProperty(name="API Key", subtype="PASSWORD", update=_endpoint_changed)
    openai_model: StringProperty(name="Model", update=_redraw)
    anthropic_url: StringProperty(name="API URL", default=ANTHROPIC_URL, update=_endpoint_changed)
    anthropic_key: StringProperty(name="API Key", subtype="PASSWORD", update=_endpoint_changed)
    anthropic_model: StringProperty(name="Model", update=_redraw)
    reasoning: EnumProperty(
        name="Reasoning",
        description="How much the model thinks before acting. Auto uses the model's default",
        items=REASONING_ITEMS,
        default="auto",
        update=_redraw,
    )

    image_model: StringProperty(name="Image model", update=_redraw)
    image_url: StringProperty(
        name="Image API URL", description="Optional. Leave empty to use the main API (OpenAI-compatible provider only)", update=_endpoint_changed
    )
    image_key: StringProperty(name="Image API Key", subtype="PASSWORD", update=_endpoint_changed)

    mcp_servers: CollectionProperty(type=BlenderAIMCPServer)
    mcp_index: IntProperty(default=0)

    mode: EnumProperty(name="Mode", items=MODE_ITEMS, default="MODEL", update=_mode_changed)
    show_bar: BoolProperty(name="Show prompt bar", default=True, update=_redraw)
    confirm_python: BoolProperty(
        name="Ask before running Python", description="Show each Python script the AI writes and wait for your OK before it runs", default=False
    )
    auto_screenshot: BoolProperty(
        name="Let the AI see the viewport",
        description="After steps that change the scene, send a viewport screenshot so the AI can check its work",
        default=True,
    )
    screenshot_size: IntProperty(name="Screenshot width", default=768, min=256, max=1568)
    max_steps: IntProperty(name="Max steps", description="Model turns per request before it pauses", default=60, min=1, max=500)
    use_fallbacks: BoolProperty(
        name="Refusal fallback (api.anthropic.com)",
        description="If a safety classifier declines a request, let the API retry it on a fallback model",
        default=True,
    )
    cache_dir: StringProperty(name="Texture folder", subtype="DIR_PATH", default="")
    pack_textures: BoolProperty(
        name="Pack textures into the .blend",
        description="Store the images the AI makes or imports inside the .blend file, so it opens anywhere without missing textures",
        default=True,
    )
    normal_convention: EnumProperty(name="Normal maps", items=[("OPENGL", "OpenGL (Blender)", ""), ("DIRECTX", "DirectX", "")], default="OPENGL")

    show_images: BoolProperty(name="Separate image API", default=False)
    show_mcp: BoolProperty(name="MCP servers", default=False)
    show_more: BoolProperty(name="More options", default=False)

    def current_mode(self) -> Mode:
        return Mode(self.mode)

    @property
    def is_anthropic(self) -> bool:
        return self.provider == models.ANTHROPIC

    def api_url(self) -> str:
        if self.is_anthropic:
            return self.anthropic_url.strip() or os.environ.get("ANTHROPIC_BASE_URL", "") or ANTHROPIC_URL
        return self.openai_url.strip() or os.environ.get("OPENAI_BASE_URL", "") or OPENAI_URL

    def api_key(self) -> str:
        if self.is_anthropic:
            return self.anthropic_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
        return self.openai_key.strip() or os.environ.get("OPENAI_API_KEY", "")

    def model(self) -> str:
        return (self.anthropic_model if self.is_anthropic else self.openai_model).strip()

    def set_model(self, model_id: str) -> None:
        if self.is_anthropic:
            self.anthropic_model = model_id
        else:
            self.openai_model = model_id

    def chat_endpoint(self) -> tuple[str, str, str]:
        return (self.provider, self.api_url(), self.api_key())

    def image_endpoint(self) -> tuple[str, str] | None:
        if self.image_url.strip():
            return self.image_url.strip(), self.image_key.strip() or (self.api_key() if not self.is_anthropic else "")
        return None if self.is_anthropic else (self.api_url(), self.api_key())

    def agent_config(self) -> AgentConfig:
        listed = state.get().chat_models
        info = models.find(listed.models, self.model()) if listed.endpoint == self.chat_endpoint() else None
        return AgentConfig(
            provider=self.provider,
            base_url=self.api_url(),
            api_key=self.api_key(),
            model=self.model(),
            reasoning=self.reasoning,
            max_steps=self.max_steps,
            model_info=info,
            send_images=self.auto_screenshot,
            use_fallbacks=self.use_fallbacks,
        )

    def image_config(self) -> Any:
        from .imaging.client import ImageConfig

        endpoint = self.image_endpoint()
        if endpoint is None:
            return None
        model = self.image_model.strip()
        info = models.find(state.get().image_models.models, model)
        image_only = info is not None and info.outputs == ("image",)
        return ImageConfig(base_url=endpoint[0], api_key=endpoint[1], model=model, image_only=image_only)

    def mcp_configs(self) -> list[Any]:
        configs = []
        for item in self.mcp_servers:
            config = item.config() if item.enabled else None
            if config is not None:
                configs.append(config)
        return configs

    def texture_dir(self) -> str:
        path = bpy.path.abspath(self.cache_dir) if self.cache_dir else _default_cache_dir()
        os.makedirs(path, exist_ok=True)
        return path

    def draw(self, context: Any) -> None:
        layout = self.layout
        st = state.get()

        box = layout.box()
        box.row().prop(self, "provider", expand=True)
        col = box.column()
        row = col.row(align=True)
        row.prop(self, "anthropic_url" if self.is_anthropic else "openai_url")
        row.menu("BLENDER_AI_MT_presets", text="", icon="DOWNARROW_HLT")
        col.prop(self, "anthropic_key" if self.is_anthropic else "openai_key")
        row = col.row()
        row.operator("blender_ai.test_llm", icon="LINKED")
        if st.connection.get("llm"):
            row.label(text=st.connection["llm"])
        draw_model_rows(col, self)
        env = "ANTHROPIC_API_KEY" if self.is_anthropic else "OPENAI_API_KEY"
        if not (self.anthropic_key if self.is_anthropic else self.openai_key) and os.environ.get(env):
            col.label(text=f"Using {env} from the environment.", icon="INFO")

        _section(layout, self, "show_images", "Separate image API (optional)", "IMAGE_DATA", self._draw_images)
        count = len(self.mcp_servers)
        _section(layout, self, "show_mcp", f"MCP servers ({count})", "PLUGIN", self._draw_mcp)
        _section(layout, self, "show_more", "More options", "PREFERENCES", self._draw_more)

        col = layout.column()
        col.label(text="Keys are stored in plain text in your Blender preferences, never in .blend files.", icon="INFO")
        if not bpy.app.online_access:
            col.label(text="Online access is off: enable it in Preferences > System > Network.", icon="ERROR")

    def _draw_images(self, col: Any) -> None:
        col.label(text="Only needed for Anthropic, or to make images with a different service.")
        col.prop(self, "image_url")
        col.prop(self, "image_key")
        row = col.row()
        row.operator("blender_ai.test_image", icon="LINKED")
        if state.get().connection.get("image"):
            row.label(text=state.get().connection["image"])

    def _draw_mcp(self, col: Any) -> None:
        col.label(text="Add tools from MCP servers, like Meshy or Tripo for AI-generated 3D models.")
        row = col.row()
        row.template_list("BLENDER_AI_UL_mcp_servers", "", self, "mcp_servers", self, "mcp_index", rows=3)
        side = row.column(align=True)
        side.operator("blender_ai.mcp_add", text="", icon="ADD")
        side.operator("blender_ai.mcp_remove", text="", icon="REMOVE")
        side.separator()
        side.operator("blender_ai.mcp_import", text="", icon="PASTEDOWN")
        if 0 <= self.mcp_index < len(self.mcp_servers):
            item = self.mcp_servers[self.mcp_index]
            sub = col.column()
            sub.prop(item, "name")
            sub.row().prop(item, "kind", expand=True)
            if item.kind == "URL":
                sub.prop(item, "url")
                sub.prop(item, "headers")
            else:
                sub.prop(item, "command")
                sub.prop(item, "env")
        row = col.row(align=True)
        row.operator("blender_ai.mcp_import", text="Paste JSON config", icon="PASTEDOWN")
        row.operator("blender_ai.mcp_connect", text="Connect", icon="FILE_REFRESH")
        manager = state.get().mcp
        for name, status in (manager.status() if manager is not None else {}).items():
            col.label(text=f"{name}: {status}"[:160], icon="ERROR" if status.startswith("failed") else "PLUGIN")
        if state.get().connection.get("mcp"):
            col.label(text=state.get().connection["mcp"], icon="INFO")
        col.label(text="New MCP tools are used from the next new conversation.", icon="INFO")

    def _draw_more(self, col: Any) -> None:
        col.prop(self, "confirm_python")
        row = col.row()
        row.prop(self, "auto_screenshot")
        sub = row.row()
        sub.enabled = self.auto_screenshot
        sub.prop(self, "screenshot_size")
        col.prop(self, "max_steps")
        if self.is_anthropic:
            col.prop(self, "use_fallbacks")
        col.prop(self, "cache_dir")
        row = col.row()
        row.prop(self, "pack_textures")
        row.prop(self, "normal_convention")
        col.prop(self, "show_bar")
        _draw_keymap(col, bpy.context)


def draw_model_rows(layout: Any, prefs: BlenderAIPreferences) -> None:
    st = state.get()
    row = layout.row(align=True)
    row.label(text="Model")
    label = models.short_label(prefs.model(), st.chat_models.models) or "Choose…"
    row.operator("blender_ai.choose_model", text=label, icon="DOWNARROW_HLT")
    row.operator("blender_ai.refresh_models", text="", icon="FILE_REFRESH")
    row = layout.row(align=True)
    row.label(text="Reasoning")
    row.prop(prefs, "reasoning", text="")
    row = layout.row(align=True)
    row.label(text="Image model")
    if prefs.image_endpoint() is None:
        row.label(text="Set a separate image API below", icon="INFO")
    else:
        label = models.short_label(prefs.image_model, st.image_models.models) or "Choose…"
        row.operator("blender_ai.choose_image_model", text=label, icon="DOWNARROW_HLT")
    for listed in (st.chat_models, st.image_models):
        if listed.loading:
            layout.label(text="Loading models…", icon="SORTTIME")
            break
        if listed.error:
            layout.label(text=listed.error[:120], icon="ERROR")


def _section(layout: Any, prefs: Any, prop: str, title: str, icon: str, draw: Any) -> None:
    box = layout.box()
    row = box.row()
    open_ = getattr(prefs, prop)
    row.prop(prefs, prop, text=title, icon="DISCLOSURE_TRI_DOWN" if open_ else "DISCLOSURE_TRI_RIGHT", emboss=False)
    row.label(icon=icon)
    if open_:
        draw(box.column())


def _draw_keymap(layout: Any, context: Any) -> None:
    import rna_keymap_ui

    kc = context.window_manager.keyconfigs.user
    km = kc.keymaps.get("3D View") if kc else None
    if km is None:
        return
    for kmi in km.keymap_items:
        if kmi.idname == "blender_ai.focus_bar":
            layout.label(text="Prompt bar shortcut:")
            rna_keymap_ui.draw_kmi([], kc, km, kmi, layout, 0)
            break


CLASSES = (BlenderAIMCPServer, BlenderAIPreferences)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
