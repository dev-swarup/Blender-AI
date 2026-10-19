from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import bpy

from .. import models, packing, state
from ..integrations import registry as integrations
from ..tools import registry
from ..tools.registry import AsyncJob, ImagePart, Mode, ToolOutput, ToolSpec
from . import context
from .config import AgentConfig


def _prefs() -> Any:
    from ..preferences import get_prefs

    return get_prefs()


def undo_push(label: str) -> None:
    window = next(iter(bpy.context.window_manager.windows), None)
    if window is None:
        return
    with bpy.context.temp_override(window=window):
        if bpy.ops.ed.undo_push.poll():
            bpy.ops.ed.undo_push(message=f"Blender AI: {label}")


def redraw() -> None:
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "PREFERENCES"}:
                area.tag_redraw()


class BlenderHost:
    def __init__(self) -> None:
        self._before: set[str] = set()

    def agent_config(self) -> AgentConfig:
        return _prefs().agent_config()

    def preflight(self, cfg: AgentConfig) -> str | None:
        if _prefs() is None:
            return "Blender AI preferences are missing; re-enable the extension."
        if not bpy.app.online_access:
            return "Online access is off. Enable it in Preferences > System > Network > Allow Online Access."
        local = any(h in cfg.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not cfg.api_key and not local:
            return "Add your API key in the Blender AI settings (gear icon > All Settings)."
        return None if cfg.model else "Pick a model on the prompt bar first (click the model button)."

    def available(self, requirement: str) -> bool:
        return integrations.available(requirement)

    def extra_tools(self) -> Sequence[ToolSpec]:
        manager = state.get().mcp
        if manager is None:
            return []
        taken = {s.name for s in registry.REGISTRY.specs()}
        return manager.tool_specs(taken)

    def confirm_python(self) -> bool:
        return bool(_prefs().confirm_python)

    def scene_snapshot(self, track: bool = True) -> str:
        if track:
            self._before = {ob.name for ob in bpy.data.objects}
        return context.scene_snapshot()

    def feedback_image(self) -> ImagePart | None:
        prefs = _prefs()
        return context.capture_viewport(prefs.screenshot_size) if prefs.auto_screenshot else None

    def execute(self, spec: ToolSpec, args: dict[str, Any]) -> ToolOutput | AsyncJob:
        before = packing.image_ids() if spec.mutates else None
        out = registry.call(spec, args)
        if isinstance(out, AsyncJob):
            return self._with_undo(spec, out)
        if spec.mutates:
            packing.pack_new(before)
            undo_push(spec.name)
        return out

    def _with_undo(self, spec: ToolSpec, job: AsyncJob) -> AsyncJob:
        def finish(payload: Any) -> Any:
            before = packing.image_ids() if spec.mutates else None
            result = job.finish(payload)
            if isinstance(result, AsyncJob):
                return self._with_undo(spec, result)
            if spec.mutates:
                packing.pack_new(before)
                undo_push(spec.name)
            return result

        return AsyncJob(job.label, job.work, finish)

    def approval_requested(self) -> None:
        from ..ui import dialogs

        dialogs.open_approval_later()

    def changed(self) -> None:
        redraw()

    def finished(self, mode: Mode) -> None:
        if mode is Mode.MODEL and self._before:
            new = [ob for ob in bpy.data.objects if ob.name not in self._before and ob.type in {"MESH", "CURVE", "ARMATURE", "FONT"}]
            if new:
                context.frame_objects(new)
        redraw()


def model_label() -> str:
    prefs = _prefs()
    return models.short_label(prefs.model(), state.get().chat_models.models) if prefs else ""
