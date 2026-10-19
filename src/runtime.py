from __future__ import annotations

import contextlib
from typing import Any

import bpy
from bpy.app.handlers import persistent

from . import packing, preferences, state
from .agent.host import BlenderHost, redraw
from .agent.loop import AgentSession
from .agent.mcp_bridge import MCPManager
from .tools import catalog
from .ui import dialogs, icons, input_modal, operators, panel

BUSY_INTERVAL = 0.05
IDLE_INTERVAL = 0.25
UI_MODULES = (operators, input_modal, panel, dialogs)


def _tick() -> float | None:
    st = state.get()
    if not st.registered:
        return None
    ran = st.bridge.drain()
    if not st.icon_pixels:
        icons.load_pixels()
    session = st.session
    busy = session is not None and session.busy
    if busy or ran:
        redraw()
    return BUSY_INTERVAL if busy or ran or st.ui.focused else IDLE_INTERVAL


def _startup() -> None:
    if not state.get().registered:
        return
    operators.refresh_model_lists()
    operators.connect_mcp()


@persistent
def _on_load(_dummy: Any = None) -> None:
    st = state.get()
    st.modal_running = False
    st.ui.focused = False
    st.ui.dropdown = None
    st.ui.layouts.clear()
    st.textures.clear()
    if st.session is not None:
        if st.session.busy:
            st.session.stop()
        if st.session.has_history:
            st.session.new_thread()


@persistent
def _on_save(_dummy: Any = None) -> None:
    prefs = preferences.get_prefs()
    if prefs is not None and prefs.pack_textures:
        packing.pack_generated_files(prefs.texture_dir())


def load_tools() -> None:
    catalog.load()


def register_ui() -> None:
    for module in UI_MODULES:
        module.register()
    icons.load()


def start() -> None:
    st = state.get()
    prefs = preferences.get_prefs()
    mode = prefs.current_mode() if prefs else None
    st.session = AgentSession(BlenderHost(), st.bridge, **({"mode": mode} if mode else {}))
    st.mcp = MCPManager(on_change=lambda: st.bridge.post(redraw))
    bpy.app.timers.register(_tick, first_interval=0.05, persistent=True)
    st.timers.append(_tick)
    bpy.app.timers.register(_startup, first_interval=1.0)
    dialogs.open_helpers_prompt_later()
    for handler_list, fn in ((bpy.app.handlers.load_post, _on_load), (bpy.app.handlers.save_pre, _on_save)):
        handler_list.append(fn)
        st.handlers.append((handler_list, fn))


STEPS = (load_tools, register_ui, start)


def unregister() -> None:
    st = state.get()
    if st.session is not None and st.session.busy:
        st.session.stop()
    if st.mcp is not None:
        st.mcp.shutdown()
    for fn in (_startup, dialogs.open_helpers_prompt):
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)
    for module in reversed(UI_MODULES):
        with contextlib.suppress(RuntimeError):
            module.unregister()
    icons.unload()
