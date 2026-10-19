from __future__ import annotations

import contextlib
import importlib
import sys
import threading
import time
import traceback
from typing import Any

import bpy

from . import deps, preferences, state
from .ui import overlay

WARM_IMPORTS = ("numpy", "asyncio", "concurrent.futures", "subprocess", "tempfile", "inspect", "urllib.request", "http.client", "email.parser", "ipaddress")
WARM_TIMEOUT = 15.0

_warm_thread: threading.Thread | None = None
_runtime: Any = None
_step = 0


def _warm() -> None:
    for name in WARM_IMPORTS:
        with contextlib.suppress(Exception):
            importlib.import_module(name)


def _redraw_viewports() -> None:
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _load_step() -> float | None:
    global _runtime, _step
    st = state.get()
    if not st.registered:
        return None
    if _warm_thread is not None and _warm_thread.is_alive() and time.monotonic() - st.load_started < WARM_TIMEOUT:
        _redraw_viewports()
        return 0.05
    try:
        if _runtime is None:
            from . import runtime

            _runtime = runtime
            return 0.01
        if _step < len(_runtime.STEPS):
            _runtime.STEPS[_step]()
            _step += 1
            return 0.01
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        st.load_error = f"{type(exc).__name__}: {exc}"
        _redraw_viewports()
        return None
    st.ready = True
    print(f"Blender AI: ready in {time.monotonic() - st.load_started:.2f} s", file=sys.stderr)
    _redraw_viewports()
    return None


def register() -> None:
    global _warm_thread, _runtime, _step
    state.reset()
    st = state.get()
    st.load_started = time.monotonic()
    _runtime, _step = None, 0
    preferences.register()
    overlay.register()
    st.registered = True
    deps.prepare()
    if bpy.app.background:
        while _load_step() is not None:
            pass
        return
    _warm_thread = threading.Thread(target=_warm, name="blender_ai-warmup", daemon=True)
    _warm_thread.start()
    bpy.app.timers.register(_load_step, first_interval=0.05, persistent=True)


def unregister() -> None:
    global _runtime, _step
    st = state.get()
    st.registered = False
    if bpy.app.timers.is_registered(_load_step):
        bpy.app.timers.unregister(_load_step)
    if _runtime is not None:
        _runtime.unregister()
    for fn in st.timers:
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)
    for handler_list, fn in st.handlers:
        if fn in handler_list:
            handler_list.remove(fn)
    overlay.unregister()
    preferences.unregister()
    _redraw_viewports()
    _runtime, _step = None, 0
    state.reset()
