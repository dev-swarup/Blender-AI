from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import addon_utils
import bpy

from . import registry

REMOTE_HOST = "extensions.blender.org"
WATCH_SECONDS = 900

Done = Callable[[list[str], list[str]], None]


class InstallError(RuntimeError): ...


def remote_repo() -> Any:
    return next((r for r in bpy.context.preferences.extensions.repos if r.enabled and r.use_remote_url and REMOTE_HOST in r.remote_url), None)


def enable_installed(ids: set[str]) -> list[str]:
    enabled = []
    for helper in registry.missing():
        module = registry.installed_module(helper.id) if helper.id in ids else None
        if module is not None and addon_utils.enable(module, default_set=True, persistent=True) is not None:
            enabled.append(helper.name)
    return enabled


def downloads(ids: set[str] | None = None) -> list[registry.Helper]:
    return [h for h in registry.missing() if h.pkg_id and registry.installed_module(h.id) is None and (ids is None or h.id in ids)]


def _install_package(directory: str, pkg_id: str, interactive: bool) -> None:
    window = (bpy.context.window or next(iter(bpy.context.window_manager.windows), None)) if interactive else None
    try:
        with bpy.context.temp_override(**({"window": window} if window else {})):
            bpy.ops.extensions.package_install("INVOKE_DEFAULT" if window else "EXEC_DEFAULT", repo_directory=directory, pkg_id=pkg_id, enable_on_install=True)
    except RuntimeError as exc:
        raise InstallError(str(exc).removeprefix("Error: ").strip()) from exc


def _install_running() -> bool:
    return any(op.bl_idname.startswith("EXTENSIONS_OT") for window in bpy.context.window_manager.windows for op in window.modal_operators)


def install(ids: set[str], interactive: bool, on_done: Done | None = None) -> tuple[list[str], list[str]]:
    addon_utils.modules(refresh=True)
    enabled = enable_installed(ids)
    pending = downloads(ids)
    if not pending:
        return enabled, []
    if not bpy.app.online_access:
        raise InstallError("Online access is off: turn it on in Preferences > System > Network, then install again.")
    repo = remote_repo()
    if repo is None:
        raise InstallError(f"The {REMOTE_HOST} repository is off or missing: turn it on in Preferences > Get Extensions > Repositories.")
    try:
        bpy.ops.extensions.repo_sync(repo_directory=repo.directory)
    except RuntimeError as exc:
        raise InstallError(f"Couldn't refresh the {REMOTE_HOST} listing: {exc}") from exc
    if interactive:
        _queue(repo.directory, pending, on_done or (lambda _missing, _errors: None))
    else:
        for helper in pending:
            _install_package(repo.directory, helper.pkg_id, False)
    return enabled, [h.name for h in pending]


def _queue(directory: str, helpers: list[registry.Helper], on_done: Done) -> None:
    deadline = time.monotonic() + WATCH_SECONDS
    queue = list(helpers)
    errors: list[str] = []

    def tick() -> float | None:
        if time.monotonic() < deadline and _install_running():
            return 0.5
        if time.monotonic() < deadline and queue:
            helper = queue.pop(0)
            try:
                _install_package(directory, helper.pkg_id, True)
            except InstallError as exc:
                errors.append(f"{helper.name}: {exc}")
            return 0.5
        missing = {h.id for h in registry.missing()}
        on_done([h.name for h in helpers if h.id in missing], errors)
        return None

    bpy.app.timers.register(tick, first_interval=0.1)
