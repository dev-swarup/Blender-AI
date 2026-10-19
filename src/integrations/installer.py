from __future__ import annotations

import time
from collections.abc import Callable

import addon_utils
import bpy

from . import registry

REMOTE_HOST = "extensions.blender.org"
WATCH_SECONDS = 900


class InstallError(RuntimeError): ...


def remote_repo_index() -> int | None:
    repos = bpy.context.preferences.extensions.repos
    return next((i for i, repo in enumerate(repos) if repo.enabled and repo.use_remote_url and REMOTE_HOST in repo.remote_url), None)


def enable_installed() -> list[str]:
    enabled = []
    for helper in registry.missing():
        module = registry.installed_module(helper.id)
        if module is not None and addon_utils.enable(module, default_set=True, persistent=True) is not None:
            enabled.append(helper.name)
    return enabled


def downloads() -> list[registry.Helper]:
    return [h for h in registry.missing() if h.pkg_id and registry.installed_module(h.id) is None]


def install(interactive: bool) -> tuple[list[str], list[str]]:
    enabled = enable_installed()
    pending = downloads()
    if not pending:
        return enabled, []
    if not bpy.app.online_access:
        raise InstallError("Online access is off: turn it on in Preferences > System > Network, then install again.")
    index = remote_repo_index()
    if index is None:
        raise InstallError(f"The {REMOTE_HOST} repository is off or missing: turn it on in Preferences > Get Extensions > Repositories.")
    if "FINISHED" not in bpy.ops.extensions.repo_sync(repo_index=index):
        raise InstallError(f"Couldn't refresh the {REMOTE_HOST} listing. Check your connection and try again.")
    for helper in pending:
        bpy.ops.extensions.package_mark_set(pkg_id=helper.pkg_id, repo_index=index)
    result = bpy.ops.extensions.package_install_marked("INVOKE_DEFAULT" if interactive else "EXEC_DEFAULT", enable_on_install=True)
    if not result & {"FINISHED", "RUNNING_MODAL"}:
        for helper in pending:
            bpy.ops.extensions.package_mark_clear(pkg_id=helper.pkg_id, repo_index=index)
        raise InstallError("Blender couldn't start the install; see the Info editor for details.")
    return enabled, [h.name for h in pending]


def _install_running() -> bool:
    return any(op.bl_idname.startswith("EXTENSIONS_OT") for window in bpy.context.window_manager.windows for op in window.modal_operators)


def watch(on_done: Callable[[list[str]], None]) -> None:
    deadline = time.monotonic() + WATCH_SECONDS

    def tick() -> float | None:
        if _install_running() and time.monotonic() < deadline:
            return 1.0
        on_done([h.name for h in registry.missing()])
        return None

    bpy.app.timers.register(tick, first_interval=1.0)
