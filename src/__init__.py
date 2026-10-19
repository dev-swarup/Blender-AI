# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations


def register() -> None:
    from . import addon

    addon.register()


def unregister() -> None:
    from . import addon

    addon.unregister()
