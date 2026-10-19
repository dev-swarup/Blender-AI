from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import bpy

from .layout import Metrics

RGBA = tuple[float, float, float, float]


def rgba(color: Sequence[float], alpha: float | None = None) -> RGBA:
    c = tuple(color)
    a = c[3] if len(c) > 3 else 1.0
    return (c[0], c[1], c[2], a if alpha is None else alpha)


def mix(a: RGBA, b: RGBA, t: float) -> RGBA:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t)


@dataclass(frozen=True)
class Palette:
    bar: RGBA
    bar_outline: RGBA
    field: RGBA
    field_active: RGBA
    text: RGBA
    pill: RGBA
    pill_hover: RGBA
    pill_text: RGBA
    pill_active: RGBA
    pill_active_text: RGBA
    button: RGBA
    button_text: RGBA
    accent: RGBA
    accent_text: RGBA
    error: RGBA
    warning: RGBA
    shadow: RGBA
    caret: RGBA
    card_text: RGBA
    card_text_dim: RGBA
    tab_bg: RGBA
    tab_active: RGBA
    tab_text: RGBA
    tab_active_text: RGBA
    focus_outline: RGBA
    menu_bg: RGBA
    menu_outline: RGBA
    menu_text: RGBA
    menu_text_dim: RGBA
    menu_item_sel: RGBA
    menu_item_text_sel: RGBA
    field_text: RGBA


def _status(theme: Any, name: str, fallback: RGBA) -> RGBA:
    state = getattr(theme.user_interface, "wcol_state", None)
    for owner, attr in ((state, name), (theme.info, f"info_{name}"), (theme.info, f"info_{name}_text")):
        value = getattr(owner, attr, None) if owner is not None else None
        if value is not None:
            return rgba(value, 1.0)
    return fallback


def palette() -> Palette:
    theme = bpy.context.preferences.themes[0]
    ui = theme.user_interface
    menu = ui.wcol_menu_back
    text = ui.wcol_text
    toggle = ui.wcol_toggle
    tool = ui.wcol_tool
    regular = ui.wcol_regular
    tab = getattr(ui, "wcol_tab", toggle)
    item = ui.wcol_menu_item
    base_text = rgba(text.text)
    shadow_alpha = min(0.5, max(0.0, ui.menu_shadow_fac))
    return Palette(
        bar=rgba(menu.inner, max(0.92, menu.inner[3])),
        bar_outline=rgba(menu.outline),
        field=rgba(text.inner),
        field_active=rgba(text.inner_sel),
        text=base_text,
        pill=rgba(regular.inner),
        pill_hover=mix(rgba(regular.inner), rgba(regular.inner_sel), 0.35),
        pill_text=rgba(regular.text),
        pill_active=rgba(toggle.inner_sel),
        pill_active_text=rgba(toggle.text_sel),
        button=rgba(tool.inner),
        button_text=rgba(tool.text),
        accent=rgba(tool.inner_sel),
        accent_text=rgba(tool.text_sel),
        error=_status(theme, "error", (0.9, 0.25, 0.25, 1.0)),
        warning=_status(theme, "warning", (0.9, 0.6, 0.15, 1.0)),
        shadow=(0.0, 0.0, 0.0, shadow_alpha),
        caret=rgba(ui.widget_text_cursor) if hasattr(ui, "widget_text_cursor") else base_text,
        card_text=rgba(menu.text),
        card_text_dim=rgba(menu.text, 0.5),
        tab_bg=rgba(menu.inner, max(0.92, menu.inner[3])),
        tab_active=rgba(regular.inner),
        tab_text=rgba(tab.text, 0.75),
        tab_active_text=rgba(tab.text_sel),
        focus_outline=mix(rgba(menu.outline), rgba(menu.text), 0.35),
        menu_bg=rgba(menu.inner, max(0.98, menu.inner[3])),
        menu_outline=rgba(menu.outline),
        menu_text=rgba(menu.text),
        menu_text_dim=rgba(menu.text, 0.5),
        menu_item_sel=rgba(item.inner_sel),
        menu_item_text_sel=rgba(item.text_sel),
        field_text=rgba(text.text),
    )


def metrics() -> Metrics:
    prefs = bpy.context.preferences
    scale = prefs.system.ui_scale or prefs.view.ui_scale or 1.0
    points = prefs.ui_styles[0].widget.points if prefs.ui_styles else 11.0
    roundness = prefs.themes[0].user_interface.wcol_regular.roundness
    return Metrics.from_scale(scale, points, roundness)
