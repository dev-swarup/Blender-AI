from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent.bridge import Bridge
from .models import ModelInfo
from .ui.textedit import TextBuffer
from .ui.transcript import Scroll


@dataclass
class UIState:
    focused: bool = False
    buffer: TextBuffer = field(default_factory=TextBuffer)
    attachment_path: str | None = None
    layouts: dict[int, Any] = field(default_factory=dict)
    caret_visible: bool = True
    expanded_log: bool = False
    log_scroll: Scroll = field(default_factory=Scroll)
    hover: str | None = None
    pressed: str | None = None
    dropdown: Any = None
    dropdown_region: int = 0
    dropdown_layout: Any = None


@dataclass
class ModelList:
    endpoint: tuple[Any, ...] = ()
    models: list[ModelInfo] = field(default_factory=list)
    loading: bool = False
    error: str = ""


@dataclass
class State:
    bridge: Bridge = field(default_factory=Bridge)
    session: Any = None
    mcp: Any = None
    ui: UIState = field(default_factory=UIState)
    draw_handles: list[tuple[Any, Any]] = field(default_factory=list)
    keymaps: list[tuple[Any, Any]] = field(default_factory=list)
    timers: list[Any] = field(default_factory=list)
    handlers: list[tuple[list[Any], Any]] = field(default_factory=list)
    icons: Any = None
    textures: dict[str, Any] = field(default_factory=dict)
    icon_pixels: dict[str, tuple[int, int, Any]] = field(default_factory=dict)
    chat_models: ModelList = field(default_factory=ModelList)
    image_models: ModelList = field(default_factory=ModelList)
    enum_cache: dict[str, list[tuple[Any, ...]]] = field(default_factory=dict)
    connection: dict[str, str] = field(default_factory=lambda: {"llm": "", "image": "", "mcp": ""})
    modal_running: bool = False
    capturing: bool = False
    registered: bool = False
    ready: bool = False
    load_started: float = 0.0
    load_error: str = ""


S = State()


def reset() -> None:
    global S
    S = State()


def get() -> State:
    return S
