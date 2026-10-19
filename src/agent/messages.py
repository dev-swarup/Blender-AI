from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from ..tools.registry import ImagePart


class LogKind(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    REASONING = "reasoning"
    TOOL = "tool"
    ERROR = "error"
    INFO = "info"
    APPROVAL = "approval"
    QUESTION = "question"
    AGENT = "agent"


class Status(enum.Enum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    WAITING = "waiting"


@dataclass
class LogEntry:
    id: int
    kind: LogKind
    text: str
    detail: str = ""
    status: Status = Status.OK
    tool: str | None = None
    destructive: bool = False
    expanded: bool = False
    agent: str = ""
    result: str = ""
    started: float = 0.0
    updated: float = 0.0


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    log_id: int | None = None


class Phase(enum.Enum):
    IDLE = "idle"
    STREAMING = "streaming"
    TOOLS = "tools"
    ASYNC = "async"
    APPROVAL = "approval"
    QUESTION = "question"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    turns: int = 0

    def add(self, usage: dict[str, Any]) -> None:
        self.input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        self.cache_read_tokens += int(usage.get("cache_read_input_tokens") or usage.get("cached_tokens") or 0)
        self.cache_write_tokens += int(usage.get("cache_creation_input_tokens") or 0)
        self.turns += int(usage.get("requests") or 1)


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    key: str = ""
    text: str = ""
    name: str = ""
    call_id: str = ""
    usage: dict[str, Any] | None = None


@dataclass
class ToolResult:
    text: str
    images: list[ImagePart] = field(default_factory=list)
    is_error: bool = False


@dataclass(frozen=True)
class UserTurn:
    text: str
    attachment: ImagePart | None
    snapshot: str


class AgentError(Exception):
    def __init__(self, message: str, kind: str = "api") -> None:
        super().__init__(message)
        self.kind = kind
