from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import deps
from ..tools.registry import AsyncJob, ImagePart, ToolError, ToolKind, ToolOutput, ToolSpec

CONNECT_TIMEOUT = 90.0
CALL_TIMEOUT = 900.0


@dataclass(frozen=True)
class ServerConfig:
    name: str
    command: str = ""
    url: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.name, self.command, self.url, tuple(sorted(self.env.items())), tuple(sorted(self.headers.items())))


@dataclass
class ServerState:
    config: ServerConfig
    status: str = "connecting…"
    error: str = ""
    tools: list[Any] = field(default_factory=list)
    session: Any = None
    stop: asyncio.Event | None = None
    task: asyncio.Task[Any] | None = None
    log_path: str = ""


def parse_pairs(text: str, sep: str = "=") -> dict[str, str]:
    out: dict[str, str] = {}
    for part in re.split(r"[;\n]", text or ""):
        if sep in part:
            key, value = part.split(sep, 1)
            if key.strip():
                out[key.strip()] = value.strip()
    return out


def format_pairs(pairs: Mapping[str, str], sep: str = "=") -> str:
    return "; ".join(f"{k}{sep}{v}" for k, v in pairs.items())


def parse_config_json(text: str) -> list[ServerConfig]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    block = data.get("mcpServers") or data.get("servers") or data.get("mcp", {}).get("servers")
    if block is None:
        block = {data.get("name", "server"): data} if ("command" in data or "url" in data) else data
    if not isinstance(block, dict):
        raise ValueError("No servers found in the JSON.")
    configs = []
    for name, spec in block.items():
        if not isinstance(spec, dict):
            continue
        args = [str(a) for a in spec.get("args") or []]
        command = " ".join(shlex.quote(p) if " " in p else p for p in [str(spec.get("command", "")), *args]).strip()
        env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
        headers = {str(k): str(v) for k, v in (spec.get("headers") or {}).items()}
        url = str(spec.get("url") or spec.get("serverUrl") or "")
        if command or url:
            configs.append(ServerConfig(str(name), command, url, env, headers))
    if not configs:
        raise ValueError("No servers with a command or url were found.")
    return configs


def tool_name(server: str, tool: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{server}_{tool}").strip("_")[:64] or "mcp_tool"
    name, n = base, 2
    while name in taken:
        suffix = f"_{n}"
        name, n = base[: 64 - len(suffix)] + suffix, n + 1
    taken.add(name)
    return name


def _schema(raw: Any) -> dict[str, Any]:
    schema = dict(raw) if isinstance(raw, Mapping) else {}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    if schema.get("type") != "object":
        schema = {"type": "object", "properties": {"value": schema}}
    return schema


def result_output(result: Any) -> ToolOutput:
    texts: list[str] = []
    images: list[ImagePart] = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(block.text)
        elif kind == "image":
            try:
                images.append(ImagePart(base64.b64decode(block.data), getattr(block, "mime_type", "image/png")))
            except (ValueError, TypeError):
                texts.append("[an image that couldn't be decoded]")
        elif kind == "resource_link":
            texts.append(f"Resource: {getattr(block, 'name', '')} {block.uri}")
        elif kind == "resource":
            res = block.resource
            texts.append(getattr(res, "text", None) or f"Resource: {res.uri}")
        else:
            texts.append(json.dumps(block.model_dump(mode="json"), ensure_ascii=False)[:2000])
    structured = getattr(result, "structured_content", None)
    if structured is not None and not texts:
        texts.append(json.dumps(structured, ensure_ascii=False, default=str))
    text = "\n".join(t for t in texts if t) or "ok"
    if getattr(result, "is_error", False):
        raise ToolError(text)
    return ToolOutput(text, images)


class MCPManager:
    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self.on_change = on_change or (lambda: None)
        self.servers: dict[str, ServerState] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=loop.run_forever, name="blender_ai-mcp", daemon=True)
                self._thread.start()
                self._loop = loop
            return self._loop

    def _submit(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())

    def sync(self, configs: list[ServerConfig]) -> None:
        wanted = {c.name: c for c in configs}
        for name, state in list(self.servers.items()):
            if name not in wanted or wanted[name].key != state.config.key:
                self._submit(self._stop(state))
                del self.servers[name]
        for name, config in wanted.items():
            if name not in self.servers:
                state = ServerState(config)
                self.servers[name] = state
                self._submit(self._start(state))
        self.on_change()

    def reconnect(self, configs: list[ServerConfig]) -> None:
        for state in list(self.servers.values()):
            self._submit(self._stop(state))
        self.servers.clear()
        self.sync(configs)

    def shutdown(self) -> None:
        if self._loop is None:
            return
        for state in list(self.servers.values()):
            with contextlib.suppress(Exception):
                self._submit(self._stop(state)).result(timeout=5)
        self.servers.clear()
        loop = self._loop
        loop.call_soon_threadsafe(loop.stop)
        self._loop = None

    def status(self) -> dict[str, str]:
        return {name: state.status for name, state in self.servers.items()}

    def tool_specs(self, taken: set[str]) -> list[ToolSpec]:
        specs = []
        for server, state in sorted(self.servers.items()):
            if state.session is None:
                continue
            for tool in state.tools:
                name = tool_name(server, tool.name, taken)
                description = (tool.description or tool.name).strip()
                specs.append(
                    ToolSpec(
                        name=name,
                        description=f"[MCP server '{server}'] {description}"[:4000],
                        input_schema=_schema(getattr(tool, "input_schema", None)),
                        func=self._caller(server, tool.name),
                        group="mcp",
                        kind=ToolKind.ASYNC,
                        mutates=False,
                    )
                )
        return specs

    def call(self, server: str, tool: str, args: dict[str, Any]) -> ToolOutput:
        state = self.servers.get(server)
        if state is None or state.session is None:
            raise ToolError(f"MCP server {server!r} isn't connected ({state.status if state else 'removed'}).")
        future = self._submit(state.session.call_tool(tool, args))
        try:
            result = future.result(timeout=CALL_TIMEOUT)
        except TimeoutError as exc:
            future.cancel()
            raise ToolError(f"{server}.{tool} didn't answer within {int(CALL_TIMEOUT)} s.") from exc
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{server}.{tool} failed: {exc}") from exc
        return result_output(result)

    def _caller(self, server: str, tool: str) -> Callable[..., AsyncJob]:
        def run(**args: Any) -> AsyncJob:
            return AsyncJob(f"{server}.{tool}", lambda: self.call(server, tool, args), lambda out: out)

        return run

    async def _start(self, state: ServerState) -> None:
        state.stop = asyncio.Event()
        state.task = asyncio.get_running_loop().create_task(self._serve(state))

    async def _stop(self, state: ServerState) -> None:
        if state.stop is not None:
            state.stop.set()
        if state.task is not None:
            try:
                await asyncio.wait_for(state.task, timeout=10)
            except (TimeoutError, Exception):
                state.task.cancel()

    async def _serve(self, state: ServerState) -> None:
        deps.prepare()
        try:
            from mcp import ClientSession

            async with self._transport(state) as streams, ClientSession(streams[0], streams[1]) as session:
                await asyncio.wait_for(session.initialize(), CONNECT_TIMEOUT)
                listed = await asyncio.wait_for(session.list_tools(), CONNECT_TIMEOUT)
                state.tools = list(listed.tools)
                state.session = session
                state.status = f"connected · {len(state.tools)} tools"
                self.on_change()
                assert state.stop is not None
                await state.stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            state.error = _describe(exc)
            hint = f" (log: {state.log_path})" if state.log_path else ""
            state.status = f"failed: {state.error}{hint}"
            print(f"Blender AI: MCP server {state.config.name!r} failed: {state.error}", file=sys.stderr)
        finally:
            state.session = None
            if not state.error:
                state.status = "stopped"
            self.on_change()

    def _transport(self, state: ServerState) -> Any:
        config = state.config
        if config.url:
            from mcp.shared._httpx_utils import create_mcp_http_client

            if config.url.rstrip("/").endswith("/sse"):
                from mcp.client.sse import sse_client

                return sse_client(config.url, headers=dict(config.headers) or None)
            from mcp.client.streamable_http import streamable_http_client

            return streamable_http_client(config.url, http_client=create_mcp_http_client(headers=dict(config.headers)))

        from mcp import StdioServerParameters

        parts = shlex.split(config.command, posix=os.name != "nt")
        parts = [p.strip('"') for p in parts]
        if not parts:
            raise ValueError("No command given.")
        safe = re.sub(r"\W+", "_", config.name)
        state.log_path = os.path.join(tempfile.gettempdir(), f"blender_ai_mcp_{safe}.log")
        params = StdioServerParameters(command=parts[0], args=parts[1:], env=dict(config.env) or None)
        return _stdio_with_log(params, state.log_path)


@contextlib.asynccontextmanager
async def _stdio_with_log(params: Any, log_path: str) -> AsyncIterator[Any]:
    from mcp.client.stdio import stdio_client

    with open(log_path, "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as streams:
            yield streams


def _describe(exc: BaseException) -> str:
    inner = getattr(exc, "exceptions", None)
    return _describe(inner[0]) if inner else f"{type(exc).__name__}: {exc}"[:300]
