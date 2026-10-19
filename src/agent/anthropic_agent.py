from __future__ import annotations

import base64
import threading
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from .. import models
from ..providers import ANTHROPIC_URL
from ..tools.registry import ImagePart, ToolSpec
from . import history
from .config import AgentConfig
from .messages import AgentError, AgentEvent, ToolResult, UserTurn

DEFAULT_BASE_URL = ANTHROPIC_URL
OFFICIAL_HOSTS = frozenset({"api.anthropic.com"})
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_JSON_RETRIES = 2
CANCELLED = "Cancelled by the user before this tool ran."


def is_official(base_url: str) -> bool:
    host = urlparse(base_url or DEFAULT_BASE_URL).hostname or ""
    return host.lower() in OFFICIAL_HOSTS


def base_url(cfg_url: str) -> str:
    url = (cfg_url or DEFAULT_BASE_URL).strip().rstrip("/")
    return url.removesuffix("/v1")


def image_block(image: ImagePart) -> dict[str, Any]:
    data = base64.standard_b64encode(image.data).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": image.media_type, "data": data}}


def tool_param(spec: ToolSpec) -> dict[str, Any]:
    return {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema}


def build_request(cfg: AgentConfig, *, system: str, tools: Sequence[ToolSpec], messages: list[dict[str, Any]], level: str | None = None) -> dict[str, Any]:
    caps = models.anthropic_caps(cfg.model, cfg.model_info)
    official = is_official(cfg.base_url)
    params: dict[str, Any] = {"model": cfg.model, "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}], "messages": messages}
    params.update(models.anthropic_params(caps, level or cfg.reasoning))
    tool_params = [tool_param(s) for s in tools]
    if official:
        for t in tool_params:
            t["eager_input_streaming"] = True
        params["cache_control"] = {"type": "ephemeral"}
    if tool_params:
        params["tools"] = tool_params
    if cfg.use_fallbacks and caps.fallbacks and official:
        params["fallbacks"] = "default"
        params["betas"] = [FALLBACK_BETA]
    return params


def map_error(exc: BaseException, url: str) -> AgentError:
    import anthropic

    where = url or DEFAULT_BASE_URL
    if isinstance(exc, AgentError):
        return exc
    if isinstance(exc, anthropic.AuthenticationError):
        return AgentError("The API key was rejected. Check it in Blender AI settings.", "auth")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return AgentError("This API key isn't allowed to use that model or endpoint.", "permission")
    if isinstance(exc, anthropic.NotFoundError):
        return AgentError(f"Not found at {where}. Check the API URL and the model.", "not_found")
    if isinstance(exc, anthropic.RateLimitError):
        retry = exc.response.headers.get("retry-after") if exc.response is not None else None
        seconds = float(retry) if retry and retry.replace(".", "", 1).isdigit() else None
        return AgentError("Rate limited by the API." + (f" Try again in {int(seconds)} s." if seconds else ""), "rate")
    if isinstance(exc, anthropic.BadRequestError):
        return AgentError(f"The API rejected the request: {exc.message}", "bad_request")
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return AgentError(f"The API had a server error ({exc.status_code}). Try again shortly.", "server")
        return AgentError(f"API error {exc.status_code}: {exc.message}", "api")
    if isinstance(exc, anthropic.APITimeoutError):
        return AgentError(f"The request to {where} timed out.", "timeout")
    if isinstance(exc, anthropic.APIConnectionError):
        return AgentError(f"Can't reach {where}. Check the API URL and your connection.", "connection")
    return AgentError(f"{type(exc).__name__}: {exc}", "internal")


def sdk_client(cfg: AgentConfig) -> Any:
    import anthropic

    return anthropic.Anthropic(api_key=cfg.api_key or None, base_url=base_url(cfg.base_url), max_retries=2, timeout=anthropic.Timeout(600.0, connect=15.0))


def list_models(cfg: AgentConfig) -> list[models.ModelInfo]:
    try:
        entries = [m.to_dict() for m in sdk_client(cfg).models.list(limit=1000)]
    except Exception as exc:
        raise map_error(exc, cfg.base_url) from exc
    return models.parse_models(entries)


def test_chat(cfg: AgentConfig) -> None:
    caps = models.anthropic_caps(cfg.model, cfg.model_info)
    params = models.anthropic_params(caps, "auto", 64)
    try:
        sdk_client(cfg).messages.create(model=cfg.model, messages=[{"role": "user", "content": "Reply with the single word OK."}], **params)
    except Exception as exc:
        raise map_error(exc, cfg.base_url) from exc


class AnthropicAgent:
    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self.messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._client: Any = None
        self._prefix: tuple[Any, ...] | None = None

    @property
    def has_history(self) -> bool:
        return bool(self.messages)

    def reset(self) -> None:
        with self._lock:
            self.messages = []
            self._prefix = None

    def run(self, run: Any, turn: UserTurn) -> str:
        with self._lock:
            cfg: AgentConfig = run.config
            self.cfg = cfg
            if self._client is None:
                self._client = sdk_client(cfg)
            prefix = (cfg.model, run.system, tuple(s.name for s in run.tools))
            if self._prefix is not None and prefix != self._prefix:
                self._drop_thinking()
            self._prefix = prefix

            content: list[dict[str, Any]] = []
            if history.needs_compaction(self.messages, cfg.context_window) and (summary := self._compact(run, cfg)):
                content.append({"type": "text", "text": history.summary_turn(summary)})
            if turn.snapshot:
                content.append({"type": "text", "text": f"<scene>\n{turn.snapshot}\n</scene>"})
            if turn.attachment is not None and cfg.vision:
                content.append(image_block(turn.attachment))
            content.append({"type": "text", "text": turn.text})
            self.messages.append({"role": "user", "content": content})
            note = history.FAILED
            try:
                reply = self._loop(run, cfg)
                note = history.STOPPED
                return reply
            except AgentError as exc:
                note = history.PAUSED if exc.kind == "paused" else history.FAILED
                raise
            finally:
                self._close(history.STOPPED if run.cancelled else note)

    def _loop(self, run: Any, cfg: AgentConfig) -> str:
        vision = cfg.vision
        steps = 0
        turn_no = 0
        while not run.cancelled:
            if steps >= cfg.max_steps:
                raise AgentError(f"Paused after {steps} steps (the limit in settings). Send a message to continue.", "paused")
            steps += 1
            turn_no += 1
            level = cfg.effort()
            params = build_request(cfg, system=run.system, tools=run.tools, messages=list(self.messages), level=level)
            try:
                message = self._stream(params, run, turn_no)
            except AgentError as exc:
                if exc.kind != "bad_request" or "effort" not in str(exc).lower() or level == "auto":
                    raise
                run.emit(AgentEvent("info", text=cfg.reject_effort(level)))
                steps -= 1
                turn_no -= 1
                continue
            if message is None:
                return ""
            blocks = [block.to_dict() for block in message.content]
            self.messages.append({"role": "assistant", "content": blocks})
            usage = getattr(message, "usage", None)
            if usage is not None:
                run.emit(AgentEvent("usage", usage=usage.to_dict()))

            calls = [b for b in blocks if b.get("type") == "tool_use"]
            reason = message.stop_reason
            if reason == "refusal":
                self._answer(calls, "Not run: the turn ended with a refusal.")
                details = getattr(message, "stop_details", None)
                category = getattr(details, "category", None) if details is not None else None
                raise AgentError("The model declined this request" + (f" ({category})." if category else "."), "refusal")
            if reason == "max_tokens" and calls:
                self._answer(calls, "Not run: the reply hit max_tokens before this tool call was complete.")
                raise AgentError("The reply was cut off at the output limit. Ask for a smaller step.", "max_tokens")
            if reason == "pause_turn":
                continue
            if not calls:
                return _final_text(blocks)

            results: list[dict[str, Any]] = []
            for call in calls:
                result = ToolResult(CANCELLED, is_error=True) if run.cancelled else run.call_tool(call["id"], call["name"], call.get("input") or {})
                results.append(self._result_block(call["id"], result, vision))
            image = run.feedback() if vision and not run.cancelled else None
            if image is not None:
                results += [{"type": "text", "text": "Viewport after these changes:"}, image_block(image)]
            self.messages.append({"role": "user", "content": results})
        return ""

    def _compact(self, run: Any, cfg: AgentConfig) -> str:
        params = models.anthropic_params(models.anthropic_caps(cfg.model, cfg.model_info), "low", 8000)
        request = [{"role": "user", "content": history.summary_prompt(self.messages, cfg.context_window)}]
        try:
            message = self._client.messages.create(model=cfg.model, system=history.SUMMARY_SYSTEM, messages=request, **params)
        except Exception as exc:
            run.emit(AgentEvent("info", text=history.compact_failed(str(map_error(exc, cfg.base_url)))))
            return ""
        if getattr(message, "usage", None) is not None:
            run.emit(AgentEvent("usage", usage=message.usage.to_dict()))
        summary = _final_text([block.to_dict() for block in message.content])
        if summary:
            self.messages = []
            run.emit(AgentEvent("info", text=history.COMPACTED))
        return summary

    def _close(self, note: str) -> None:
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.append({"role": "assistant", "content": [{"type": "text", "text": note}]})

    def _stream(self, params: dict[str, Any], run: Any, turn_no: int) -> Any:
        json_failures = 0
        while True:
            try:
                with self._client.beta.messages.stream(**params) as stream:
                    for event in stream:
                        if run.cancelled:
                            return None
                        _forward(event, run, turn_no)
                    return stream.get_final_message()
            except ValueError:
                json_failures += 1
                if json_failures > MAX_JSON_RETRIES:
                    raise AgentError("The model produced unreadable tool input several times in a row.", "json") from None
            except Exception as exc:
                raise map_error(exc, self.cfg.base_url) from exc

    @staticmethod
    def _result_block(tool_use_id: str, result: ToolResult, vision: bool) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": result.text or "ok"}]
        if vision:
            content.extend(image_block(img) for img in result.images)
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
        if result.is_error:
            block["is_error"] = True
        return block

    def _answer(self, calls: list[dict[str, Any]], text: str) -> None:
        if calls:
            self.messages.append(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": c["id"], "content": text, "is_error": True} for c in calls]}
            )

    def _drop_thinking(self) -> None:
        for message in self.messages:
            if message["role"] == "assistant" and isinstance(message["content"], list):
                message["content"] = [b for b in message["content"] if b.get("type") not in ("thinking", "redacted_thinking")] or [
                    {"type": "text", "text": "(no text)"}
                ]


def _final_text(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


def _forward(event: Any, run: Any, turn_no: int) -> None:
    etype = getattr(event, "type", "")
    if etype == "content_block_start":
        block = event.content_block
        if block.type == "tool_use":
            run.emit(AgentEvent("tool", name=block.name, call_id=block.id))
    elif etype == "content_block_delta":
        delta = event.delta
        key = f"{turn_no}:{event.index}"
        if delta.type == "text_delta":
            run.emit(AgentEvent("text", key=key, text=delta.text))
        elif delta.type == "thinking_delta":
            run.emit(AgentEvent("reasoning", key=key, text=delta.thinking))
