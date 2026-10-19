from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .. import deps, models
from ..providers import OPENAI_URL
from ..tools.registry import ImagePart, ToolKind, ToolSpec
from . import history
from .config import AgentConfig
from .messages import AgentError, AgentEvent, UserTurn

DEFAULT_BASE_URL = OPENAI_URL
CANCELLED = "Cancelled by the user before this tool ran."
_VERSION = re.compile(r"/v\d+[a-z0-9]*(/|$)", re.IGNORECASE)
_ROOTS: dict[str, str] = {}
_tracing_off = False


def api_roots(base_url: str) -> list[str]:
    url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/models", "/images/generations"):
        if url.lower().endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    if "://" not in url:
        local = url.startswith(("localhost", "127.", "0.0.0.0", "192.168.", "10."))
        url = ("http://" if local else "https://") + url
    return [url] if _VERSION.search(urlparse(url).path + "/") else [f"{url}/v1", url]


def api_root(base_url: str) -> str:
    return _ROOTS.get(base_url) or api_roots(base_url)[0]


def data_url(image: ImagePart) -> str:
    return f"data:{image.media_type};base64,{base64.standard_b64encode(image.data).decode('ascii')}"


def map_error(exc: BaseException, url: str) -> AgentError:
    import openai

    where = urlparse(url).netloc or url
    if isinstance(exc, AgentError):
        return exc
    message = getattr(exc, "message", None) or str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            message = str(err["message"])
            meta = err.get("metadata")
            raw = meta.get("raw") if isinstance(meta, dict) else None
            if isinstance(raw, str) and raw.strip() and raw.strip() not in message:
                message = f"{message} ({raw.strip()[:300]})"
    if isinstance(exc, openai.AuthenticationError):
        return AgentError(f"The API key was rejected by {where}. Check it in Blender AI settings.", "auth")
    if isinstance(exc, openai.PermissionDeniedError):
        return AgentError(f"This key isn't allowed to do that at {where}: {message}", "permission")
    if isinstance(exc, openai.NotFoundError):
        return AgentError(f"Not found at {where}: {message}. Check the API URL and the model.", "not_found")
    if isinstance(exc, openai.RateLimitError):
        return AgentError(f"Rate limited by {where}: {message}", "rate")
    if isinstance(exc, openai.BadRequestError):
        return AgentError(f"The API rejected the request: {message}", "bad_request")
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return AgentError(f"{where} had a server error ({exc.status_code}). Try again shortly.", "server")
        return AgentError(f"API error {exc.status_code} from {where}: {message}", "api")
    if isinstance(exc, openai.APITimeoutError):
        return AgentError(f"The request to {where} timed out.", "timeout")
    if isinstance(exc, openai.APIConnectionError):
        return AgentError(f"Can't reach {where}. Check the API URL and your connection.", "connection")
    return AgentError(f"{type(exc).__name__}: {message}", "internal")


def sdk_client(cfg_url: str, api_key: str, root: str | None = None, timeout: float = 600.0) -> Any:
    import openai

    return openai.OpenAI(base_url=root or api_root(cfg_url), api_key=api_key or "not-needed", max_retries=2, timeout=timeout)


def list_models(base_url: str, api_key: str) -> list[models.ModelInfo]:
    import openai

    last: BaseException | None = None
    for root in api_roots(base_url):
        params = {"output_modalities": "all"} if "openrouter.ai" in root else {}
        try:
            payload = sdk_client(base_url, api_key, root, timeout=30.0).get("/models", cast_to=object, options={"params": params})
        except openai.NotFoundError as exc:
            last = exc
            continue
        except Exception as exc:
            raise map_error(exc, root) from exc
        _ROOTS[base_url] = root
        return models.parse_models(payload)
    raise map_error(last or AgentError(f"No model list at {base_url}"), base_url)


def test_chat(cfg: AgentConfig) -> None:
    try:
        sdk_client(cfg.base_url, cfg.api_key, timeout=60.0).chat.completions.create(
            model=cfg.model, messages=[{"role": "user", "content": "Reply with the single word OK."}], max_tokens=16
        )
    except Exception as exc:
        raise map_error(exc, api_root(cfg.base_url)) from exc


def _close_open_calls(items: list[Any]) -> list[Any]:
    answered = {i.get("call_id") for i in items if isinstance(i, dict) and i.get("type") == "function_call_output"}
    open_calls = [i["call_id"] for i in items if isinstance(i, dict) and i.get("type") == "function_call" and i.get("call_id") not in answered]
    return [*items, *({"type": "function_call_output", "call_id": call_id, "output": CANCELLED} for call_id in open_calls)]


@dataclass
class _RunState:
    images_ok: bool
    turn: int = 0
    tools_called: int = 0
    pending_images: list[ImagePart] = field(default_factory=list)
    order: asyncio.Lock = field(default_factory=asyncio.Lock)


class OpenAIAgent:
    def __init__(self, cfg: AgentConfig) -> None:
        deps.prepare()
        global _tracing_off
        if not _tracing_off:
            from agents import set_tracing_disabled

            set_tracing_disabled(True)
            _tracing_off = True
        self.cfg = cfg
        self.history: list[Any] = []
        self.no_images: set[str] = set()
        self.no_usage: set[str] = set()
        self._lock = threading.Lock()

    @property
    def has_history(self) -> bool:
        return bool(self.history)

    def reset(self) -> None:
        with self._lock:
            self.history = []

    def run(self, run: Any, turn: UserTurn) -> str:
        with self._lock:
            self.cfg = run.config
            note = history.FAILED
            try:
                reply = asyncio.run(self._run(run, turn))
                note = history.STOPPED
                return reply
            except AgentError as exc:
                note = history.PAUSED if exc.kind == "paused" else history.FAILED
                raise
            finally:
                self._close(history.STOPPED if run.cancelled else note)

    async def _run(self, run: Any, turn: UserTurn) -> str:
        cfg: AgentConfig = run.config
        lead = ""
        if history.needs_compaction(self.history, cfg.context_window) and (summary := await self._compact(run, cfg)):
            lead = history.summary_turn(summary)
        for _attempt in range(4):
            state = _RunState(images_ok=cfg.vision and cfg.model not in self.no_images)
            before = list(self.history)
            try:
                return await self._attempt(run, turn, state, lead)
            except AgentError:
                raise
            except Exception as exc:
                error = map_error(exc, api_root(cfg.base_url))
                text = str(error).lower()
                retry = state.tools_called == 0 and error.kind == "bad_request"
                if "image" in text and cfg.model not in self.no_images and state.images_ok:
                    self.no_images.add(cfg.model)
                    if not retry:
                        raise AgentError(
                            "This model doesn't accept images, so screenshots are now off for it. Send your message again to continue.", "vision"
                        ) from exc
                    run.emit(AgentEvent("info", text="This model doesn't accept images; continuing without them."))
                elif retry and ("reasoning" in text or "effort" in text) and (effort := cfg.effort()) != "auto":
                    run.emit(AgentEvent("info", text=cfg.reject_effort(effort)))
                elif retry and "stream_options" in text and cfg.model not in self.no_usage:
                    self.no_usage.add(cfg.model)
                else:
                    raise error from exc
                self.history = before
        raise AgentError("The request kept failing; check the model and API URL.", "api")

    async def _compact(self, run: Any, cfg: AgentConfig) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=api_root(cfg.base_url), api_key=cfg.api_key or "not-needed", max_retries=2, timeout=600.0)
        request = [{"role": "system", "content": history.SUMMARY_SYSTEM}, {"role": "user", "content": history.summary_prompt(self.history, cfg.context_window)}]
        try:
            reply = await client.chat.completions.create(model=cfg.model, messages=request)
        except Exception as exc:
            run.emit(AgentEvent("info", text=history.compact_failed(str(map_error(exc, api_root(cfg.base_url))))))
            return ""
        if reply.usage is not None:
            run.emit(AgentEvent("usage", usage={"input_tokens": reply.usage.prompt_tokens, "output_tokens": reply.usage.completion_tokens}))
        summary = (reply.choices[0].message.content or "").strip() if reply.choices else ""
        if summary:
            self.history = []
            run.emit(AgentEvent("info", text=history.COMPACTED))
        return summary

    def _close(self, note: str) -> None:
        last = self.history[-1] if self.history else None
        if isinstance(last, dict) and not (last.get("role") == "assistant" and last.get("type", "message") == "message"):
            self.history.append({"role": "assistant", "content": note})

    async def _attempt(self, run: Any, turn: UserTurn, state: _RunState, lead: str = "") -> str:
        from agents import Agent, FunctionTool, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner
        from agents.exceptions import MaxTurnsExceeded
        from agents.run_config import ModelInputData
        from openai import AsyncOpenAI
        from openai.types.shared import Reasoning

        cfg: AgentConfig = run.config
        root = api_root(cfg.base_url)
        client = AsyncOpenAI(base_url=root, api_key=cfg.api_key or "not-needed", max_retries=2, timeout=600.0)
        settings = ModelSettings(include_usage=cfg.model not in self.no_usage or None)
        effort = cfg.effort()
        if effort != "auto":
            settings = ModelSettings(include_usage=settings.include_usage, reasoning=Reasoning(effort=effort))

        tools = [self._tool(spec, run, state, FunctionTool) for spec in run.tools]
        agent = Agent(
            name="Blender AI",
            instructions=run.system,
            tools=tools,
            model=OpenAIChatCompletionsModel(model=cfg.model, openai_client=client),
            model_settings=settings,
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": lead}] if lead else []
        if turn.snapshot:
            content.append({"type": "input_text", "text": f"<scene>\n{turn.snapshot}\n</scene>"})
        if turn.attachment is not None and state.images_ok:
            content.append({"type": "input_image", "image_url": data_url(turn.attachment), "detail": "auto"})
        content.append({"type": "input_text", "text": turn.text})
        items = [*self.history, {"role": "user", "content": content}]
        loop = asyncio.get_running_loop()

        async def add_images(data: Any) -> Any:
            model_input = data.model_data
            images = list(state.pending_images)
            state.pending_images.clear()
            if state.images_ok and state.tools_called:
                shot = await loop.run_in_executor(None, run.feedback)
                if shot is not None:
                    images.append(shot)
            if not images or not state.images_ok:
                return model_input
            parts: list[dict[str, Any]] = [{"type": "input_text", "text": "Images from your last tool calls; the last one is the viewport now:"}]
            parts += [{"type": "input_image", "image_url": data_url(img), "detail": "auto"} for img in images]
            return ModelInputData(input=[*model_input.input, {"role": "user", "content": parts}], instructions=model_input.instructions)

        result = Runner.run_streamed(
            agent, input=items, max_turns=cfg.max_steps, run_config=RunConfig(tracing_disabled=True, call_model_input_filter=add_images)
        )
        paused = False
        try:
            async for event in result.stream_events():
                if run.cancelled:
                    result.cancel()
                    break
                self._forward(event, run, state)
        except MaxTurnsExceeded:
            paused = True
        finally:
            try:
                self.history = _close_open_calls(result.to_input_list())
            except Exception:
                self.history = items
            usage = getattr(result.context_wrapper, "usage", None)
            if usage is not None and (usage.input_tokens or usage.output_tokens):
                details = getattr(usage, "input_tokens_details", None)
                run.emit(
                    AgentEvent(
                        "usage",
                        usage={
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
                            "requests": usage.requests or 1,
                        },
                    )
                )
        if paused:
            raise AgentError(f"Paused after {cfg.max_steps} steps (the limit in settings). Send a message to continue.", "paused")
        output = getattr(result, "final_output", None)
        return output.strip() if isinstance(output, str) else ""

    def _tool(self, spec: ToolSpec, run: Any, state: _RunState, function_tool: Any) -> Any:
        async def invoke(ctx: Any, raw: str) -> str:
            state.tools_called += 1
            try:
                args = json.loads(raw) if raw and raw.strip() else {}
            except json.JSONDecodeError as exc:
                return f"Error: the arguments were not valid JSON ({exc}). Send a JSON object."
            call_id = getattr(ctx, "tool_call_id", "") or f"call_{state.tools_called}"
            loop = asyncio.get_running_loop()
            if spec.kind is ToolKind.AGENT:
                result = await loop.run_in_executor(None, run.call_tool, call_id, spec.name, args)
            else:
                async with state.order:
                    result = await loop.run_in_executor(None, run.call_tool, call_id, spec.name, args)
            text = result.text or "ok"
            if result.images:
                if state.images_ok:
                    state.pending_images.extend(result.images)
                    text += f"\n[{len(result.images)} image(s) follow in the next message]"
                else:
                    text += "\n[image omitted: images are off for this model]"
            return f"Error: {text}" if result.is_error else text

        return function_tool(
            name=spec.name, description=spec.description, params_json_schema=spec.input_schema, on_invoke_tool=invoke, strict_json_schema=False
        )

    @staticmethod
    def _forward(event: Any, run: Any, state: _RunState) -> None:
        if getattr(event, "type", "") != "raw_response_event":
            return
        data = event.data
        kind = getattr(data, "type", "")
        index = getattr(data, "output_index", 0)
        if kind == "response.created":
            state.turn += 1
        elif kind == "response.output_text.delta":
            run.emit(AgentEvent("text", key=f"{state.turn}:t{index}", text=data.delta))
        elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            run.emit(AgentEvent("reasoning", key=f"{state.turn}:r{index}", text=data.delta))
        elif kind == "response.output_item.added":
            item = data.item
            if getattr(item, "type", "") == "function_call":
                run.emit(AgentEvent("tool", name=item.name, call_id=getattr(item, "call_id", "") or ""))
