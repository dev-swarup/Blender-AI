from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

OPENAI = "OPENAI"
ANTHROPIC = "ANTHROPIC"

REASONING_LEVELS = ("auto", "low", "medium", "high")
LEVEL_LABELS = {"auto": "Auto", "low": "Low", "medium": "Medium", "high": "High"}


_IMAGE_ID = re.compile(
    r"gpt-image|dall-?e|imagen|flux|stable-diffusion|sdxl|\bsd-?3|seedream|ideogram|recraft|kandinsky|hidream|"
    r"playground-v|qwen-image|image-gen|nano-banana|midjourney|-image\b|image-preview|text-to-image",
    re.IGNORECASE,
)
_NOT_CHAT = re.compile(
    r"embed|whisper|tts|transcri|moderation|rerank|realtime|speech|audio|davinci|babbage|\bada\b|search-api|"
    r"computer-use-preview|-ocr\b|-native-audio",
    re.IGNORECASE,
)
_VISION_ID = re.compile(
    r"claude|gpt-4o|gpt-4\.1|gpt-5|\bo3\b|\bo4|gemini|vision|-vl\b|-vl-|llava|pixtral|grok-4|llama-4|gemma-3|"
    r"mistral-medium|mistral-small-3|kimi-k2\.5|glm-4\.\dv|minicpm-v",
    re.IGNORECASE,
)
_CHAT_PREFERENCE = (
    "claude-opus",
    "claude-sonnet",
    "gpt-5",
    "gemini-3",
    "gemini-2.5-pro",
    "gpt-4.1",
    "claude",
    "gpt-4o",
    "grok-4",
    "deepseek-chat",
    "qwen3",
    "gemini",
    "llama",
)
_IMAGE_PREFERENCE = ("gpt-image", "nano-banana", "gemini-2.5-flash-image", "imagen", "flux", "seedream", "dall-e-3")
_SMALL = ("mini", "nano", "lite", "free", "flash", "haiku")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    tools: bool | None = None
    capabilities: Mapping[str, Any] | None = None
    max_output: int | None = None
    free: bool | None = None

    @property
    def label(self) -> str:
        return self.name or self.id

    @property
    def is_image(self) -> bool:
        return "image" in self.outputs if self.outputs else bool(_IMAGE_ID.search(self.id))

    @property
    def is_chat(self) -> bool:
        if self.outputs:
            return "text" in self.outputs and self.tools is not False
        if _NOT_CHAT.search(self.id):
            return False
        return not _IMAGE_ID.search(self.id)

    @property
    def vision(self) -> bool | None:
        if self.inputs:
            return "image" in self.inputs
        if self.capabilities and isinstance(self.capabilities.get("image_input"), Mapping):
            return bool(self.capabilities["image_input"].get("supported"))
        return True if _VISION_ID.search(self.id) else None


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(p.strip().lower() for p in re.split(r"[+,\s]+", value) if p.strip())
    return tuple(str(v).lower() for v in value) if isinstance(value, (list, tuple)) else ()


def _free(pricing: Any) -> bool | None:
    if not isinstance(pricing, Mapping) or not pricing:
        return None
    try:
        values = [float(v) for v in pricing.values() if v is not None]
    except (TypeError, ValueError):
        return None
    return None if not values or any(v < 0 for v in values) else all(v == 0 for v in values)


def parse_model(entry: Any) -> ModelInfo | None:
    if isinstance(entry, str):
        return ModelInfo(entry)
    if not isinstance(entry, Mapping):
        return None
    model_id = str(entry.get("id") or entry.get("model") or entry.get("name") or "").strip()
    if not model_id:
        return None
    arch = entry.get("architecture") if isinstance(entry.get("architecture"), Mapping) else {}
    inputs = _strings(arch.get("input_modalities") or entry.get("input_modalities"))
    outputs = _strings(arch.get("output_modalities") or entry.get("output_modalities"))
    modality = arch.get("modality")
    if not inputs and not outputs and isinstance(modality, str) and "->" in modality:
        left, right = modality.split("->", 1)
        inputs, outputs = _strings(left), _strings(right)

    caps = entry.get("capabilities") if isinstance(entry.get("capabilities"), Mapping) else None
    tools: bool | None = None
    if caps is not None and not outputs:
        if caps.get("image_generation") is True:
            outputs = ("image",)
        if caps.get("vision") is True and not inputs:
            inputs = ("text", "image")
        if isinstance(caps.get("function_calling"), bool):
            tools = caps["function_calling"]
    supported = entry.get("supported_parameters")
    if isinstance(supported, list) and supported:
        tools = "tools" in supported

    name = entry.get("display_name") or entry.get("name") or ""
    name = name if isinstance(name, str) and name != model_id else ""
    max_out = entry.get("max_tokens") or entry.get("max_output_tokens")
    if not isinstance(max_out, int):
        top = entry.get("top_provider") if isinstance(entry.get("top_provider"), Mapping) else {}
        max_out = top.get("max_completion_tokens")
    return ModelInfo(model_id, name, inputs, outputs, tools, caps, max_out if isinstance(max_out, int) and max_out > 0 else None, _free(entry.get("pricing")))


def parse_models(payload: Any) -> list[ModelInfo]:
    entries: Any = payload
    if isinstance(payload, Mapping):
        entries = payload.get("data") or payload.get("models") or []
    if not isinstance(entries, list):
        return []
    seen: set[str] = set()
    out: list[ModelInfo] = []
    for entry in entries:
        info = parse_model(entry)
        if info is not None and info.id not in seen:
            seen.add(info.id)
            out.append(info)
    return sorted(out, key=lambda m: m.id.lower())


def chat_models(listed: list[ModelInfo]) -> list[ModelInfo]:
    return [m for m in listed if m.is_chat]


def image_models(listed: list[ModelInfo]) -> list[ModelInfo]:
    return [m for m in listed if m.is_image]


def _natural(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text.lower())]


def _pick(listed: list[ModelInfo], preference: tuple[str, ...]) -> str:
    if not listed:
        return ""
    for prefix in preference:
        hits = [m.id for m in listed if prefix in m.id.lower()]
        if hits:
            hits.sort(key=_natural, reverse=True)
            hits.sort(key=lambda i: any(s in i.lower() for s in _SMALL))
            return hits[0]
    return listed[0].id


def default_chat(listed: list[ModelInfo]) -> str:
    return _pick(chat_models(listed), _CHAT_PREFERENCE)


def default_image(listed: list[ModelInfo]) -> str:
    return _pick(image_models(listed), _IMAGE_PREFERENCE)


def find(listed: list[ModelInfo], model_id: str) -> ModelInfo | None:
    return next((m for m in listed if m.id == model_id), None)


def short_label(model_id: str, listed: list[ModelInfo] | None = None) -> str:
    info = find(listed or [], model_id)
    if info is not None and info.name:
        return info.name.split(": ", 1)[-1].removeprefix("Claude ")
    return model_id.rsplit("/", 1)[-1] if model_id else ""


ThinkingStyle = Literal["always", "adaptive", "explicit", "budget", "none"]

EFFORTS = ("low", "medium", "high")
_BUDGETS = {"low": 2048, "medium": 8000, "high": 16000}
_MIN_MAX_TOKENS = {"low": 8000, "medium": 16000, "high": 32000}
DEFAULT_MAX_TOKENS = 32000


@dataclass(frozen=True)
class AnthropicCaps:
    thinking: ThinkingStyle
    max_output: int = 128_000
    fallbacks: bool = False
    vision: bool = True


_FAMILIES: tuple[tuple[str, AnthropicCaps], ...] = (
    ("opus-5-5", AnthropicCaps("always", fallbacks=True)),
    ("fable-5", AnthropicCaps("always", fallbacks=True)),
    ("sonnet-5-5", AnthropicCaps("adaptive", fallbacks=True)),
    ("haiku-5", AnthropicCaps("adaptive")),
    ("opus-5", AnthropicCaps("adaptive", fallbacks=True)),
    ("sonnet-5", AnthropicCaps("adaptive")),
    ("opus-4-8", AnthropicCaps("explicit")),
    ("opus-4-7", AnthropicCaps("explicit")),
    ("opus-4-6", AnthropicCaps("explicit")),
    ("sonnet-4-6", AnthropicCaps("explicit")),
    ("-4-5", AnthropicCaps("budget", max_output=64_000)),
    ("-4-1", AnthropicCaps("budget", max_output=32_000)),
    ("-4-", AnthropicCaps("budget", max_output=32_000)),
    ("3-7", AnthropicCaps("budget", max_output=64_000)),
)


def _caps_from_api(caps: Mapping[str, Any], max_output: int | None) -> AnthropicCaps | None:
    def supported(*path: str) -> bool:
        node: Any = caps
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return False
            node = node[key]
        return isinstance(node, Mapping) and bool(node.get("supported"))

    if supported("thinking", "types", "adaptive"):
        style: ThinkingStyle = "adaptive" if supported("thinking", "types", "disabled") else "always"
    elif supported("thinking", "types", "enabled"):
        style = "budget"
    elif "thinking" in caps:
        style = "none"
    else:
        return None
    vision = supported("image_input") if "image_input" in caps else True
    return AnthropicCaps(style, max_output or 64_000, vision=vision)


def anthropic_caps(model_id: str, info: ModelInfo | None = None) -> AnthropicCaps:
    if info is not None and info.capabilities:
        from_api = _caps_from_api(info.capabilities, info.max_output)
        if from_api is not None:
            return from_api
    key = model_id.lower().replace(".", "-")
    for fragment, caps in _FAMILIES:
        if fragment in key and "claude" in key:
            return caps
    if "claude" in key:
        return AnthropicCaps("adaptive")
    return AnthropicCaps("none", max_output=info.max_output if info and info.max_output else 32_000)


def anthropic_params(caps: AnthropicCaps, level: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    params: dict[str, Any] = {"max_tokens": min(max_tokens, caps.max_output)}
    if level not in EFFORTS or caps.thinking == "none":
        return params
    params["max_tokens"] = min(max(max_tokens, _MIN_MAX_TOKENS[level]), caps.max_output)
    if caps.thinking == "budget":
        budget = _BUDGETS[level]
        params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        params["max_tokens"] = min(max(params["max_tokens"], budget + 8000), caps.max_output)
        return params
    params["thinking"] = {"type": "adaptive", "display": "summarized"}
    params["output_config"] = {"effort": level}
    return params
