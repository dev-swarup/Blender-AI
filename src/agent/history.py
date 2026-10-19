from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

IMAGE_TOKENS = 1600
COMPACT_SHARE = 0.7
COMPACT_CEILING = 200_000
CALL_CHARS = 600
RESULT_CHARS = 1500
SKIPPED_KEYS = frozenset({"signature", "data", "encrypted_content"})
STOPPED = "(Stopped by the user before finishing.)"
PAUSED = "(Paused at the step limit. The user can say continue.)"
FAILED = "(This reply ended early because of an error.)"
COMPACTED = "Compacted the conversation to free up context."
SUMMARY_SYSTEM = "You compact long conversations between a user and Blender AI, an assistant that works in the user's live Blender scene through tools."
SUMMARY_REQUEST = """\
Summarize the transcript above so the work can continue in a fresh context. Cover:
- what the user asked for, in their words where it matters, and every preference or constraint they stated;
- what exists in the scene because of this work: object, collection and material names, sizes and locations;
- decisions made and why, and what failed or was abandoned;
- what was in progress or planned next.
Plain text, concise, no preamble."""


def estimate_tokens(value: Any) -> int:
    if isinstance(value, str):
        return len(value) // 4
    if isinstance(value, dict):
        if value.get("type") in ("image", "input_image"):
            return IMAGE_TOKENS
        return sum(estimate_tokens(v) for k, v in value.items() if k not in SKIPPED_KEYS)
    if isinstance(value, list | tuple):
        return sum(estimate_tokens(v) for v in value)
    return 0


def compact_budget(window: int) -> int:
    return min(int(window * COMPACT_SHARE), COMPACT_CEILING)


def needs_compaction(items: Sequence[Any], window: int) -> bool:
    return bool(items) and estimate_tokens(list(items)) > compact_budget(window)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _plain(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("text", "input_text", "output_text"):
            parts.append(str(block.get("text", "")))
        elif isinstance(block, dict) and block.get("type") in ("image", "input_image"):
            parts.append("[image]")
    return "\n".join(parts)


def _block_lines(role: str, block: Any) -> list[str]:
    if isinstance(block, str):
        block = {"type": "text", "text": block}
    if not isinstance(block, dict):
        return []
    kind = block.get("type")
    if kind in ("text", "input_text", "output_text"):
        text = str(block.get("text", "")).strip()
        return [f"{role}: {text}"] if text and not text.startswith("<scene>") else []
    if kind == "tool_use":
        return [f"[tool call] {block.get('name')}({_clip(json.dumps(block.get('input') or {}, ensure_ascii=False), CALL_CHARS)})"]
    if kind == "tool_result":
        return [f"[tool result{' error' if block.get('is_error') else ''}] {_clip(_plain(block.get('content')), RESULT_CHARS)}"]
    return []


def _item_lines(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return []
    kind = item.get("type")
    if kind == "function_call":
        return [f"[tool call] {item.get('name')}({_clip(str(item.get('arguments') or ''), CALL_CHARS)})"]
    if kind == "function_call_output":
        return [f"[tool result] {_clip(_plain(item.get('output')), RESULT_CHARS)}"]
    role = item.get("role")
    if role not in ("user", "assistant") or kind not in (None, "message"):
        return []
    content = item.get("content")
    blocks = content if isinstance(content, list) else [content]
    return [line for block in blocks for line in _block_lines(role.capitalize(), block)]


def render(items: Sequence[Any], limit: int) -> str:
    text = "\n".join(line for item in items for line in _item_lines(item))
    if len(text) <= limit:
        return text
    head = limit // 5
    return text[:head] + "\n[… the middle of the conversation is left out …]\n" + text[-(limit - head) :]


def summary_prompt(items: Sequence[Any], window: int) -> str:
    return f"<transcript>\n{render(items, compact_budget(window) * 3)}\n</transcript>\n\n{SUMMARY_REQUEST}"


def summary_turn(summary: str) -> str:
    return (
        f"<conversation_summary>\n{summary.strip()}\n</conversation_summary>\n"
        "The earlier part of this conversation was compacted into the summary above. Continue from it."
    )


def compact_failed(reason: str) -> str:
    return f"Couldn't compact the conversation ({reason}); continuing with the full history."
