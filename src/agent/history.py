from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

IMAGE_TOKENS = 1600
COMPACT_SHARE = 0.9
CLEAR_TRIGGER = 40_000
CLEAR_AT_LEAST = 15_000
CLEAR_KEEP = 4
CLEAR_MIN_GAIN = 100
INPUT_KEEP = 400
KEPT_TOOLS = frozenset({"spawn_agents"})
CALL_CHARS = 600
RESULT_CHARS = 1500
SKIPPED_KEYS = frozenset({"signature", "data", "encrypted_content"})
MODEL_ITEMS = frozenset({"function_call", "reasoning"})
STOPPED = "(Stopped by the user before finishing.)"
PAUSED = "(Paused at the step limit. The user can say continue.)"
FAILED = "(This reply ended early because of an error.)"
COMPACTED = "Compacted the conversation to free up context."
CLEARED = "(Old tool result cleared to save context. Inspect the scene if you need it again.)"
SUMMARY_SYSTEM = "You compact long conversations between a user and Blender AI, an assistant that works in the user's live Blender scene through tools."
SUMMARY_REQUEST = """\
Summarize the transcript above so the work can continue in a fresh context. Cover:
- what the user asked for, in their words where it matters, and every preference or constraint they stated;
- what exists in the scene because of this work: object, collection and material names, sizes and locations;
- decisions made and why, and what failed or was abandoned;
- the latest request: which of its steps are done and which remain.
Plain text, concise, no preamble. Don't call tools."""


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


def fixed_tokens(system: str, tools: Sequence[Any]) -> int:
    return estimate_tokens(system) + sum(estimate_tokens([t.name, t.description, json.dumps(t.input_schema)]) for t in tools)


def fresh_tokens(items: Sequence[Any]) -> int:
    return estimate_tokens([i for i in items if isinstance(i, dict) and i.get("role") != "assistant" and i.get("type") not in MODEL_ITEMS])


def compact_budget(window: int, reserve: int = 0) -> int:
    share = int(window * COMPACT_SHARE)
    return min(share, window - reserve) if reserve < window // 2 else share


def clear_trigger(window: int) -> int:
    return min(CLEAR_TRIGGER, compact_budget(window) // 2)


def clear_at_least(window: int) -> int:
    return min(CLEAR_AT_LEAST, compact_budget(window) // 8)


def needs_compaction(items: Sequence[Any], window: int, tokens: int | None = None, reserve: int = 0) -> bool:
    size = estimate_tokens(list(items))
    budget = compact_budget(window, reserve)
    return bool(items) and (size if tokens is None else tokens) > budget and size > budget // 4


def _cleared(content: Any) -> bool:
    return content == CLEARED or _plain(content) == CLEARED


def _shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > INPUT_KEEP:
        return value[:INPUT_KEEP] + f"… [{len(value) - INPUT_KEEP} more characters cleared from history; this already ran]"
    if isinstance(value, dict):
        return {k: _shorten(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shorten(v) for v in value]
    return value


def _clip_json(arguments: Any) -> Any:
    try:
        return json.dumps(_shorten(json.loads(arguments)), ensure_ascii=False)
    except (TypeError, ValueError):
        return arguments


def _input(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _tool_uses(items: Sequence[Any]) -> dict[str, dict[str, Any]]:
    uses: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "function_call":
            uses[str(item.get("call_id"))] = {"name": item.get("name"), "input": _input(item.get("arguments")), "result": None}
        elif kind == "function_call_output" and str(item.get("call_id")) in uses:
            uses[str(item.get("call_id"))]["result"] = item.get("output")
        elif isinstance(item.get("content"), list):
            for block in item["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    uses[str(block.get("id"))] = {"name": block.get("name"), "input": block.get("input"), "result": None}
                elif isinstance(block, dict) and block.get("type") == "tool_result" and str(block.get("tool_use_id")) in uses:
                    uses[str(block.get("tool_use_id"))]["result"] = block.get("content")
    return uses


def _gain(use: dict[str, Any]) -> int:
    return estimate_tokens(use["result"]) - estimate_tokens(CLEARED) + estimate_tokens(use["input"]) - estimate_tokens(_shorten(use["input"]))


def plan_clear(items: Sequence[Any], window: int) -> set[str]:
    answered = [(key, use) for key, use in _tool_uses(items).items() if use["result"] is not None]
    old = [
        (key, use)
        for key, use in answered[: max(0, len(answered) - CLEAR_KEEP)]
        if use["name"] not in KEPT_TOOLS and not _cleared(use["result"]) and _gain(use) >= CLEAR_MIN_GAIN
    ]
    return {key for key, _ in old} if old and sum(_gain(use) for _, use in old) >= clear_at_least(window) else set()


def _clear_block(block: Any, ids: set[str]) -> Any:
    if isinstance(block, dict) and block.get("type") == "tool_use" and str(block.get("id")) in ids:
        return {**block, "input": _shorten(block.get("input"))}
    if isinstance(block, dict) and block.get("type") == "tool_result" and str(block.get("tool_use_id")) in ids:
        cleared = {"type": "tool_result", "tool_use_id": block["tool_use_id"], "content": [{"type": "text", "text": CLEARED}]}
        return {**cleared, "is_error": True} if block.get("is_error") else cleared
    return block


def clear(items: Sequence[Any], ids: set[str]) -> list[Any]:
    out: list[Any] = []
    for item in items:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "function_call" and str(item.get("call_id")) in ids:
            out.append({**item, "arguments": _clip_json(item.get("arguments"))})
        elif kind == "function_call_output" and str(item.get("call_id")) in ids:
            out.append({**item, "output": CLEARED})
        elif isinstance(item, dict) and isinstance(item.get("content"), list) and ids:
            content = [_clear_block(b, ids) for b in item["content"]]
            out.append({**item, "content": content} if any(a is not b for a, b in zip(content, item["content"], strict=True)) else item)
        else:
            out.append(item)
    return out


def first_change(before: Sequence[Any], after: Sequence[Any]) -> int:
    return next((i for i, (a, b) in enumerate(zip(before, after, strict=False)) if a is not b), min(len(before), len(after)))


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


def resume_turn(request: str, scene: str) -> str:
    head = f"<scene>\n{scene}\n</scene>\n" if scene else ""
    return (
        f"{head}The conversation was compacted while you were working on the user's latest request:\n<request>\n{request.strip()}\n</request>\n"
        + ("The scene summary above is its current state. " if scene else "")
        + "Carry on from where the summary leaves off; don't redo finished steps."
    )


def compact_failed(reason: str) -> str:
    return f"Couldn't compact the conversation ({reason}); continuing with the full history."
