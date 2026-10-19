from __future__ import annotations

from blender_ai.agent import history

ANTHROPIC = [
    {"role": "user", "content": [{"type": "text", "text": "<scene>\nBig scene\n</scene>"}, {"type": "text", "text": "build a chair"}]},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Plan.", "signature": "s" * 4000},
            {"type": "tool_use", "id": "t1", "name": "create_primitive", "input": {"name": "Chair.Seat"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "made " + "x" * 5000}]},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 90_000}},
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "Chair built."}]},
]
OPENAI = [
    {"role": "user", "content": [{"type": "input_text", "text": "<scene>\nBig\n</scene>"}, {"type": "input_text", "text": "make a table"}]},
    {"type": "function_call", "call_id": "c1", "name": "run_python", "arguments": '{"code": "print(1)"}'},
    {"type": "function_call_output", "call_id": "c1", "output": "1"},
    {"type": "reasoning", "summary": []},
    {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "Table done."}]},
]


def test_estimate_counts_images_and_skips_signatures() -> None:
    assert history.estimate_tokens(ANTHROPIC[1]) < 100, "thinking signatures don't count"
    assert history.IMAGE_TOKENS <= history.estimate_tokens(ANTHROPIC[2]) < history.IMAGE_TOKENS + 1500, "an image counts once, not by its bytes"
    assert history.needs_compaction(ANTHROPIC, 1000) and not history.needs_compaction(ANTHROPIC, 200_000)
    assert not history.needs_compaction([], 10)
    assert history.compact_budget(1_000_000) == history.COMPACT_CEILING


def test_render_reads_both_formats() -> None:
    text = history.render(ANTHROPIC, 100_000)
    assert "User: build a chair" in text and "Big scene" not in text, "old scene summaries are left out"
    assert "[tool call] create_primitive" in text and "Chair.Seat" in text and "Plan." not in text
    assert "[tool result]" in text and len(text) < 2500, "long tool results are clipped"
    assert "User: [image]" not in text and "Assistant: Chair built." in text
    other = history.render(OPENAI, 100_000)
    assert other.splitlines() == ["User: make a table", '[tool call] run_python({"code": "print(1)"})', "[tool result] 1", "Assistant: Table done."]
    short = history.render(ANTHROPIC * 20, 600)
    assert len(short) < 700 and "left out" in short and short.endswith("Assistant: Chair built.")


def test_summary_turns() -> None:
    prompt = history.summary_prompt(OPENAI, 128_000)
    assert prompt.startswith("<transcript>") and prompt.endswith(history.SUMMARY_REQUEST)
    assert history.summary_turn("Chair exists.").startswith("<conversation_summary>\nChair exists.\n</conversation_summary>")
