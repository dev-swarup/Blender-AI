from __future__ import annotations

import json
from typing import Any

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
    assert history.compact_budget(1_000_000) == 900_000


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
    resume = history.resume_turn("build a chair", "Objects: Chair")
    assert resume.startswith("<scene>\nObjects: Chair\n</scene>") and "<request>\nbuild a chair\n</request>" in resume
    assert "scene summary above" in resume and "scene summary above" not in history.resume_turn("build a chair", "")


def test_the_reported_count_decides_compaction() -> None:
    assert history.compact_budget(1_000_000) == 900_000, "90% of the window, however large"
    assert history.compact_budget(200_000, 32_000) == 168_000 and history.compact_budget(1_000_000, 32_000) == 900_000, "room for the reply"
    assert history.compact_budget(2000, 32_000) == 1800, "a reserve larger than half the window is ignored"
    assert history.compact_budget(128_000) == 115_200 and history.clear_trigger(128_000) == history.CLEAR_TRIGGER
    assert history.needs_compaction(ANTHROPIC, 10_000, tokens=9500) and not history.needs_compaction(ANTHROPIC, 10_000), (
        "the API's count wins over the estimate"
    )
    assert not history.needs_compaction(ANTHROPIC, 1000, tokens=500)
    assert not history.needs_compaction([{"role": "user", "content": "hi"}], 1000, tokens=5000), "nothing worth compacting when the fixed prompt is the bulk"
    model_output = [{"type": "function_call", "arguments": "x" * 4000}, {"role": "assistant", "content": "y" * 4000}]
    result = {"type": "function_call_output", "output": "z" * 400}
    assert history.fresh_tokens([*model_output, result]) == history.estimate_tokens(result), "the model's own output is already counted"


def anthropic_steps(n: int, result: str = "r" * 20_000) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": "build"}]}]
    for i in range(n):
        name = "spawn_agents" if i == 1 else "run_python"
        items.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": {"code": "c" * 2000}}]})
        items.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": [{"type": "text", "text": result}]}]})
    return items


def openai_steps(n: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [{"role": "user", "content": "build"}]
    for i in range(n):
        items.append({"type": "function_call", "call_id": f"c{i}", "name": "run_python", "arguments": json.dumps({"code": "c" * 2000, "purpose": "p"})})
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": "r" * 20_000})
    return items


def test_old_tool_uses_are_cleared_but_the_latest_stay() -> None:
    items = anthropic_steps(8)
    ids = history.plan_clear(items, 128_000)
    assert ids == {"t0", "t2", "t3"}, "the last four stay and sub-agent reports are kept"
    cleared = history.clear(items, ids)
    assert history.first_change(items, cleared) == 1 and cleared[-1] is items[-1], "untouched turns are the same objects"
    assert cleared[2]["content"][0]["content"] == [{"type": "text", "text": history.CLEARED}]
    code = cleared[1]["content"][0]["input"]["code"]
    assert code.startswith("c" * history.INPUT_KEEP) and "1600 more characters cleared" in code
    assert cleared[4]["content"][0]["content"][0]["text"] == "r" * 20_000, "sub-agent reports are never cleared"
    assert history.plan_clear(cleared, 128_000) == set(), "already cleared results aren't cleared twice"
    assert history.plan_clear(anthropic_steps(8, "ok"), 128_000) == set(), "too little to gain to be worth a cache reset"
    assert history.estimate_tokens(cleared) < history.estimate_tokens(items) - 12_000


def test_openai_items_clear_the_same_way() -> None:
    items = openai_steps(7)
    ids = history.plan_clear(items, 128_000)
    assert ids == {"c0", "c1", "c2"}
    cleared = history.clear(items, ids)
    assert cleared[2]["output"] == history.CLEARED and cleared[8]["output"] == "r" * 20_000
    args = json.loads(cleared[1]["arguments"])
    assert args["purpose"] == "p" and "more characters cleared" in args["code"], "arguments stay valid JSON"
    assert history.clear([{"type": "function_call", "call_id": "x", "arguments": "not json"}], {"x"})[0]["arguments"] == "not json"
