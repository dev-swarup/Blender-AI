from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

pytest.importorskip("agents")
pytest.importorskip("anthropic")

from test_loop import FakeHost, catalog, pump

from blender_ai import models, providers
from blender_ai.agent.bridge import Bridge
from blender_ai.agent.config import AgentConfig
from blender_ai.agent.loop import AgentSession
from blender_ai.agent.mcp_bridge import MCPManager, ServerConfig, parse_config_json
from blender_ai.agent.messages import LogKind
from blender_ai.imaging import client as image_api
from blender_ai.tools.registry import ImagePart, Mode, ToolOutput

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
    payload = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(payload)}\n\n"


USAGE = (
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],'
    '"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}\n\n'
)
CHAT_TOOL = (
    chunk({"role": "assistant", "content": "Making a cube. "})
    + chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "make", "arguments": ""}}]})
    + chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"name": "cu'}}]})
    + chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'be"}'}}]})
    + chunk({}, "tool_calls")
    + USAGE
    + "data: [DONE]\n\n"
)
DONE = "data: [DONE]\n\n"
CHAT_DONE = chunk({"role": "assistant", "content": "Done."}) + chunk({}, "stop") + USAGE + DONE


def sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def anthropic_message(blocks: list[dict[str, Any]], stop: str) -> bytes:
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        }
    ]
    for i, block in enumerate(blocks):
        if block["type"] == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [{"type": "thinking_delta", "thinking": block["thinking"]}, {"type": "signature_delta", "signature": block["signature"]}]
        elif block["type"] == "text":
            start, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": block["text"]}]
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        events.append({"type": "content_block_start", "index": i, "content_block": start})
        events += [{"type": "content_block_delta", "index": i, "delta": d} for d in deltas]
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 9}})
    events.append({"type": "message_stop"})
    return sse(events)


MODELS = {
    "object": "list",
    "data": [
        {"id": "model-a", "object": "model", "type": "model", "display_name": "Model A", "created_at": "2026-01-01T00:00:00Z"},
        {"id": "gpt-image-1", "object": "model", "type": "model", "display_name": "Image", "created_at": "2026-01-01T00:00:00Z"},
    ],
    "has_more": False,
    "first_id": "model-a",
    "last_id": "gpt-image-1",
}


class Handler(BaseHTTPRequestHandler):
    seen: list[dict[str, Any]] = []
    chat: list[str] = []
    messages: list[bytes] = []
    route: Any = None
    reject_efforts: frozenset[str] = frozenset()
    summary = "The user asked for a chair; Chair exists at the origin."

    def log_message(self, *args: Any) -> None: ...

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/v1/models", "/alt/models"):
            self._json(200, MODELS)
        else:
            self._json(404, {"error": {"message": f"no {path}"}})

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        path = self.path.split("?")[0]
        headers = {k.lower(): v for k, v in self.headers.items()}
        Handler.seen.append({"path": path, "headers": headers, "raw": raw})
        if path.endswith("/chat/completions"):
            body = json.loads(raw)
            Handler.seen[-1]["body"] = body
            if path.startswith("/chatimg") and body["model"].startswith("flux") and body.get("modalities") != ["image"]:
                self._json(400, {"error": {"message": "This model only supports image output"}})
                return
            if path.startswith("/chatimg"):
                url = "data:image/png;base64," + base64.b64encode(PNG).decode()
                message = {"role": "assistant", "content": "", "images": [{"type": "image_url", "image_url": {"url": url}}]}
                self._json(
                    200,
                    {
                        "id": "x",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    },
                )
                return
            if not body.get("stream"):
                message = {"role": "assistant", "content": Handler.summary}
                usage = {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}
                choice = {"index": 0, "message": message, "finish_reason": "stop"}
                self._json(200, {"id": "x", "object": "chat.completion", "created": 1, "model": body["model"], "choices": [choice], "usage": usage})
                return
            if body.get("reasoning_effort") in Handler.reject_efforts:
                self._json(400, {"error": {"message": f"reasoning_effort '{body['reasoning_effort']}' is not supported by this model"}})
                return
            has_image = "image_url" in json.dumps(body["messages"])
            if body["model"] == "no-vision" and has_image:
                self._json(400, {"error": {"message": "image_url is not supported by this model"}})
                return
            if Handler.route is not None:
                time.sleep(0.3)
                self._stream(Handler.route(body).encode())
                return
            self._stream(Handler.chat.pop(0).encode())
        elif path == "/v1/messages":
            Handler.seen[-1]["body"] = body = json.loads(raw)
            if not body.get("stream"):
                content = [{"type": "text", "text": Handler.summary}]
                reply = {"id": "msg", "type": "message", "role": "assistant", "model": body["model"], "content": content, "stop_reason": "end_turn"}
                self._json(200, {**reply, "stop_sequence": None, "usage": {"input_tokens": 40, "output_tokens": 9}})
                return
            effort = body.get("output_config", {}).get("effort")
            if effort in Handler.reject_efforts:
                message = f"effort '{effort}' is not supported by this model"
                self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": message}})
                return
            if Handler.route is not None:
                time.sleep(0.3)
                self._stream(Handler.route(body))
                return
            self._stream(Handler.messages.pop(0))
        elif path == "/v1/images/generations":
            self._json(200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]})
        elif path == "/v1/images/edits":
            assert b"reference.png" in raw
            self._json(200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]})
        else:
            self._json(404, {"error": {"message": "unknown path"}})


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


ALL_EFFORTS = ("low", "medium", "high", "xhigh", "max")


@pytest.fixture(autouse=True)
def fresh_rejections() -> Any:
    models.forget_rejections()
    yield
    models.forget_rejections()


def listed(model_id: str, *efforts: str) -> models.ModelInfo:
    return models.ModelInfo(model_id, efforts=efforts)


def run_session(host: FakeHost, text: str, attachment: ImagePart | None = None) -> AgentSession:
    bridge = Bridge()
    session = AgentSession(host, bridge, catalog=catalog())
    session.submit(text, attachment)
    pump(bridge, lambda: not session.busy, timeout=30)
    return session


def chat_requests() -> list[dict[str, Any]]:
    return [s["body"] for s in Handler.seen if s["path"].endswith("/chat/completions") and "body" in s]


def test_openai_agent_runs_tools_and_sends_screenshots(server: str) -> None:
    Handler.seen.clear()
    Handler.chat[:] = [CHAT_TOOL, CHAT_DONE]
    host = FakeHost()
    host.cfg = AgentConfig("OPENAI", server, "sk-test", "model-a", reasoning="high", model_info=listed("model-a", "low", "medium", "high"))
    session = run_session(host, "make a cube")
    assert host.executed == ["make"], [e.text for e in session.log]
    first, second = chat_requests()
    assert first["reasoning_effort"] == "high"
    assert any(t["function"]["name"] == "make" for t in first["tools"])
    roles = [m["role"] for m in second["messages"]]
    assert roles[:4] == ["system", "user", "assistant", "tool"]
    tool_message = second["messages"][3]
    assert tool_message["tool_call_id"] == "call_1" and "made cube" in str(tool_message["content"])
    assert roles[-1] == "user" and "image_url" in json.dumps(second["messages"][-1]), "viewport screenshot follows"
    assert Handler.seen[0]["headers"]["authorization"] == "Bearer sk-test"
    texts = [e.text for e in session.log if e.kind is LogKind.ASSISTANT]
    assert texts == ["Making a cube. ", "Done."]
    assert session.usage.input_tokens >= 11


def test_openai_agent_retries_without_images(server: str) -> None:
    Handler.seen.clear()
    Handler.chat[:] = [CHAT_DONE]
    host = FakeHost()
    host.cfg = AgentConfig("OPENAI", server, "sk-test", "no-vision")
    session = run_session(host, "what is this?", ImagePart(PNG))
    assert any("doesn't accept images" in e.text for e in session.log)
    assert [e.text for e in session.log if e.kind is LogKind.ASSISTANT] == ["Done."]
    assert "image_url" not in json.dumps(chat_requests()[-1]["messages"])


def test_openai_agent_steps_down_from_unsupported_effort(server: str) -> None:
    Handler.seen.clear()
    Handler.chat[:] = [CHAT_DONE]
    Handler.reject_efforts = frozenset({"max", "xhigh"})
    host = FakeHost()
    host.cfg = AgentConfig("OPENAI", server, "sk-test", "model-a", reasoning="max", model_info=listed("model-a", *ALL_EFFORTS))
    try:
        session = run_session(host, "think hard")
    finally:
        Handler.reject_efforts = frozenset()
    assert [b.get("reasoning_effort") for b in chat_requests()] == ["max", "xhigh", "high"]
    notes = [e.text for e in session.log if e.kind is LogKind.INFO]
    assert any("using Extra high" in n for n in notes) and any("using High" in n for n in notes), notes
    assert [e.text for e in session.log if e.kind is LogKind.ASSISTANT] == ["Done."]


def test_reasoning_without_listed_levels(server: str) -> None:
    def effort(info: models.ModelInfo, model: str = "model-a") -> Any:
        Handler.seen.clear()
        Handler.chat[:] = [CHAT_DONE]
        host = FakeHost()
        host.cfg = AgentConfig("OPENAI", server, "sk-test", model, reasoning="max", model_info=info)
        session = run_session(host, "hello")
        return chat_requests()[-1].get("reasoning_effort"), [e.text for e in session.log if e.kind is LogKind.INFO]

    assert effort(listed("model-a", "low", "medium", "high"))[0] == "high", "the nearest listed level"
    assert effort(listed("model-a"))[0] == "high", "nothing listed: the standard levels"
    assert effort(models.ModelInfo("model-a", reasoning=False))[0] is None, "the listing says the model has no reasoning"
    assert effort(models.ModelInfo("gpt-5.2"), "gpt-5.2")[0] == "xhigh", "a known family without a listing"
    Handler.reject_efforts = frozenset({"high"})
    try:
        sent, notes = effort(models.ModelInfo("plain-model"), "plain-model")
    finally:
        Handler.reject_efforts = frozenset()
    assert sent is None and any("using its default" in n for n in notes), "a server that takes no effort is remembered"
    assert models.offered_efforts("OPENAI", "plain-model", None, models.effort_key(server, "plain-model")) == ()


def test_anthropic_agent_steps_down_from_unsupported_effort(server: str) -> None:
    Handler.seen.clear()
    Handler.messages[:] = [anthropic_message([{"type": "text", "text": "Done."}], "end_turn")]
    Handler.reject_efforts = frozenset({"max"})
    host = FakeHost()
    host.cfg = AgentConfig("ANTHROPIC", server, "ak-test", "claude-opus-5-5", reasoning="max", model_info=listed("claude-opus-5-5", *ALL_EFFORTS))
    try:
        session = run_session(host, "think hard")
    finally:
        Handler.reject_efforts = frozenset()
    efforts = [s["body"]["output_config"]["effort"] for s in Handler.seen if s["path"] == "/v1/messages"]
    assert efforts == ["max", "xhigh"]
    assert [e.text for e in session.log if e.kind is LogKind.ASSISTANT] == ["Done."]


def test_anthropic_agent_replays_thinking_and_results(server: str) -> None:
    Handler.seen.clear()
    Handler.messages[:] = [
        anthropic_message(
            [
                {"type": "thinking", "thinking": "Plan the cube.", "signature": "sig-abc"},
                {"type": "text", "text": "Adding a cube."},
                {"type": "tool_use", "id": "toolu_1", "name": "make", "input": {"name": "cube"}},
            ],
            "tool_use",
        ),
        anthropic_message([{"type": "text", "text": "Done."}], "end_turn"),
    ]
    host = FakeHost()
    host.cfg = AgentConfig("ANTHROPIC", server, "ak-test", "claude-opus-5-5", reasoning="high", model_info=listed("claude-opus-5-5", *ALL_EFFORTS))
    session = run_session(host, "make a cube")
    assert host.executed == ["make"], [e.text for e in session.log]
    first, second = (s["body"] for s in Handler.seen if s["path"] == "/v1/messages")
    assert first["thinking"]["type"] == "adaptive" and first["output_config"] == {"effort": "high"}
    assert "eager_input_streaming" not in first["tools"][0], "custom URL: portable request only"
    assistant = second["messages"][1]["content"]
    assert assistant[0] == {"type": "thinking", "thinking": "Plan the cube.", "signature": "sig-abc"}
    results = second["messages"][2]["content"]
    assert results[0]["type"] == "tool_result" and results[0]["content"][0]["text"] == "made cube"
    assert results[-1]["type"] == "image", "viewport screenshot after a scene change"
    assert Handler.seen[0]["headers"]["x-api-key"] == "ak-test"


def conversation(host: FakeHost) -> tuple[AgentSession, Any]:
    bridge = Bridge()
    session = AgentSession(host, bridge, catalog=catalog())

    def say(text: str) -> None:
        session.submit(text)
        pump(bridge, lambda: not session.busy, timeout=30)

    return session, say


def anthropic_requests() -> list[dict[str, Any]]:
    return [s["body"] for s in Handler.seen if s["path"] == "/v1/messages" and s["body"].get("stream")]


def test_openai_follow_ups_keep_the_conversation_across_tabs(server: str) -> None:
    Handler.seen.clear()
    Handler.chat[:] = [CHAT_DONE, CHAT_DONE]
    host = FakeHost()
    host.cfg = AgentConfig("OPENAI", server, "sk-test", "model-a")
    session, say = conversation(host)
    say("build a chair")
    session.set_mode(Mode.MATERIAL)
    say("make it oak")
    first, second = chat_requests()
    texts = json.dumps(second["messages"])
    assert "Material Generator" in second["messages"][0]["content"] and "Model Builder" in first["messages"][0]["content"]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert "build a chair" in texts and "Done." in texts and "make it oak" in texts


def test_anthropic_follow_ups_keep_thinking_until_the_tab_changes(server: str) -> None:
    Handler.seen.clear()
    thought = [{"type": "thinking", "thinking": "Plan.", "signature": "sig-1"}, {"type": "text", "text": "Done."}]
    Handler.messages[:] = [anthropic_message(thought, "end_turn") for _ in range(3)]
    host = FakeHost()
    host.cfg = AgentConfig("ANTHROPIC", server, "ak-test", "claude-opus-5-5", model_info=listed("claude-opus-5-5", *ALL_EFFORTS))
    session, say = conversation(host)
    say("build a chair")
    say("taller legs")
    session.set_mode(Mode.MATERIAL)
    say("make it oak")
    _first, second, third = anthropic_requests()
    assert [m["role"] for m in third["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert second["messages"][1]["content"][0]["type"] == "thinking", "same tab: the history is append-only"
    kinds = [b["type"] for m in third["messages"] if m["role"] == "assistant" for b in m["content"]]
    assert "thinking" not in kinds and "text" in kinds, "a new system prompt and tool set start without old thinking"
    assert "build a chair" in json.dumps(third["messages"]) and "Material Generator" in third["system"][0]["text"]


@pytest.mark.parametrize("provider", ["OPENAI", "ANTHROPIC"])
def test_long_conversations_are_compacted(server: str, provider: str) -> None:
    Handler.seen.clear()
    Handler.chat[:] = [CHAT_DONE, CHAT_DONE]
    Handler.messages[:] = [anthropic_message([{"type": "text", "text": "Done."}], "end_turn") for _ in range(2)]
    host = FakeHost()
    model = "claude-opus-5-5" if provider == "ANTHROPIC" else "model-a"
    host.cfg = AgentConfig(provider, server, "key", model, model_info=models.ModelInfo(model, context=2000))
    session, say = conversation(host)
    say("build a chair " + "with carved details " * 400)
    say("now a table")
    if provider == "ANTHROPIC":
        summary_request, main = [s["body"] for s in Handler.seen if s["path"] == "/v1/messages"][1:]
        turns = main["messages"]
    else:
        summary_request, main = chat_requests()[1:]
        turns = main["messages"][1:]
    assert "<transcript>" in json.dumps(summary_request) and "build a chair" in json.dumps(summary_request)
    assert len(turns) == 1 and turns[0]["role"] == "user", "the old turns are replaced by the summary"
    sent = json.dumps(turns[0])
    assert "<conversation_summary>" in sent and "Chair exists" in sent and "now a table" in sent
    assert any("Compacted the conversation" in e.text for e in session.log if e.kind is LogKind.INFO)
    assert session.usage.turns == 3


def tool_calls(*calls: tuple[str, str, dict[str, Any]]) -> str:
    out = ""
    for i, (call_id, name, args) in enumerate(calls):
        out += chunk({"tool_calls": [{"index": i, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}]})
        out += chunk({"tool_calls": [{"index": i, "function": {"arguments": json.dumps(args)}}]})
    return out + chunk({}, "tool_calls") + USAGE + DONE


def final(text: str) -> str:
    return chunk({"role": "assistant", "content": text}) + chunk({}, "stop") + USAGE + DONE


def test_openai_agent_spawns_parallel_subagents(server: str) -> None:
    Handler.seen.clear()
    tasks = [{"agent": "model_builder", "task": "Build house A.", "name": "A"}, {"agent": "model_builder", "task": "Build house B.", "name": "B"}]

    def route(body: dict[str, Any]) -> str:
        text = json.dumps(body["messages"])
        answered = any(m["role"] == "tool" for m in body["messages"])
        if "You are a sub-agent" in text:
            name = "A" if "house A" in text else "B"
            return final(f"Built house {name}.") if answered else tool_calls(("c1", "make", {"name": name}))
        return final("Village done.") if answered else tool_calls(("s1", "spawn_agents", {"tasks": tasks}))

    Handler.route = route
    try:
        host = FakeHost()
        host.cfg = AgentConfig("OPENAI", server, "sk-test", "model-a")
        start = time.monotonic()
        session = run_session(host, "build a village")
        elapsed = time.monotonic() - start
    finally:
        Handler.route = None
    assert sorted(a["name"] for _, a in host.calls) == ["A", "B"], [e.text for e in session.log]
    parent = [b for b in chat_requests() if "You are a sub-agent" not in json.dumps(b["messages"])]
    assert any(t["function"]["name"] == "spawn_agents" for t in parent[0]["tools"])
    report = next(m for m in parent[-1]["messages"] if m["role"] == "tool")["content"]
    assert "2 of 2 sub-agents finished" in str(report) and "Built house A." in str(report)
    assert [e.text for e in session.log if e.kind is LogKind.ASSISTANT and not e.agent][-1] == "Village done."
    assert elapsed < 6 * 0.3 + 1.0, elapsed


def test_openai_agent_runs_a_turns_tool_calls_in_order(server: str) -> None:
    Handler.seen.clear()
    names = [f"part{n}" for n in range(6)]
    Handler.chat[:] = [tool_calls(*((f"c{n}", "make", {"name": name}) for n, name in enumerate(names))), CHAT_DONE]
    host = FakeHost()
    host.cfg = AgentConfig("OPENAI", server, "sk-test", "model-a")
    run_session(host, "make parts")
    assert [a["name"] for _, a in host.calls] == names


def test_anthropic_agent_spawns_subagents(server: str) -> None:
    Handler.seen.clear()
    task = {"agent": "material_generator", "task": "Make an oak material.", "name": "Oak"}

    def route(body: dict[str, Any]) -> bytes:
        system = json.dumps(body["system"])
        answered = any(isinstance(m["content"], list) and any(b.get("type") == "tool_result" for b in m["content"]) for m in body["messages"])
        if "You are a sub-agent" in system:
            if answered:
                return anthropic_message([{"type": "text", "text": "Oak material ready."}], "end_turn")
            return anthropic_message([{"type": "tool_use", "id": "t1", "name": "make", "input": {"name": "Oak"}}], "tool_use")
        if answered:
            return anthropic_message([{"type": "text", "text": "All set."}], "end_turn")
        return anthropic_message([{"type": "tool_use", "id": "s1", "name": "spawn_agents", "input": {"tasks": [task]}}], "tool_use")

    Handler.route = route
    try:
        host = FakeHost()
        host.cfg = AgentConfig("ANTHROPIC", server, "ak-test", "claude-opus-5-5")
        session = run_session(host, "texture it")
    finally:
        Handler.route = None
    assert host.calls == [("make", {"name": "Oak"})], [e.text for e in session.log]
    bodies = [s["body"] for s in Handler.seen if s["path"] == "/v1/messages"]
    child = next(b for b in bodies if "You are a sub-agent" in json.dumps(b["system"]))
    assert "You are Material Generator" in json.dumps(child["system"])
    last = bodies[-1]["messages"][-1]["content"][0]
    assert last["type"] == "tool_result" and "Oak material ready." in json.dumps(last["content"])


def test_list_models_for_both_providers_and_root_fallback(server: str) -> None:
    listed = providers.list_models(models.OPENAI, server, "k")
    assert [m.id for m in listed] == ["gpt-image-1", "model-a"]
    assert providers.list_models(models.OPENAI, f"{server}/alt", "k")[0].id == "gpt-image-1"
    from blender_ai.agent import openai_agent

    assert openai_agent.api_root(f"{server}/alt") == f"{server}/alt"
    claude = providers.list_models(models.ANTHROPIC, server, "k")
    assert models.find(claude, "model-a").label == "Model A"


def test_images_generate_edit_and_chat_fallback(server: str) -> None:
    cfg = image_api.ImageConfig(server, "img-key", "gpt-image-1")
    assert image_api.generate(cfg, image_api.texture_prompt("moss")) == PNG
    sent = json.loads(Handler.seen[-1]["raw"])
    assert sent["model"] == "gpt-image-1" and "Seamless tileable texture of moss" in sent["prompt"]
    assert image_api.generate(cfg, "rock", reference=PNG) == PNG
    assert Handler.seen[-1]["path"] == "/v1/images/edits"
    via_chat = image_api.ImageConfig(f"{server}/chatimg/v1", "k", "google/gemini-2.5-flash-image")
    assert image_api.generate(via_chat, "bark") == PNG
    assert Handler.seen[-1]["body"]["modalities"] == ["image", "text"]
    flux = image_api.ImageConfig(f"{server}/chatimg/v1", "k", "flux-klein", image_only=True)
    assert image_api.generate(flux, "bark") == PNG
    assert Handler.seen[-1]["body"]["modalities"] == ["image"]
    guessed = image_api.ImageConfig(f"{server}/chatimg/v1", "k", "flux-klein")
    assert image_api.generate(guessed, "bark") == PNG, "falls back to image-only output"


def test_parse_mcp_json_configs() -> None:
    configs = parse_config_json(
        json.dumps(
            {
                "mcpServers": {
                    "meshy": {"command": "npx", "args": ["-y", "meshy-mcp"], "env": {"MESHY_API_KEY": "k"}},
                    "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer t"}},
                }
            }
        )
    )
    assert configs[0] == ServerConfig("meshy", "npx -y meshy-mcp", env={"MESHY_API_KEY": "k"})
    assert configs[1].url == "https://example.com/mcp" and configs[1].headers == {"Authorization": "Bearer t"}
    with pytest.raises(ValueError):
        parse_config_json("[]")


def test_mcp_stdio_server_tools() -> None:
    script = os.path.join(os.path.dirname(__file__), "mcp_demo_server.py")
    manager = MCPManager()
    try:
        manager.sync([ServerConfig("demo", command=f'"{sys.executable}" "{script}"')])
        end = time.monotonic() + 60
        while time.monotonic() < end and not manager.status()["demo"].startswith(("connected", "failed")):
            time.sleep(0.1)
        assert manager.status()["demo"].startswith("connected"), manager.status()
        specs = {s.name: s for s in manager.tool_specs({"make"})}
        assert set(specs) == {"demo_add", "demo_preview", "demo_broken"}
        assert specs["demo_add"].input_schema["required"] == ["a", "b"]
        job = specs["demo_add"].func(a=2, b=3)
        out = job.finish(job.work())
        assert isinstance(out, ToolOutput) and out.text == "5"
        image = manager.call("demo", "preview", {"name": "x"})
        assert image.images and image.images[0].data == PNG
        from blender_ai.tools.registry import ToolError

        with pytest.raises(ToolError, match="broken"):
            manager.call("demo", "broken", {})
    finally:
        manager.shutdown()
