from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from blender_ai.agent import roles
from blender_ai.agent.bridge import Bridge
from blender_ai.agent.config import AgentConfig
from blender_ai.agent.loop import CANCELLED_TEXT, AgentSession
from blender_ai.agent.messages import AgentError, AgentEvent, LogKind, Phase, Status
from blender_ai.tools.registry import AsyncJob, ImagePart, Mode, ToolError, ToolKind, ToolRegistry, ToolSpec, obj, string


def catalog() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(obj({"name": string()}, ["name"]), group="objects")
    def make(name: str) -> str:
        """Make an object with this name (test tool)."""
        if name == "bad":
            raise ToolError("cannot make bad things")
        return f"made {name}"

    @reg.tool(obj({}), group="scene", mutates=False)
    def look() -> str:
        """Inspect without changing anything (test tool)."""
        return "nothing changed"

    @reg.tool(obj({"code": string(), "purpose": string()}, ["code", "purpose"]), group="meta", kind=ToolKind.APPROVAL)
    def run_python(code: str, purpose: str) -> str:
        """Run code (test tool)."""
        return f"ran {code}"

    @reg.tool(obj({"question": string()}, ["question"]), group="meta", kind=ToolKind.QUESTION, mutates=False)
    def ask_user(question: str) -> None:
        """Ask the user (test tool)."""

    @reg.tool(obj({}), group="meta")
    def undo() -> str:
        """Undo (test tool)."""
        return "undone"

    @reg.tool(obj({"prompt": string()}, ["prompt"]), group="textures", kind=ToolKind.ASYNC)
    def slow(prompt: str) -> AsyncJob:
        """Two-stage async job (test tool)."""
        stage2 = AsyncJob("stage 2", lambda: prompt.upper(), lambda p: f"finished {p}")
        return AsyncJob("stage 1", lambda: prompt, lambda p: stage2)

    @reg.tool(obj({}), group="animation")
    def keyframe() -> str:
        """Animation-only tool (test tool)."""
        return "keyed"

    @reg.tool(obj({}), group="integrations", requires="helper")
    def helper_tool() -> str:
        """Only when the helper is available (test tool)."""
        return "helped"

    return reg


class FakeHost:
    def __init__(self) -> None:
        self.confirm = False
        self.helper = False
        self.problem: str | None = None
        self.executed: list[str] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.finished_modes: list[Mode] = []
        self.feedback = True
        self.extra: list[ToolSpec] = []
        self.approvals_requested = 0
        self.cfg = AgentConfig("OPENAI", "https://api.example.com/v1", "key", "model-a")

    def agent_config(self) -> AgentConfig:
        return self.cfg

    def preflight(self, cfg: AgentConfig) -> str | None:
        return self.problem

    def available(self, requirement: str) -> bool:
        return self.helper

    def extra_tools(self) -> list[ToolSpec]:
        return self.extra

    def confirm_python(self) -> bool:
        return self.confirm

    def scene_snapshot(self, track: bool = True) -> str:
        return "Scene: empty"

    def approval_requested(self) -> None:
        self.approvals_requested += 1

    def feedback_image(self) -> ImagePart | None:
        return ImagePart(b"\x89PNGfake") if self.feedback else None

    def execute(self, spec: Any, args: dict[str, Any]) -> Any:
        from blender_ai.tools import registry

        self.executed.append(spec.name)
        self.calls.append((spec.name, dict(args)))
        return registry.call(spec, args)

    def changed(self) -> None: ...

    def finished(self, mode: Mode) -> None:
        self.finished_modes.append(mode)


class ScriptedAgent:
    instances: list[ScriptedAgent] = []

    def __init__(self, cfg: AgentConfig, scripts: list[list[Callable[[Any], Any]]]) -> None:
        self.cfg = cfg
        self.scripts = scripts
        self.results: list[Any] = []
        self.runs: list[Any] = []
        self.history: list[str] = []
        ScriptedAgent.instances.append(self)

    @property
    def has_history(self) -> bool:
        return bool(self.history)

    def reset(self) -> None:
        self.history = []

    def run(self, run: Any, turn: Any) -> str:
        self.runs.append(run)
        self.history.append(turn.text)
        for step in self.scripts.pop(0):
            self.results.append(step(run))
        return f"report for {turn.text}"


def text(t: str) -> Callable[[Any], Any]:
    return lambda run: run.emit(AgentEvent("text", key="1:0", text=t))


def call(call_id: str, name: str, args: Any) -> Callable[[Any], Any]:
    def step(run: Any) -> Any:
        run.emit(AgentEvent("tool", name=name, call_id=call_id))
        return run.call_tool(call_id, name, args)

    return step


def feedback() -> Callable[[Any], Any]:
    return lambda run: run.feedback()


def fail(error: AgentError) -> Callable[[Any], Any]:
    def step(run: Any) -> Any:
        raise error

    return step


def session_with(scripts: list[list[Any]], host: FakeHost | None = None) -> tuple[AgentSession, FakeHost, Bridge]:
    host = host or FakeHost()
    bridge = Bridge()
    ScriptedAgent.instances.clear()
    session = AgentSession(host, bridge, agent_factory=lambda cfg: ScriptedAgent(cfg, scripts), catalog=catalog())
    return session, host, bridge


def pump(bridge: Bridge, until: Callable[[], bool], timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        bridge.drain()
        if until():
            return
        time.sleep(0.002)
    raise AssertionError("timed out waiting for the session")


def agent() -> ScriptedAgent:
    return ScriptedAgent.instances[-1]


def test_tool_then_feedback_then_text() -> None:
    session, host, bridge = session_with([[text("Making it. "), call("t1", "make", {"name": "cube"}), feedback()]])
    session.submit("make a cube")
    pump(bridge, lambda: not session.busy)
    result, image = agent().results[1], agent().results[2]
    assert result.text == "made cube" and not result.is_error
    assert isinstance(image, ImagePart), "a mutating tool earns a screenshot"
    assert host.executed == ["make"] and host.finished_modes == [Mode.MODEL]
    kinds = [e.kind for e in session.log]
    assert kinds == [LogKind.USER, LogKind.ASSISTANT, LogKind.TOOL]
    assert session.log[1].text == "Making it. " and session.log[1].status is Status.OK
    assert session.log[2].status is Status.OK and session.log[2].result == "made cube"


def test_no_screenshot_after_read_only_tools() -> None:
    session, _host, bridge = session_with([[call("t1", "look", {}), feedback()]])
    session.submit("look")
    pump(bridge, lambda: not session.busy)
    assert agent().results[1] is None


def test_errors_bad_input_and_unknown_tools() -> None:
    session, _host, bridge = session_with([[call("t1", "make", {"name": "bad"}), call("t2", "make", {}), call("t3", "nope", {}), call("t4", "make", [])]])
    session.submit("go")
    pump(bridge, lambda: not session.busy)
    r1, r2, r3, r4 = agent().results
    assert r1.is_error and "cannot make bad things" in r1.text
    assert r2.is_error and "missing required property" in r2.text
    assert r3.is_error and "Unknown tool" in r3.text
    assert r4.is_error and "JSON object" in r4.text


def test_python_runs_directly_unless_confirmation_is_on() -> None:
    session, host, bridge = session_with([[call("t1", "run_python", {"code": "x=1", "purpose": "p"})]])
    session.submit("run it")
    pump(bridge, lambda: not session.busy)
    assert host.executed == ["run_python"] and agent().results[0].text == "ran x=1"

    host = FakeHost()
    host.confirm = True
    session, host, bridge = session_with([[call("t1", "run_python", {"code": "x=1", "purpose": "p"})]], host)
    session.submit("run it")
    pump(bridge, lambda: session.phase is Phase.APPROVAL)
    assert host.executed == [] and session.log[-1].kind is LogKind.APPROVAL
    assert host.approvals_requested == 1, "the host is asked to show the approval dialog"
    session.approve(True)
    pump(bridge, lambda: not session.busy)
    assert host.executed == ["run_python"]


def test_denied_python() -> None:
    host = FakeHost()
    host.confirm = True
    session, host, bridge = session_with([[call("t1", "run_python", {"code": "x", "purpose": "p"})]], host)
    session.submit("run")
    pump(bridge, lambda: session.phase is Phase.APPROVAL)
    session.approve(False)
    pump(bridge, lambda: not session.busy)
    assert agent().results[0].is_error and "declined" in agent().results[0].text and host.executed == []


def test_question_is_answered_by_next_submit() -> None:
    session, _host, bridge = session_with([[call("q1", "ask_user", {"question": "Wood or metal?"})]])
    session.submit("make a chair")
    pump(bridge, lambda: session.phase is Phase.QUESTION)
    assert session.log[-1].kind is LogKind.QUESTION
    session.submit("")
    assert session.phase is Phase.QUESTION, "empty answers are ignored"
    session.submit("metal")
    pump(bridge, lambda: not session.busy)
    assert agent().results[0].text == "metal"


def test_async_jobs_chain() -> None:
    session, _host, bridge = session_with([[call("a1", "slow", {"prompt": "moss"})]])
    session.submit("texture")
    pump(bridge, lambda: not session.busy)
    assert agent().results[0].text == "finished MOSS"


def test_stop_cancels_waiting_calls() -> None:
    host = FakeHost()
    host.confirm = True
    session, host, bridge = session_with([[call("t1", "make", {"name": "a"}), call("t2", "run_python", {"code": "x", "purpose": "p"})]], host)
    session.submit("go")
    pump(bridge, lambda: session.phase is Phase.APPROVAL)
    session.stop()
    assert not session.busy and session.log[-1].text == "Stopped."
    pump(bridge, lambda: len(agent().results) == 2)
    assert agent().results[0].text == "made a"
    assert agent().results[1].text == CANCELLED_TEXT and agent().results[1].is_error


def test_agent_errors_and_pauses_are_logged() -> None:
    session, _host, bridge = session_with([[fail(AgentError("The API key was rejected.", "auth"))]])
    session.submit("hello")
    pump(bridge, lambda: not session.busy)
    assert session.log[-1].kind is LogKind.ERROR and session.log[-1].text == "The API key was rejected."

    session, _host, bridge = session_with([[fail(AgentError("Paused after 3 steps", "paused"))]])
    session.submit("hello")
    pump(bridge, lambda: not session.busy)
    assert session.log[-1].kind is LogKind.INFO


def test_preflight_blocks_sending() -> None:
    host = FakeHost()
    host.problem = "Add your API key"
    session, _host, _bridge = session_with([], host)
    session.submit("hi")
    assert not session.busy and session.log[-1].text == "Add your API key"


def test_tools_are_frozen_per_conversation_and_include_mcp() -> None:
    host = FakeHost()
    session, host, bridge = session_with([[], [], []], host)
    session.submit("one")
    pump(bridge, lambda: not session.busy)
    first = {s.name for s in agent().runs[0].tools}
    assert "helper_tool" not in first and "make" in first

    host.helper = True
    host.extra = [ToolSpec("meshy_text_to_3d", "[MCP] make a model", obj({}), lambda: "ok", "mcp")]
    session.submit("two")
    pump(bridge, lambda: not session.busy)
    assert {s.name for s in agent().runs[1].tools} == first, "same conversation, same tools"

    session.new_thread()
    session.submit("three")
    pump(bridge, lambda: not session.busy)
    third = {s.name for s in agent().runs[2].tools}
    assert {"helper_tool", "meshy_text_to_3d"} <= third


def test_tabs_share_one_conversation_with_their_own_prompt_and_tools() -> None:
    session, _host, bridge = session_with([[], [], []])
    session.submit("build a chair")
    pump(bridge, lambda: not session.busy)
    shared = agent()
    session.set_mode(Mode.MATERIAL)
    session.submit("make it oak")
    pump(bridge, lambda: not session.busy)
    assert agent() is shared and shared.history == ["build a chair", "make it oak"], "the next tab continues the same conversation"
    first, second = shared.runs
    assert "You are Model Builder" in first.system and "You are Material Generator" in second.system
    first_tools = {s.name for s in first.tools}
    second_tools = {s.name for s in second.tools}
    assert "keyframe" not in first_tools and {"run_python", "spawn_agents"} <= first_tools and {"run_python", "spawn_agents"} <= second_tools
    assert any("Material Generator" in e.text for e in session.log if e.kind is LogKind.INFO)
    session.set_mode(Mode.MODEL)
    session.submit("taller legs")
    pump(bridge, lambda: not session.busy)
    assert agent() is shared and shared.history == ["build a chair", "make it oak", "taller legs"]
    session.new_thread()
    assert not session.has_history


def test_tools_of_another_agent_are_explained() -> None:
    session, _host, bridge = session_with([[call("t1", "keyframe", {})]])
    session.submit("animate it")
    pump(bridge, lambda: not session.busy)
    result = agent().results[0]
    assert result.is_error and "another agent" in result.text


class RoutedAgent:
    def __init__(self, scripts: dict[str, list[Callable[[Any], Any]]], log: list[tuple[str, int]]) -> None:
        self.scripts = scripts
        self.log = log
        self.history: list[str] = []
        self.results: list[Any] = []

    @property
    def has_history(self) -> bool:
        return bool(self.history)

    def reset(self) -> None:
        self.history = []

    def run(self, run: Any, turn: Any) -> str:
        self.history.append(turn.text)
        self.log.append((run.role.id, run.depth))
        for step in self.scripts.get(turn.text, []):
            self.results.append(step(run))
        if turn.text == "fail":
            raise AgentError("Paused after 3 steps", "paused")
        return f"built {turn.text}"


def routed_session(scripts: dict[str, list[Any]]) -> tuple[AgentSession, list[tuple[str, int]], Bridge, list[Any]]:
    log: list[tuple[str, int]] = []
    agents: list[RoutedAgent] = []

    def factory(cfg: AgentConfig) -> RoutedAgent:
        agents.append(RoutedAgent(scripts, log))
        return agents[-1]

    bridge = Bridge()
    session = AgentSession(FakeHost(), bridge, agent_factory=factory, catalog=catalog())
    return session, log, bridge, agents


def spawn(call_id: str, *tasks: tuple[str, str, str]) -> Callable[[Any], Any]:
    payload = {"tasks": [{"agent": a, "task": t, "name": n} for a, t, n in tasks]}
    return call(call_id, roles.SPAWN_TOOL, payload)


def test_spawned_subagents_run_in_parallel_and_report_back() -> None:
    session, log, bridge, agents = routed_session(
        {
            "build a village": [spawn("s1", ("model_builder", "house", "House"), ("model_builder", "well", "Well"))],
            "house": [call("h1", "make", {"name": "House"})],
            "well": [call("w1", "make", {"name": "Well"})],
        }
    )
    session.set_mode(Mode.SCENE)
    session.submit("build a village")
    pump(bridge, lambda: not session.busy)
    assert sorted(log) == [("model_builder", 1), ("model_builder", 1), ("scene_builder", 0)]
    report = agents[0].results[0]
    assert not report.is_error and "2 of 2 sub-agents finished" in report.text
    assert "## model_builder · House\nbuilt house" in report.text and "built well" in report.text
    children = [e for e in session.log if e.kind is LogKind.AGENT]
    assert sorted(e.text for e in children) == ["Model Builder · House", "Model Builder · Well"]
    assert all(e.status is Status.OK and e.result.startswith("built") for e in children)
    tool_rows = [e for e in session.log if e.kind is LogKind.TOOL and e.tool == "make"]
    assert sorted(e.agent for e in tool_rows) == ["Model Builder · House", "Model Builder · Well"]
    assert all(e.status is Status.OK for e in tool_rows)
    spawn_row = next(e for e in session.log if e.tool == roles.SPAWN_TOOL)
    assert spawn_row.status is Status.OK and spawn_row.detail == "2 in parallel: House, Well" and spawn_row.result == "2 of 2 finished"


def test_subagents_nest_up_to_the_depth_limit() -> None:
    session, log, bridge, agents = routed_session(
        {
            "map": [spawn("s1", ("model_builder", "district", "District"))],
            "district": [spawn("s2", ("model_builder", "tower", "Tower"))],
            "tower": [spawn("s3", ("model_builder", "brick", "Brick"))],
        }
    )
    session.submit("map")
    pump(bridge, lambda: not session.busy)
    assert ("model_builder", 2) in log and ("model_builder", 3) not in log
    tower = next(a for a in agents if a.history == ["tower"])
    assert tower.results[0].is_error and "Unknown tool" in tower.results[0].text, "no spawn tool at max depth"


def test_failed_subagent_is_reported_and_others_continue() -> None:
    session, _log, bridge, agents = routed_session({"go": [spawn("s1", ("material_generator", "fail", "Broken"), ("material_generator", "wood", "Wood"))]})
    session.submit("go")
    pump(bridge, lambda: not session.busy)
    report = agents[0].results[0]
    assert "1 of 2 sub-agents finished" in report.text and "stopped: Paused after 3 steps" in report.text
    assert not report.is_error


def test_subagents_have_no_undo_and_the_main_agent_cant_undo_while_they_work() -> None:
    def tools(run: Any) -> list[str]:
        return sorted(run.specs)

    def parent_undoes(run: Any) -> Any:
        return run.parent.call_tool("u1", "undo", {})

    session, _log, bridge, agents = routed_session({"go": [tools, spawn("s1", ("model_builder", "house", "House"))], "house": [tools, parent_undoes]})
    session.submit("go")
    pump(bridge, lambda: not session.busy)
    main_tools, child = agents[0].results[0], agents[1]
    assert {"undo", "ask_user"} <= set(main_tools)
    assert "undo" not in child.results[0] and "ask_user" not in child.results[0] and "make" in child.results[0]
    refused = child.results[1]
    assert refused.is_error and "Undo is off while sub-agents are working" in refused.text


def test_stop_cancels_subagents() -> None:
    def wait_until_stopped(run: Any) -> Any:
        while not run.cancelled:
            time.sleep(0.005)
        return "stopped"

    session, _log, bridge, agents = routed_session({"go": [spawn("s1", ("animator", "slow", "Slow"))], "slow": [wait_until_stopped]})
    session.submit("go")
    pump(bridge, lambda: session.subagents_running == 1)
    session.stop()
    pump(bridge, lambda: len(agents) == 2 and len(agents[0].results) == 1)
    assert agents[0].results[0].text == CANCELLED_TEXT


def test_changing_the_endpoint_starts_a_new_agent() -> None:
    host = FakeHost()
    bridge = Bridge()
    created: list[ScriptedAgent] = []

    def factory(cfg: AgentConfig) -> ScriptedAgent:
        created.append(ScriptedAgent(cfg, [[]]))
        return created[-1]

    session = AgentSession(host, bridge, agent_factory=factory, catalog=catalog())
    session.submit("one")
    pump(bridge, lambda: not session.busy)
    host.cfg = AgentConfig("ANTHROPIC", "https://api.anthropic.com", "k2", "claude-x")
    session.submit("two")
    pump(bridge, lambda: not session.busy)
    assert len(created) == 2 and created[1].cfg.provider == "ANTHROPIC"
    assert any("new conversation" in e.text for e in session.log)


def test_tool_results_are_summarised_for_the_transcript() -> None:
    from blender_ai.agent.loop import _result_summary

    assert _result_summary('{"name": "Crate", "type": "MESH", "dimensions": [1, 1, 1]}') == "name=Crate  type=MESH"
    assert _result_summary("[1, 2, 3]") == "3 items"
    assert _result_summary("Created Crate\nmore") == "Created Crate"
    assert _result_summary("") == "Done" and _result_summary("{not json") == "Done"
