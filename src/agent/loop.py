from __future__ import annotations

import concurrent.futures
import itertools
import json
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..tools import registry
from ..tools.registry import AsyncJob, ImagePart, Mode, ToolError, ToolKind, ToolOutput, ToolSpec
from . import roles
from .bridge import Bridge
from .config import AgentConfig, make_agent
from .messages import AgentError, AgentEvent, LogEntry, LogKind, Phase, Status, ToolCall, ToolResult, Usage, UserTurn
from .roles import Role

CANCELLED_TEXT = "Cancelled by the user before this tool ran."
MAX_LOG = 400


class Host(Protocol):
    def agent_config(self) -> AgentConfig: ...
    def preflight(self, cfg: AgentConfig) -> str | None: ...
    def available(self, requirement: str) -> bool: ...
    def extra_tools(self) -> Sequence[ToolSpec]: ...
    def confirm_python(self) -> bool: ...
    def scene_snapshot(self, track: bool = True) -> str: ...
    def feedback_image(self) -> ImagePart | None: ...
    def execute(self, spec: ToolSpec, args: dict[str, Any]) -> ToolOutput | AsyncJob: ...
    def approval_requested(self) -> None: ...
    def changed(self) -> None: ...
    def finished(self, mode: Mode) -> None: ...


class Agent(Protocol):
    @property
    def has_history(self) -> bool: ...
    def run(self, run: Run, turn: UserTurn) -> str: ...
    def reset(self) -> None: ...


Spawner = Callable[[Callable[[], None]], None]
_ids = itertools.count(1)


def spawn_thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="blender_ai-worker", daemon=True).start()


class Run:
    def __init__(
        self, session: AgentSession, config: AgentConfig, tools: list[ToolSpec], system: str, role: Role, *, parent: Run | None = None, label: str = ""
    ) -> None:
        self.session = session
        self.id = next(_ids)
        self.config = config
        self.tools = tools
        self.system = system
        self.role = role
        self.parent = parent
        self.root: Run = parent.root if parent is not None else self
        self.depth = parent.depth + 1 if parent is not None else 0
        self.label = label
        self.cancel = parent.cancel if parent is not None else threading.Event()
        self.specs = {s.name: s for s in tools}
        self.mutated = False

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    def emit(self, event: AgentEvent) -> None:
        self.session.bridge.post(self.session._on_event, self, event)

    def call_tool(self, call_id: str, name: str, args: Any) -> ToolResult:
        cancelled = ToolResult(CANCELLED_TEXT, is_error=True)
        if self.cancelled:
            return cancelled
        spec = self.specs.get(name)
        if spec is not None and spec.kind is ToolKind.AGENT:
            return self.spawn(call_id, args)
        future: concurrent.futures.Future[ToolResult] = concurrent.futures.Future()
        self.session.bridge.post(self.session._on_tool_call, self, ToolCall(call_id, name, args), future)
        return self._wait(future, cancelled)

    def feedback(self) -> ImagePart | None:
        if self.cancelled:
            return None
        future: concurrent.futures.Future[ImagePart | None] = concurrent.futures.Future()
        self.session.bridge.post(self.session._on_feedback, self, future)
        return self._wait(future, None)

    def spawn(self, call_id: str, args: Any) -> ToolResult:
        errors = registry.validate(roles.SPAWN_SPEC.input_schema, args) if isinstance(args, dict) else ["not an object"]
        if errors:
            return ToolResult("Invalid input:\n" + "\n".join(f"- {e}" for e in errors[:12]), is_error=True)
        if self.depth >= roles.MAX_DEPTH:
            return ToolResult("Sub-agents can't start more sub-agents at this depth; do the work yourself.", True)
        tasks: list[dict[str, Any]] = args["tasks"]
        self.session.bridge.post(self.session._on_spawn, self, call_id, tasks)
        workers = min(len(tasks), roles.MAX_PARALLEL)
        with concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix="blender_ai-subagent") as pool:
            reports = list(pool.map(self._run_child, tasks))
        if self.cancelled:
            return ToolResult(CANCELLED_TEXT, is_error=True)
        done = sum(1 for ok, _ in reports if ok)
        self.session.bridge.post(self.session._on_spawn_done, self, call_id, done, len(tasks))
        header = f"{done} of {len(tasks)} sub-agents finished."
        return ToolResult("\n\n".join([header, *(text for _, text in reports)]), is_error=done == 0)

    def _run_child(self, task: dict[str, Any]) -> tuple[bool, str]:
        role = roles.BY_ID[task["agent"]]
        name = (task.get("name") or "").strip()
        label = f"{role.name} · {name}" if name else role.name
        title = f"## {role.id}" + (f" · {name}" if name else "")
        if self.cancelled:
            return False, f"{title} (cancelled)"
        if not self.session._take_budget():
            return False, f"{title} (not started: the limit of sub-agents per request was reached; do it yourself)"
        tools = self.session._toolset(role, self.depth + 1)
        parent = self.role.name if not self.label else self.label
        child = Run(self.session, self.config, tools, role.subagent_prompt(parent), role, parent=self, label=label)
        snapshot = self._on_main(self.session._child_started, child, task["task"]) or ""
        try:
            agent = self.session.agent_factory(self.config)
            report = agent.run(child, UserTurn(task["task"], None, snapshot)) or "(no report)"
        except AgentError as exc:
            self.session.bridge.post(self.session._child_finished, child, str(exc), exc.kind != "paused")
            return False, f"{title} (stopped: {exc})"
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self.session.bridge.post(self.session._child_finished, child, f"{type(exc).__name__}: {exc}", True)
            return False, f"{title} (failed: {type(exc).__name__}: {exc})"
        if self.cancelled:
            self.session.bridge.post(self.session._child_finished, child, "Cancelled.", False)
            return False, f"{title} (cancelled)"
        self.session.bridge.post(self.session._child_finished, child, report, False)
        return True, f"{title}\n{report.strip()}"

    def _on_main(self, fn: Callable[..., Any], *args: Any) -> Any:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def call() -> None:
            try:
                future.set_result(fn(*args))
            except Exception as exc:
                future.set_exception(exc)

        self.session.bridge.post(call)
        return self._wait(future, None)

    def _wait(self, future: concurrent.futures.Future[Any], on_cancel: Any) -> Any:
        while True:
            try:
                return future.result(timeout=0.1)
            except concurrent.futures.TimeoutError:
                if self.cancelled:
                    return on_cancel


@dataclass
class _Pending:
    run: Run
    call: ToolCall
    future: concurrent.futures.Future[ToolResult]
    spec: ToolSpec
    kind: str
    entry: LogEntry | None = None


class AgentSession:
    def __init__(
        self,
        host: Host,
        bridge: Bridge,
        *,
        agent_factory: Callable[[AgentConfig], Agent] = make_agent,
        catalog: registry.ToolRegistry = registry.REGISTRY,
        spawn: Spawner = spawn_thread,
        mode: Mode = Mode.MODEL,
    ) -> None:
        self.host = host
        self.bridge = bridge
        self.agent_factory = agent_factory
        self.catalog = catalog
        self.spawn = spawn
        self.mode = mode

        self.log: list[LogEntry] = []
        self.usage = Usage()
        self._log_ids = 0
        self._entries: dict[int, LogEntry] = {}

        self._agent: Agent | None = None
        self._endpoint: tuple[Any, ...] | None = None
        self._base_specs: list[ToolSpec] | None = None
        self._run: Run | None = None
        self._last_mode: Mode | None = None
        self._pending: list[_Pending] = []
        self._jobs: dict[tuple[int, str], concurrent.futures.Future[ToolResult]] = {}
        self._executing = False
        self._stream_entries: dict[tuple[int, str], LogEntry] = {}
        self._tool_entries: dict[tuple[int, str], LogEntry] = {}
        self._child_entries: dict[int, LogEntry] = {}
        self._budget = 0
        self._budget_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._run is not None

    @property
    def phase(self) -> Phase:
        if self._run is None:
            return Phase.IDLE
        if self._pending:
            return Phase.APPROVAL if self._pending[0].kind == "approval" else Phase.QUESTION
        if self._jobs:
            return Phase.ASYNC
        return Phase.TOOLS if self._executing else Phase.STREAMING

    @property
    def pending(self) -> ToolCall | None:
        return self._pending[0].call if self._pending else None

    @property
    def pending_agent(self) -> str:
        if not self._pending:
            return ""
        run = self._pending[0].run
        return run.label or run.role.name

    @property
    def tool_names(self) -> list[str]:
        return [s.name for s in self._toolset(roles.info(self.mode), 0)]

    @property
    def has_history(self) -> bool:
        return self._agent is not None and self._agent.has_history

    @property
    def subagents_running(self) -> int:
        return sum(1 for e in self._child_entries.values() if e.status is Status.RUNNING)

    def submit(self, text: str, attachment: ImagePart | None = None) -> None:
        text = text.strip()
        if self.phase is Phase.QUESTION:
            if text:
                self._answer_question(text)
            return
        if self.busy:
            self._log(LogKind.INFO, "Still working. Press Stop (Esc) to interrupt.")
            return
        if not text and attachment is None:
            return
        cfg = self.host.agent_config()
        problem = self.host.preflight(cfg)
        if problem:
            self._log(LogKind.ERROR, problem, status=Status.ERROR)
            return
        if self._endpoint != cfg.endpoint:
            if self.has_history:
                self._log(LogKind.INFO, "The provider or API settings changed, so this is a new conversation.")
            self._agent = None
            self._endpoint = cfg.endpoint
            self._base_specs = None
        agent = self._agent
        if agent is None:
            try:
                agent = self._agent = self.agent_factory(cfg)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                self._log(LogKind.ERROR, f"Couldn't start the agent: {exc}", status=Status.ERROR)
                return
        if self._base_specs is None or not self.has_history:
            self._base_specs = [*self.catalog.specs(None, self.host.available), *self.host.extra_tools()]

        role = roles.info(self.mode)
        if self._last_mode is not None and self._last_mode is not self.mode:
            self._log(LogKind.INFO, f"Now talking to the {role.name}.")
        self._last_mode = self.mode
        snapshot = self.host.scene_snapshot()
        self._log(LogKind.USER, text or "(attached image)")
        run = Run(self, cfg, self._toolset(role, 0), role.system_prompt, role)
        self._run = run
        self._stream_entries = {}
        self._tool_entries = {}
        self._child_entries = {}
        with self._budget_lock:
            self._budget = roles.MAX_SUBAGENTS_PER_REQUEST
        turn = UserTurn(text or "Use the attached image.", attachment, snapshot)

        def work() -> None:
            error: AgentError | None = None
            try:
                agent.run(run, turn)
            except AgentError as exc:
                error = exc
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                error = AgentError(f"{type(exc).__name__}: {exc}", "internal")
            self.bridge.post(self._on_done, run, error)

        self.spawn(work)
        self.host.changed()

    def stop(self) -> None:
        run = self._run
        if run is None:
            return
        run.cancel.set()
        self._run = None
        cancelled = ToolResult(CANCELLED_TEXT, is_error=True)
        for pending in self._pending:
            if not pending.future.done():
                pending.future.set_result(cancelled)
        for future in self._jobs.values():
            if not future.done():
                future.set_result(cancelled)
        self._pending.clear()
        self._jobs.clear()
        for entry in self.log:
            if entry.status in (Status.RUNNING, Status.WAITING):
                entry.status = Status.CANCELLED
        self._log(LogKind.INFO, "Stopped.")

    def approve(self, allow: bool) -> None:
        if not self._pending or self._pending[0].kind != "approval":
            return
        pending = self._pending.pop(0)
        if pending.entry is not None:
            pending.entry.status = Status.OK
        if allow:
            self._execute(pending.run, pending.call, pending.spec, pending.future)
        else:
            self._resolve(pending.run, pending.call, pending.spec, pending.future, error="The user declined to run this code.")
        self._show_pending()

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self.host.changed()

    def new_thread(self) -> None:
        if self.busy:
            self.stop()
        if self._agent is not None:
            self._agent.reset()
        self._base_specs = None
        self._last_mode = None
        self._log(LogKind.INFO, "New conversation.")

    def notify(self, text: str, *, error: bool = False) -> None:
        self._log(LogKind.ERROR if error else LogKind.INFO, text, status=Status.ERROR if error else Status.OK)

    def clear_log(self) -> None:
        self.log.clear()
        self._entries.clear()
        self.host.changed()

    def _toolset(self, role: Role, depth: int) -> list[ToolSpec]:
        base = self._base_specs
        if base is None:
            base = [*self.catalog.specs(None, None)]
        groups = set(role.groups)
        specs = [s for s in base if s.group in groups and not (depth and s.name in roles.SUBAGENT_EXCLUDED)]
        if depth < roles.MAX_DEPTH:
            specs.append(roles.SPAWN_SPEC)
        return sorted(specs, key=lambda s: (registry.GROUP_ORDER.index(s.group), s.name))

    def _take_budget(self) -> bool:
        with self._budget_lock:
            if self._budget <= 0:
                return False
            self._budget -= 1
            return True

    def _active(self, run: Run) -> bool:
        return self._run is not None and run.root is self._run

    def _on_event(self, run: Run, ev: AgentEvent) -> None:
        if not self._active(run):
            return
        if ev.kind == "text":
            entry = self._stream_entries.get((run.id, ev.key))
            if entry is None:
                self._settle_stream(run)
                entry = self._stream_entries[(run.id, ev.key)] = self._log(LogKind.ASSISTANT, "", status=Status.RUNNING, agent=run.label)
            entry.text += ev.text
            entry.updated = time.monotonic()
        elif ev.kind == "reasoning":
            entry = self._stream_entries.get((run.id, ev.key))
            if entry is None:
                self._settle_stream(run)
                entry = self._stream_entries[(run.id, ev.key)] = self._log(LogKind.REASONING, "Reasoning", status=Status.RUNNING, agent=run.label)
            entry.detail += ev.text
            entry.updated = time.monotonic()
        elif ev.kind == "tool":
            self._settle_stream(run)
            if (run.id, ev.call_id) not in self._tool_entries:
                self._tool_entries[(run.id, ev.call_id)] = self._log(LogKind.TOOL, ev.name, status=Status.RUNNING, tool=ev.name, agent=run.label)
        elif ev.kind == "usage" and ev.usage:
            self.usage.add(ev.usage)
        elif ev.kind == "info" and ev.text:
            self._log(LogKind.INFO, ev.text, agent=run.label)
        self.host.changed()

    def _settle_stream(self, run: Run) -> None:
        for (run_id, _key), entry in self._stream_entries.items():
            if run_id == run.id and entry.status is Status.RUNNING:
                entry.status = Status.OK

    def _on_done(self, run: Run, error: AgentError | None) -> None:
        if run is not self._run:
            return
        self._run = None
        for entry in self._stream_entries.values():
            if entry.status is Status.RUNNING:
                entry.status = Status.OK
        for entry in (*self._tool_entries.values(), *self._child_entries.values()):
            if entry.status in (Status.RUNNING, Status.WAITING):
                entry.status = Status.CANCELLED
        self._pending.clear()
        self._jobs.clear()
        if error is not None:
            if error.kind == "paused":
                self._log(LogKind.INFO, str(error))
            else:
                self._log(LogKind.ERROR, str(error), status=Status.ERROR)
        self.host.finished(run.role.mode)
        self.host.changed()

    def _on_spawn(self, run: Run, call_id: str, tasks: list[dict[str, Any]]) -> None:
        if not self._active(run):
            return
        entry = self._tool_entries.get((run.id, call_id))
        if entry is None:
            entry = self._tool_entries[(run.id, call_id)] = self._log(
                LogKind.TOOL, roles.SPAWN_TOOL, status=Status.RUNNING, tool=roles.SPAWN_TOOL, agent=run.label
            )
        names = [t.get("name") or roles.BY_ID[t["agent"]].name for t in tasks]
        entry.detail = f"{len(tasks)} in parallel: " + ", ".join(names)
        self.host.changed()

    def _on_spawn_done(self, run: Run, call_id: str, done: int, total: int) -> None:
        entry = self._tool_entries.get((run.id, call_id))
        if entry is None or not self._active(run):
            return
        entry.status = Status.OK if done else Status.ERROR
        entry.result = f"{done} of {total} finished"
        entry.updated = time.monotonic()
        self.host.changed()

    def _child_started(self, child: Run, task: str) -> str:
        if not self._active(child):
            return ""
        first = task.strip().splitlines()[0] if task.strip() else ""
        self._child_entries[child.id] = self._log(
            LogKind.AGENT, child.label, detail=first, status=Status.RUNNING, agent=child.parent.label if child.parent else ""
        )
        return self.host.scene_snapshot(track=False)

    def _child_finished(self, child: Run, report: str, failed: bool) -> None:
        entry = self._child_entries.get(child.id)
        if entry is None or not self._active(child):
            return
        entry.status = Status.ERROR if failed else (Status.CANCELLED if child.cancelled else Status.OK)
        entry.result = report.strip()
        entry.updated = time.monotonic()
        for (run_id, _call_id), tool_entry in self._tool_entries.items():
            if run_id == child.id and tool_entry.status in (Status.RUNNING, Status.WAITING):
                tool_entry.status = Status.CANCELLED
        self.host.changed()

    def _on_tool_call(self, run: Run, call: ToolCall, future: concurrent.futures.Future[ToolResult]) -> None:
        if not self._active(run):
            if not future.done():
                future.set_result(ToolResult(CANCELLED_TEXT, is_error=True))
            return
        entry = self._tool_entry(run, call)
        entry.detail = _summarize(call.input)
        spec = run.specs.get(call.name)
        if spec is None:
            known = "" if self.catalog.get(call.name) is None else " (it belongs to another agent's toolset)"
            self._resolve(run, call, None, future, error=f"Unknown tool {call.name!r}{known}. Use one of the tools you were given.")
            return
        entry.destructive = spec.destructive
        if not isinstance(call.input, dict):
            self._resolve(run, call, spec, future, error="Tool input must be a JSON object.")
            return
        errors = registry.validate(spec.input_schema, call.input)
        if errors:
            self._resolve(run, call, spec, future, error="Invalid input:\n" + "\n".join(f"- {e}" for e in errors[:12]))
            return
        if call.name == "undo" and self.subagents_running:
            error = "Undo is off while sub-agents are working. Fix or delete the objects directly."
            self._resolve(run, call, spec, future, error=error)
            return
        if spec.kind is ToolKind.QUESTION or (spec.kind is ToolKind.APPROVAL and self.host.confirm_python()):
            entry.status = Status.WAITING
            kind = "question" if spec.kind is ToolKind.QUESTION else "approval"
            self._pending.append(_Pending(run, call, future, spec, kind))
            self._show_pending()
            self.host.changed()
            return
        self._execute(run, call, spec, future)

    def _show_pending(self) -> None:
        if not self._pending or self._pending[0].entry is not None:
            return
        head = self._pending[0]
        if head.kind == "approval":
            head.entry = self._log(
                LogKind.APPROVAL,
                "Approve running this Python code?",
                detail=str(head.call.input.get("code", "")),
                status=Status.WAITING,
                tool=head.call.name,
                agent=head.run.label,
            )
            self.host.approval_requested()
        else:
            options = head.call.input.get("options") or []
            detail = "Options: " + " / ".join(map(str, options)) if options else ""
            head.entry = self._log(
                LogKind.QUESTION, str(head.call.input.get("question", "")), detail=detail, status=Status.WAITING, tool=head.call.name, agent=head.run.label
            )

    def _answer_question(self, text: str) -> None:
        pending = self._pending.pop(0)
        if pending.entry is not None:
            pending.entry.status = Status.OK
        self._log(LogKind.USER, text)
        self._resolve(pending.run, pending.call, pending.spec, pending.future, ToolOutput(text))
        self._show_pending()

    def _execute(self, run: Run, call: ToolCall, spec: ToolSpec, future: concurrent.futures.Future[ToolResult]) -> None:
        self._tool_entry(run, call).status = Status.RUNNING
        self._executing = True
        try:
            out = self.host.execute(spec, call.input)
        except ToolError as exc:
            self._resolve(run, call, spec, future, error=str(exc))
            return
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self._resolve(run, call, spec, future, error=f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._executing = False
        if isinstance(out, AsyncJob):
            self._spawn_job(run, call, spec, out, future)
            return
        self._resolve(run, call, spec, future, out)

    def _spawn_job(self, run: Run, call: ToolCall, spec: ToolSpec, job: AsyncJob, future: concurrent.futures.Future[ToolResult]) -> None:
        self._jobs[(run.id, call.id)] = future
        entry = self._tool_entry(run, call)
        entry.text = f"{call.name}: {job.label}"
        entry.status = Status.RUNNING

        def work() -> None:
            try:
                payload = job.work()
            except Exception as exc:
                self.bridge.post(self._on_job, run, call, spec, job, None, exc)
            else:
                self.bridge.post(self._on_job, run, call, spec, job, payload, None)

        self.spawn(work)
        self.host.changed()

    def _on_job(self, run: Run, call: ToolCall, spec: ToolSpec, job: AsyncJob, payload: Any, exc: BaseException | None) -> None:
        future = self._jobs.pop((run.id, call.id), None)
        if future is None or future.done():
            return
        if exc is not None:
            message = str(exc) if isinstance(exc, ToolError) else f"{type(exc).__name__}: {exc}"
            self._resolve(run, call, spec, future, error=message)
            return
        try:
            outcome = registry.normalize(job.finish(payload))
        except ToolError as err:
            self._resolve(run, call, spec, future, error=str(err))
            return
        except Exception as err:
            traceback.print_exc(file=sys.stderr)
            self._resolve(run, call, spec, future, error=f"{type(err).__name__}: {err}")
            return
        if isinstance(outcome, AsyncJob):
            self._spawn_job(run, call, spec, outcome, future)
            return
        self._resolve(run, call, spec, future, outcome)

    def _resolve(
        self,
        run: Run,
        call: ToolCall,
        spec: ToolSpec | None,
        future: concurrent.futures.Future[ToolResult],
        output: ToolOutput | None = None,
        *,
        error: str | None = None,
    ) -> None:
        entry = self._tool_entry(run, call)
        entry.updated = time.monotonic()
        if error is not None:
            entry.status = Status.ERROR
            entry.result = error
            result = ToolResult(error, is_error=True)
        else:
            assert output is not None
            entry.status = Status.OK
            text = (output.text or "").strip()
            entry.result = _result_summary(text)
            if spec is not None and spec.mutates:
                run.mutated = True
            result = ToolResult(output.text or "ok", list(output.images))
        if not future.done():
            future.set_result(result)
        self.host.changed()

    def _on_feedback(self, run: Run, future: concurrent.futures.Future[ImagePart | None]) -> None:
        image = None
        if self._active(run) and run.mutated:
            run.mutated = False
            try:
                image = self.host.feedback_image()
            except Exception:
                traceback.print_exc(file=sys.stderr)
        if not future.done():
            future.set_result(image)

    def _log(self, kind: LogKind, text: str, *, detail: str = "", status: Status = Status.OK, tool: str | None = None, agent: str = "") -> LogEntry:
        self._log_ids += 1
        now = time.monotonic()
        entry = LogEntry(self._log_ids, kind, text, detail, status, tool, agent=agent, started=now, updated=now)
        self.log.append(entry)
        self._entries[entry.id] = entry
        if len(self.log) > MAX_LOG:
            dropped = self.log[: len(self.log) - MAX_LOG]
            del self.log[: len(dropped)]
            for old in dropped:
                self._entries.pop(old.id, None)
        self.host.changed()
        return entry

    def _tool_entry(self, run: Run, call: ToolCall) -> LogEntry:
        entry = self._tool_entries.get((run.id, call.id))
        if entry is None:
            entry = self._tool_entries[(run.id, call.id)] = self._log(LogKind.TOOL, call.name, status=Status.RUNNING, tool=call.name, agent=run.label)
        call.log_id = entry.id
        return entry


def _result_summary(text: str) -> str:
    if not text.startswith(("{", "[")):
        return text.splitlines()[0][:240] if text else "Done"
    try:
        value = json.loads(text)
    except ValueError:
        return "Done"
    if isinstance(value, list):
        return f"{len(value)} item{'s' * (len(value) != 1)}"
    scalars = {k: v for k, v in value.items() if isinstance(v, str | int | float | bool)} if isinstance(value, dict) else {}
    return _summarize(scalars, 120) if scalars else "Done"


def _summarize(value: Any, limit: int = 160) -> str:
    if not isinstance(value, dict):
        return str(value)[:limit]
    parts = []
    for key, item in value.items():
        shown = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        parts.append(f"{key}={shown}")
    text = "  ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"
