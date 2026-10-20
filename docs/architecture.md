# Architecture

This document describes how Blender AI is put together: the process and threading model, the agents,
the tool layer, the interface, and the packaging. It also records the constraints the code works
around, since the source carries no comments. The user-facing documentation is in the
[README](../README.md) and the tool reference is generated into [tools.md](tools.md).

- [Repository layout](#repository-layout)
- [System overview](#system-overview)
- [Runtime](#runtime)
- [Agents](#agents)
- [Tools](#tools)
- [Integrations](#integrations)
- [User interface](#user-interface)
- [Packaging and releases](#packaging-and-releases)
- [Constraints](#constraints)
- [Testing](#testing)

## Repository layout

| Path | Contents |
|---|---|
| `src/` | The extension: package code, manifest, icons and bundled wheels. Installed, this is the folder Blender loads. |
| `tests/` | pytest suites and the Blender-side scripts (`run_tools.py`, `ui_*.py`, `live_*.py`). |
| `scripts/` | Build, wheel vendoring, icon and documentation generation, development sync, and `devimport.py`. |
| `docs/` | This document, the generated tool reference and the screenshots used by the README. |

Tests and scripts import the source as `blender_ai`, not `src`. `scripts/devimport.py` installs an
import hook that maps that name to `src/`, and `tests/conftest.py` calls it for pytest. Inside the
package every import is relative, so the folder name never matters. Paths below are relative to `src/`.

## System overview

```
prompt bar (ui/overlay.py, ui/input_modal.py)          sidebar and settings (ui/panel.py, preferences.py)
        │ submit                                                │
        ▼                                                       ▼
AgentSession (agent/loop.py) ── Host protocol ──► BlenderHost (agent/host.py)
   main thread: log, phases,                           tool execution and undo, scene snapshot,
   approvals, questions, async jobs                    screenshots, MCP tool specs
        │ spawn worker                                          ▲
        ▼                                                       │ Run.call_tool / Run.feedback
Agent.run(run, turn) on a worker thread ────────────────────────┘   (posted through the Bridge)
   ├─ OpenAIAgent    (agent/openai_agent.py)     OpenAI Agents SDK ─► any Chat Completions API
   ├─ AnthropicAgent (agent/anthropic_agent.py)  Anthropic SDK ─────► Messages API
   └─ spawn_agents ─► Run.spawn: sub-agents on their own threads, same gateway, same cancel flag
        │ events (text, reasoning, tool starts, usage)
        ▼
Bridge (agent/bridge.py), drained on the main thread by a bpy.app.timers callback (runtime.py)

MCPManager (agent/mcp_bridge.py): its own asyncio thread, one long-lived task per server,
tools exposed as ToolSpecs
```

`AgentSession` owns the conversation and all state that the UI reads. `Host` is the narrow interface
through which the session reaches Blender, so the session, both agents and the sub-agent machinery
run under plain pytest against scripted hosts.

## Runtime

### Threading

`bpy` is not thread-safe, so a request runs in an agent on a worker thread and never touches Blender
directly. When the model calls a tool, the agent calls `Run.call_tool`, which posts the call to the
bridge and blocks on a future. The main thread validates the arguments, asks for approval or an answer
when the tool needs one, executes the tool as a single undo step, and resolves the future.

Asynchronous tools (`AsyncJob`: texture generation, URL imports and MCP calls) run their slow half on
another worker and finish on the main thread. A timer drains the bridge every 50 ms while a request
runs and every 250 ms when idle. Stop sets the run's cancel flag and resolves every waiting future as
cancelled, so the agent winds down and leaves a valid history.

### Start-up

`addon.register()` registers only the preferences and the bar's draw handler, about 70 ms, so the
extension barely affects Blender's start-up time. A background thread pre-imports the heavy pure-Python
modules (numpy, asyncio, urllib, subprocess). A timer then runs the steps in `runtime.py` on the main
thread, one per tick: load the tool catalog, register operators, panels, keymaps and the approval
dialog, then create the session and the MCP manager. Until `state.ready` the bar draws a
*Loading Blender AI…* card. In background mode everything loads inside `register()`. The provider SDKs
are imported later still, on worker threads, when the model list is first fetched.

## Agents

### Common contract

An agent implements `Agent.run(run, turn)`. It streams events to the session, calls tools through the
run and keeps its own history. Both implementations guarantee the same invariants:

- Unanswered tool calls are closed as cancelled before a run ends.
- A run that ends on a user message (stopped, failed or paused at the step limit) gets a short assistant
  note appended, such as "Stopped by the user before finishing.", so roles keep alternating and the next
  prompt starts a clean turn.
- Tool results and the viewport screenshot reach the model on the next request; only the latest
  screenshot is ever sent.

### OpenAI-compatible agent

`OpenAIAgent` builds an Agents SDK `Agent` per request with one `FunctionTool` per Blender AI tool
(`strict_json_schema=False`) and an `OpenAIChatCompletionsModel` over `AsyncOpenAI(base_url=...)`.
`Runner.run_streamed` drives the loop and `result.to_input_list()` is the history for the next request.

Chat Completions tool messages cannot carry images, so tool images and the screenshot are added as a user
message just before each model call through `RunConfig.call_model_input_filter`. They are not persisted.
A request that a model rejects for images or `stream_options` is retried once without them, and the choice
is remembered for that model. Tracing is disabled. The SDK runs a turn's tool calls concurrently, so the
agent takes a FIFO lock around ordinary tools to let Blender see them in the order the model emitted them,
and lets `spawn_agents` calls run in parallel.

### Anthropic agent

`AnthropicAgent` is a hand-written loop on `client.beta.messages.stream`. The history is append-only, so
thinking blocks replay intact and the prompt cache keeps hitting: the tool list is frozen for a
conversation, and tool results plus the screenshot go in one user message.

When the model, system prompt or tool names differ from the previous request (a model switch, or a move
to another tab), earlier thinking blocks are dropped first, because on models with preserved thinking a
block is bound to the prompt and tools that produced it. Fields only Anthropic's own API accepts (eager
tool-input streaming, top-level cache control, refusal fallbacks, adaptive thinking) are left out for any
other URL.

### Models and reasoning levels

`models.py` parses what `/models` returns (OpenAI, OpenRouter with modalities, LiteLLM-style
capabilities, Anthropic with capabilities) into `ModelInfo`, sorts the entries into chat and image
models and picks defaults. No model list is built in. The only model knowledge in the source is how to
shape an Anthropic request: adaptive thinking with an effort, or a thinking budget on older families.

`models.offered_efforts` chooses the reasoning levels from the best source available, in this order:

1. `ModelInfo.efforts` from the listing (OpenRouter's `reasoning.supported_efforts` with `default_effort`,
   or Anthropic's `capabilities.effort`).
2. None, when the listing says the model does not reason (`ModelInfo.reasoning` is false).
3. The model family: `_EFFORT_FAMILIES` for GPT-5 and GPT-5.x, none for GPT-4 and chat-latest models,
   Low to Max for Claude models with thinking.
4. Low, Medium and High for anything unknown, such as a local server.

The picker offers Auto plus those levels, and `models.effective_effort` maps the saved choice to the
nearest offered level at or below it, never to "none". When a request is rejected for its effort,
`AgentConfig.reject_effort` records the refusal in a process-wide table keyed by API URL and model. An edge
level (Off, Minimal, Extra high, Max) is dropped and the request retried with the next one. A middle level
means the endpoint takes no effort parameter, so none is sent again for that model. The picker reads the
same table and stops offering what the API refused.

### Conversation and compaction

The four tabs share one conversation. `AgentSession` keeps a single agent, and switching tabs changes only
the role (system prompt and toolset) of the next run. A new conversation starts on **New Conversation**,
when the provider, URL or key changes, and when another `.blend` is opened.

Before each message, `agent/history.py` estimates the history's size: text length over four, a fixed cost
per image, thinking signatures ignored. Past 70% of the model's context window, capped at 200k tokens, the
agent compacts. The window comes from `ModelInfo.context` (`context_length` or `max_input_tokens`) and
defaults to 200k for Anthropic and 128k otherwise. Compaction renders the history as a plain transcript
(user and assistant text, tool calls and clipped results; scene summaries, images and thinking omitted),
asks the same model for a summary in a separate request without tools, and replaces the history with an
empty one. The summary then leads the next user message in a `<conversation_summary>` block.

The transcript format works for every OpenAI-compatible server and for Anthropic, and the separate request
keeps the main conversation append-only. If summarizing fails, the full history is sent and the chat says so.

### Roles and sub-agents

`agent/roles.py` defines one role per tab: Model Builder, Scene Builder, Animator and Material Generator.
A role is a system prompt (shared conventions plus the role's workflow) and a set of tool groups. Every
role includes `meta`, `agents`, `scene`, `objects`, `materials`, `render` and `mcp`. The base catalog is
frozen per conversation so the request prefix stays cacheable, and a role's toolset is that catalog
filtered by its groups.

`spawn_agents` is a `ToolKind.AGENT` tool. `Run.call_tool` does not send it to the main thread. The calling
worker validates it and runs the tasks in a thread pool of up to `MAX_PARALLEL` threads. Each task is a
child `Run` with the role's tools, the role's prompt plus a sub-agent section, a fresh agent instance and a
scene snapshot. The result of the tool is each child's final message.

| Limit | Value | Purpose |
|---|---|---|
| `MAX_DEPTH` | 2 | Levels of nesting; children below the limit receive `spawn_agents` too. |
| `MAX_PARALLEL` | 4 | Concurrent children per call. |
| `MAX_SUBAGENTS_PER_REQUEST` | 16 | Total children per request. |

Children share the root run's cancel flag, so Stop ends everything, and their tool calls go through the
same main-thread gateway with log entries labelled by the child. Blender keeps one undo history per scene,
so sub-agents do not receive `undo` or `ask_user` (`roles.SUBAGENT_EXCLUDED`), and the main agent's `undo`
is refused while any sub-agent runs.

## Tools

Each tool is a function registered with `ToolRegistry.tool(schema, group=...)`. Its docstring is the
model-facing description, and registration fails without one. The registry validates arguments against the
JSON schema before the function runs.

| Kind | Behavior |
|---|---|
| `NORMAL` | Runs on the main thread as one undo step. |
| `ASYNC` | Returns an `AsyncJob`: the work runs on a worker, `finish` runs on the main thread. MCP tools use this kind. |
| `APPROVAL` | Asks the user first when *Ask before running Python* is on. |
| `QUESTION` | Pauses the run until the user answers. |
| `AGENT` | Handled by the calling worker; see sub-agents. |

A spec may also declare `destructive`, `mutates` and `requires`. A tool with `requires` joins a toolset
only while the named helper add-on is usable, and the helpers' tools appear from the next conversation after
they are enabled. The [tool reference](tools.md) is generated from the registry by `scripts/gen_tool_docs.py`.

`packing.py` packs every image that appears during a mutating tool call (compared by pointer before and
after, including async finishes) when *Pack textures into the .blend* is on, which is the default. Untouched
generated images are not packed, because Blender rebuilds them from their settings, and images that existed
before the call stay as they are. A `save_pre` handler also packs images still pointing into the texture
folder or the temp folder.

## Integrations

### MCP servers

`MCPManager` runs an asyncio loop on its own thread. Each enabled server gets a task that opens the
transport (stdio, streamable HTTP or SSE) and a `ClientSession`, lists the tools and waits until told to
stop, since anyio requires the task that opened them to close them. Tools become `ToolSpec`s named
`server_tool` with kind `ASYNC`, so both agents call them like any other tool. On Windows `deps.prepare()`
processes pywin32's `.pth` file first, because Blender adds extension packages without running `.pth` files.

### Helper add-ons

Blender's extension manifest cannot declare dependencies on other extensions, so Blender AI installs its
helpers through Blender's own extension system. `integrations/registry.py` lists them: Rigify, which ships
with Blender, and three packages from extensions.blender.org (`mpfb`, `looptools`, `extra_mesh_objects`).

`integrations/installer.py` enables the ticked helpers that are installed but off, syncs the
extensions.blender.org listing by repository directory rather than by index, then installs each missing
package with `extensions.package_install(repo_directory, pkg_id, enable_on_install=True)`. In a window each
install is Blender's modal job with progress in the status bar. A timer watches `Window.modal_operators`,
starts the next package when one finishes and reports what is still missing.

The installer avoids `package_install_marked`: in a window that operator drops every marked package the
Get Extensions list hides, so a search left in that list failed with "No installable packages marked".

On first start, when online access is on and a helper is missing, `ui/dialogs.py` opens the *Install helper
add-ons?* dialog once, and either answer sets the hidden `helpers_prompted` preference. Blender opens
dialogs at the mouse cursor and has no position option, so the prompt first warps the cursor to the point
that centres the dialog in the window. The preferences' *Helper add-ons* section shows each helper's
status and the same install button.

## User interface

### Prompt bar

The bar is the only custom-drawn element: a `POST_PIXEL` draw handler on every 3D Viewport that reads theme
colours at draw time and uses Blender's 20 px widget unit. Two modules are free of `bpy` and unit-tested:

- `ui/layout.py` lays out the card, the pill row (dropping the image pill, then reasoning, then model on
  narrow viewports), the Submit button, the tab strip and the transcript above the text field. The card grows
  upwards with the transcript, up to the rows allowed for the current state. Text lines that do not fit are
  skipped whole, because Blender's text clipping does not trim glyphs.
- `ui/transcript.py` turns the session log into chat lines: user messages on a band, replies and tool calls as
  bullets, each tool's one-line result (`LogEntry.result`; JSON results become `key=value` pairs) on a
  connector line, "Thought for Ns" from the entry's timestamps, and sub-agents folded into one row whose
  children appear when it is expanded. Its `Scroll` keeps a pixel offset from the bottom: zero follows new
  lines, anything else holds while content grows.

### Input handling

No modal operator runs permanently, because Blender skips autosave while one does. A 3D View keymap item
hit-tests left clicks and passes misses through, and the modal runs only while the bar has focus or a
dropdown is open. A second keymap item (`blender_ai.log_scroll`, wheel and trackpad) scrolls the transcript
when the pointer is over it and passes the event through otherwise, so the viewport still zooms.

The model, image-model and reasoning dropdowns are drawn by the overlay too (`ui/dropdown.py`, no `bpy`):
anchored above their pill, filtered as you type, scrolled with the wheel and picked on release. Buttons that
open Blender popups (the settings popover, the file browser) act on mouse release, since a Blender menu
opened on press would receive the release and close at once. Every drawing primitive switches alpha blending
back on first, because `blf.draw` leaves it off.

The approval dialog (`ui/dialogs.py`) is a native `invoke_props_dialog`, opened from a timer when an approval
becomes the head of the queue. `execute` approves and `cancel` declines.

## Packaging and releases

The SDKs depend on native wheels (pydantic_core, jiter, cryptography, rpds-py, cffi, pywin32), so the
zip carries a build of each for every platform and Python version.

1. `scripts/vendor_wheels.py` resolves dependencies per target with uv, because pip cannot evaluate markers
   such as `sys_platform` for another platform. It downloads wheels for Python 3.11, 3.12 and 3.13 and skips
   packages Blender already ships.
2. `scripts/build.py` stages `src/` into `build/`, lists every wheel in the staged manifest, sets `platforms`
   to the ones pydantic_core has wheels for, and runs `blender --command extension build` and
   `extension validate`. The result is one `blender_ai-<version>.zip` of about 80 MB for every platform.
   Blender extracts only the wheels whose tags match the running platform and Python; this was checked by
   installing the zip into empty 4.2 and 5.2 profiles on Windows.
3. With `--download-url`, `build.py` also runs `extension server-generate` and rewrites each entry's
   `archive_url` to the release asset, writing `dist/index.json`.

The `Release` workflow (`.github/workflows/build.yml`) runs on `v*` tags. It checks that the tag matches the
manifest version, builds the zip and `index.json` and attaches both to the GitHub release. Because
`releases/latest/download/index.json` always resolves to the newest non-prerelease, that URL works as a
Blender remote repository: users who add it get update notifications and one-click updates from Blender
itself. The `CI` workflow runs ruff and pytest, then the headless tool suite on Blender 4.2 and 5.2.

## Constraints

- **Blender**
  - `src/__init__.py` imports its submodules inside `register()`, so the pure-Python parts (registry, models,
    PBR, agent loop) stay importable under plain pytest.
  - `preferences.py` is imported while Blender starts, so it keeps to light imports. Blender 4.2 evaluates
    property annotations without a comprehension scope, so the values they use live at module level.
  - Blender keeps only C pointers to dynamic `EnumProperty` items, so `state.py` holds the Python lists.
  - The render engine enum is filled by whichever engines are registered, so `set_render` and
    `render_preview` probe for the engine id by assigning it.
  - `invoke_props_dialog` has `title` and `confirm_text` only from 4.1; the dialog falls back without them.
  - `system.ui_scale` includes the OS DPI factor and is 0 only when there is no window (headless).
- **Viewport capture**
  - Headless Blender has a screen with a 3D View but no GPU context, so captures fail there with a clear
    error.
  - On Linux, Blender loads `libEGL.so.1` for any OpenGL work, even with `-b` (a Workbench or EEVEE render,
    offscreen drawing). Without it libepoxy aborts the process, so CI installs Mesa's EGL and software driver
    and runs the suite under Xvfb.
  - Offscreen view drawing runs the viewport's draw handlers, so the prompt bar hides itself while a capture
    is drawn.
  - Multi-angle renders use a temporary, unlinked camera to get Blender's own view and projection matrices.
- **Threads and the bridge**
  - A failing bridge callback is caught, so the timer that drains the queue keeps running.
  - `MCPManager.on_change` fires on the manager's thread, so redraws are posted to the main thread.
  - Feedback screenshots are best effort; a failed capture never stalls a run.
- **Providers**
  - The Agents SDK's tracing is switched off at start-up: it would upload runs to OpenAI's dashboard with the
    user's key.
  - OpenRouter lists only text-output models unless asked for image output too, which the image picker needs.
  - Fields that gateways may reject (beta headers, adaptive thinking) go only to Anthropic's own API.
  - When the Anthropic SDK cannot parse a tool input as JSON there is no `tool_use_id` to answer, so the
    turn is re-issued.
  - The image client remembers per API root and model whether the Images API or chat completions worked.
    Models without edits get the prompt alone.
- **MCP servers and dependencies**
  - Stdio MCP servers get a minimal safe environment (PATH, HOME, APPDATA…) plus their own `env`, as Claude
    Desktop does.
  - A broken `.pth` file must not stop the extension from loading.
- **Texture maps**
  - Normal maps use OpenGL's +Y up, while array rows grow downward.
  - Roughness derived from albedo is relative to the image's average, so a dark material is not simply rough.
  - PNGs use the Up filter, which compresses smooth maps better.

## Testing

| Layer | Command |
|---|---|
| Registry, schemas, model lists, Anthropic request shape | `pytest tests/test_registry.py tests/test_models.py` |
| Session and tool gateway with scripted agents on real threads | `pytest tests/test_loop.py` |
| History size estimate and compaction | `pytest tests/test_history.py` |
| Both SDKs, the image client and an MCP stdio server against local stubs | `pytest tests/test_http_stubs.py` |
| PBR maths, PNG encoding, text editing, layout and transcript | `pytest tests/test_pbr.py tests/test_ui_pure.py` |
| Every tool in headless Blender | `blender -b --factory-startup --python tests/run_tools.py` |
| The installed extension in a real window | `blender --online-mode --python tests/ui_smoke.py` |
| Clicks, dropdowns, focus, shadow and the approval dialog | `blender --enable-event-simulate --online-mode --python tests/ui_interactions.py` |

`tests/live_smoke.py` and `tests/live_texture.py` drive the installed extension with a configured provider and are not part of CI.
