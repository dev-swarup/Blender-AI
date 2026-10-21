# Contributing to Blender AI

Thank you for helping improve Blender AI. Bug reports, new tools, helper add-on integrations,
documentation fixes and test coverage are all welcome.

- [Before you start](#before-you-start)
- [Development setup](#development-setup)
- [Running the checks](#running-the-checks)
- [Trying your changes in Blender](#trying-your-changes-in-blender)
- [Adding a tool](#adding-a-tool)
- [Code conventions](#code-conventions)
- [Commits and pull requests](#commits-and-pull-requests)
- [Reference](#reference)

## Before you start

For anything larger than a small fix, open an issue first so the approach can be agreed before you
invest time. Use the templates for [bug reports](.github/ISSUE_TEMPLATE/bug_report.yml) and
[feature requests](.github/ISSUE_TEMPLATE/feature_request.yml). Security problems do not belong in
public issues; follow [SECURITY.md](SECURITY.md).

[docs/architecture.md](docs/architecture.md) explains how the pieces fit together, including the
threading model and the Blender quirks the code works around. Read it before changing the agent loop,
the bridge or the prompt bar.

By contributing you agree that your work is released under the project's
[GPL-3.0-or-later license](LICENSE).

## Development setup

You need Python 3.11 or later and Blender 4.2 or later. CI runs the headless suite on Blender 4.2.0
and 5.2.0, and a change should work on both.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pytest ruff numpy uv -r scripts/requirements-vendor.txt
```

The requirements file pins the SDKs that the extension bundles, so the unit tests run against the
same versions users get. `uv` is only needed to vendor wheels for a build.

## Running the checks

CI runs the first three checks on every push and pull request. Run them before you push.

```bash
ruff check src tests scripts
ruff format --check src tests scripts
pytest
```

Pure-Python modules (the tool registry, model lists, the session, both agents, the MCP bridge, PBR
maths, text editing and layout) never import `bpy` and are covered by pytest.
`tests/test_http_stubs.py` runs the real OpenAI Agents SDK, the Anthropic SDK, the image client and an
MCP stdio server against local stubs.

Everything that touches Blender runs in the headless suite. It enables the extension straight from the
repository and calls every tool against a fresh scene:

```bash
BLENDER_USER_RESOURCES=/tmp/blender-ai-test blender -b --factory-startup --python-exit-code 1 --python tests/run_tools.py
```

- Set `BLENDER_AI_TESTS=mesh_editing,rigging` to run only some cases.
- Always point `BLENDER_USER_RESOURCES` at an empty folder when Blender AI is installed in your own
  profile. With `--factory-startup`, Blender treats installed extensions as disabled and removes their
  wheels, which breaks a running Blender session until it restarts.
- Each case prints `RUN` before it starts and `PASS` or `FAIL` when it ends. If Blender crashes, the
  Python stack is printed and the last `RUN` line names the case that took it down.
- On Linux, Blender creates its OpenGL context through EGL even in background mode, so a machine
  without a GPU or display needs Mesa's EGL and a virtual display, as CI uses:

  ```bash
  sudo apt-get install -y libegl1 libegl-mesa0 libgl1-mesa-dri libglx-mesa0 xvfb
  xvfb-run -a blender -b --factory-startup --python-exit-code 1 --python tests/run_tools.py
  ```

## Trying your changes in Blender

Build the extension and drag the zip onto a Blender window:

```bash
python scripts/vendor_wheels.py
python scripts/build.py
```

`dist/blender_ai-<version>.zip` is a single package for every platform. To build faster while
developing, vendor only your own platform with `--platform windows-x64` (or `macos-arm64`,
`macos-x64`, `linux-x64`). The manifest then lists only the platforms whose wheels are present, so that
zip installs only there.

To test without touching your own Blender configuration, start Blender with `BLENDER_USER_RESOURCES`
pointing at an empty folder. To try source changes in an installed copy without rebuilding:

```bash
python scripts/sync_dev.py <extensions folder>/user_default/blender_ai
```

## Adding a tool

Tools live in `src/tools/<group>.py`, or in `src/integrations/` when they depend on another add-on. A
tool is a function with a JSON-schema input and a docstring:

```python
from .common import describe, get_mesh_object
from .registry import number, obj, string, tool


@tool(obj({"name": string("Mesh object."), "angle_deg": number("Split angle.", minimum=0, maximum=180, default=30)}, ["name"]), group="mesh")
def split_sharp_edges(name: str, angle_deg: float = 30.0) -> dict:
    """Split edges sharper than an angle so flat shading stays crisp. Call this after modelling
    hard-surface parts that will be shaded smooth."""
    ob = get_mesh_object(name)
    ...
    return describe(ob)
```

- **The docstring is the description the model reads.** Say what the tool does and when to call it.
- **Schemas are closed.** Those built with `obj()` set `additionalProperties: false`, so a misspelled
  argument comes back to the model as an error. Use the helpers in `tools/registry.py` (`vec3()`,
  `color()`, `names()`, `integer()`, `boolean()`) and `common.SELECTION` for element selections.
- **Fail with `ToolError`.** Raise it with a message the model can act on: what was wrong and what is
  valid. Unexpected exceptions are returned to the model too, but they read worse.
- **Prefer data APIs.** Use `bpy.data` and `bmesh` over `bpy.ops`. When an operator is unavoidable, wrap
  it in `common.op_context()` or `common.edit_mode()`.
- **Choose the group deliberately.** The group decides which agents receive the tool, and each role in
  `agent/roles.py` lists its groups. A tool in a new group must be added to at least one role;
  `tests/run_tools.py` fails when a tool reaches no agent. Prefer tools that do a whole job in one call,
  which is what the agents' instructions steer them towards.
- **Declare behavior with flags.** `destructive=True` marks the step in the log; `mutates=False` skips
  the undo step and the viewport feedback; `kind=ToolKind.ASYNC` returns an `AsyncJob` whose `work` runs
  off the main thread and must not touch `bpy`; `requires="mpfb"` offers the tool only while that helper
  add-on is usable.
- **Test and document it.** Add a case to `tests/run_tools.py`, then regenerate the reference:

  ```bash
  blender -b --factory-startup --python scripts/gen_tool_docs.py
  ```

## Code conventions

- Python 3.11 syntax, because Blender 4.2 ships 3.11. Use `from __future__ import annotations` and full
  type hints.
- No comments. Names and small functions carry the meaning. A rule the code cannot show, such as a Blender
  quirk or a threading constraint, belongs in [docs/architecture.md](docs/architecture.md#constraints).
- Docstrings only where the code reads them: tool functions (the description the model sees), the demo
  MCP server's tools and `scripts/build.py` (its `--help`). Blender operators, panels and menus set
  `bl_description`.
- Lines up to 160 characters. `ruff format` ignores magic trailing commas, so anything that fits goes on
  one line; prefer a conditional expression or a comprehension over a short `if`/`else` or an append loop.
  No lint pragmas: per-file exceptions live in `pyproject.toml`.
- Never call `bpy` from a worker thread. Workers post callables to `state.get().bridge`.
- Mutable module state lives only in `src/state.py`.
- Code in `src/` uses relative imports. Tests and scripts import it as `blender_ai`, which
  `scripts/devimport.py` maps to `src/`.

## Commits and pull requests

1. Fork the repository and branch from `main`.
2. Keep each commit focused, with a [Conventional Commits](https://www.conventionalcommits.org/)
   message such as `feat: add split_sharp_edges`, `fix: close the MCP log file` or `docs: update the
   README`.
3. Run the checks above, and `run_tools.py` for any change to a tool.
4. Update [CHANGELOG.md](CHANGELOG.md) under the next version for user-visible changes.
5. Open a pull request and complete the [template](.github/pull_request_template.md). For UI changes,
   attach screenshots in a dark and a light theme.

## Reference

| Command | Purpose |
|---|---|
| `blender -b --factory-startup --python-exit-code 1 --python tests/run_tools.py` | Calls every tool against a fresh scene and asserts on `bpy.data`. Tools that need a visible 3D Viewport must fail with a clear `ToolError` here. |
| `blender --online-mode --window-geometry 0 0 1600 900 --python tests/ui_smoke.py` | Runs the installed extension in a real window against a local stub of an OpenAI-compatible API and saves screenshots and `ui_smoke.json` to `BLENDER_AI_SHOTS` (default `docs/screenshots`). |
| `blender --enable-event-simulate --online-mode --window-geometry 0 0 1600 900 --python tests/ui_interactions.py` | Drives the bar with simulated mouse and keyboard events and checks the dropdowns, focus outline, shadow and approval dialog. `--enable-event-simulate` makes Blender ignore your real mouse and keyboard. |
| `blender --online-mode --window-geometry 0 0 1600 900 --python tests/live_smoke.py` | Sends one request to a real provider; see the variables below. |
| `blender --online-mode --window-geometry 0 0 1600 900 --python tests/live_texture.py` | Makes exactly one `generate_texture` call to a real image model, so a run costs one image. Uses `BLENDER_AI_LIVE_URL`, `BLENDER_AI_LIVE_KEY_FILE` and `BLENDER_AI_LIVE_IMAGE_MODEL`. |
| `blender -b --factory-startup --python scripts/gen_tool_docs.py` | Regenerates `docs/tools.md` from the live registry. |
| `python scripts/build.py [--vendor] [--blender PATH] [--download-url URL]` | Stages `src/`, lists the vendored wheels and the platforms they cover in the manifest, builds and validates the zip with Blender, and with a download URL writes `dist/index.json`. |
| `python scripts/vendor_wheels.py [--platform P] [--clean]` | Downloads the bundled wheels for each platform and Blender Python version into `src/wheels/`. Needs `uv`, which resolves markers per target platform. |
| `python scripts/make_icons.py` | Draws the monochrome icons into `src/icons/`. They must stay pure white with alpha, because the overlay tints them with the theme's text colour. |
| `python scripts/sync_dev.py <installed blender_ai folder>` | Copies the source over an installed copy. |

`tests/live_smoke.py` reads its settings from the environment and never prints or saves the key:

| Variable | Meaning |
|---|---|
| `BLENDER_AI_LIVE_PROVIDER` | `OPENAI` (default) or `ANTHROPIC` |
| `BLENDER_AI_LIVE_URL` | API URL, for example `https://openrouter.ai/api/v1` |
| `BLENDER_AI_LIVE_KEY_FILE` | File whose only line is the API key |
| `BLENDER_AI_LIVE_MODEL` | Model id (default: the extension's pick) |
| `BLENDER_AI_LIVE_REQUIRE_FREE` | Any value: refuse to run unless the model is listed as free |
| `BLENDER_AI_LIVE_TAB` | `MODEL`, `SCENE`, `ANIMATE` or `MATERIAL` |
| `BLENDER_AI_LIVE_PROMPT` | The request to send |
| `BLENDER_AI_LIVE_EXPECT` | Objects or collections that must exist afterwards, comma-separated |
| `BLENDER_AI_LIVE_REASONING` | `auto`, `low`, `medium` or `high` |
| `BLENDER_AI_LIVE_TIMEOUT` | Seconds before the run is stopped (default 300) |
| `BLENDER_AI_LIVE_IMAGE_MODEL` | Image model; empty disables image generation for the run |
| `BLENDER_AI_SHOTS` | Output folder for `live_smoke.json`, `live_log.txt` and screenshots |
