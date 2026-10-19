# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Added

- Four specialist agents behind the prompt bar's tabs: Model Builder, Scene Builder, Animator and
  Material Generator. Each has its own conversation, instructions and toolset, and keeps `run_python`
  as a fallback. The instructions encode modelling, layout, animation and look-dev practice such as
  blockout and detail passes, real-world scale, naming, UV packing pitfalls and layered materials.
- Sub-agents through `spawn_agents`. Any agent can start specialists that work concurrently in the same
  scene and report back, up to two levels deep. Their steps are labelled in the log, and Stop halts all
  of them.
- Approval dialog for `run_python` when *Ask before running Python* is enabled: the requesting agent,
  the script's purpose and its code, with Yes and Cancel. The setting is also available in the ⚙ popover.
- Helper add-on installer. On first start, and from the preferences at any time, Blender AI can install
  and enable Rigify, MPFB, LoopTools and Extra Mesh Objects through Blender's extension system, one
  checkbox per package. The preferences list each helper's status.
- Update delivery through a Blender extension repository. Releases publish an `index.json` listing, so
  installing from the repository URL gives in-Blender update notifications and one-click updates.
- `scripts/devimport.py`, which lets tests and scripts import the extension as `blender_ai`.

### Changed

- Textures are packed into the `.blend` by default, including images imported with models or created by
  scripts. Images you linked externally are left untouched.
- Start-up is faster. Only the preferences and the bar's drawing load with Blender (about 70 ms); the
  rest loads in the background behind a "Loading Blender AI…" card.
- The model, image-model and reasoning pickers are dropdowns anchored to their pill. They stay open until
  a choice is made and support type-to-search, arrow keys and scrolling.
- The OpenAI-compatible agent executes a turn's tool calls in the order the model emitted them.
- The scene summary sent with each message includes the Blender version. Agent instructions cover the
  Blender 4.2+ API renames models most often get wrong (Principled BSDF sockets, auto smooth,
  `shadow_method`, boolean operators, the EEVEE engine id) and how to write rerunnable build scripts.
- Tools that accept a `collection` now create it when it does not exist.
- The extension source moved to `src/`.
- `scripts/build.py` produces a single `blender_ai-<version>.zip` (about 80 MB) containing the wheels
  for Windows, macOS and Linux. Blender installs only the ones that match the user's system, and the
  `--platform` option is removed.
- Source code carries no comments and uses 160-column lines under stricter lint rules. Rationale that
  used to live in comments is documented in `docs/architecture.md`.

### Fixed

- Pickers no longer disappear on mouse release.
- Removed the focus ring around the bar and the solid black line beneath it, which came from blending
  being disabled after text drawing.
- Sub-agents no longer receive `undo`, and the main agent's `undo` is refused while any sub-agent runs.
  Blender keeps one undo history per scene, so it could revert other agents' work.
- Installing helper add-ons no longer fails with "No installable packages marked" when text is present
  in the Get Extensions search field.
- `create_curve` and `create_text` with `convert_to_mesh` no longer read the evaluated depsgraph, which
  intermittently crashed Blender 4.2 in the headless suite. The resulting mesh is identical.
- Tools no longer leave a half-created object behind when the target collection is missing.
- The log file of a stdio MCP server is closed when the server stops.
- The headless CI job installs Mesa's EGL and runs under Xvfb. Blender previously aborted on Linux at the
  suite's first render for lack of `libEGL.so.1`, and the suite now prints each case and the Python stack
  if Blender crashes.

## [0.2.0] - Unreleased

### Added

- Support for any AI provider. Choose OpenAI-compatible (OpenAI, OpenRouter, Gemini, Groq, DeepSeek, xAI,
  Mistral, Together, Ollama, LM Studio and gateways) or Anthropic. The OpenAI-compatible agent runs on
  the OpenAI Agents SDK with tracing disabled. The Anthropic agent keeps streaming, thinking replay,
  prompt caching and refusal fallback. Both share one session, with tools executed on Blender's main
  thread.
- Model and image-model pickers populated from the provider's `/models` endpoint, searchable, with
  defaults chosen automatically. No model list is built in.
- A new prompt bar with a tab strip (3D Model, Scene, Animation, Material), a multi-line prompt, model,
  reasoning and image-model pickers, and a Generate button.
- `run_python` with unrestricted `bpy` access and an optional approval step.
- MCP servers over stdio or HTTP as additional tools, with a *Paste JSON config* import for the snippets
  services such as Meshy and Tripo publish.
- Tools: `scatter_on_surface` (geometry nodes), `add_physics`, `bake_physics`, `create_curve`,
  `create_text` and `import_from_url` (GLB, FBX, OBJ, USDZ and zip downloads, scaled and placed on the
  ground).
- Image generation through the OpenAI SDK, falling back to chat completions with image output for
  endpoints that lack an Images API, such as OpenRouter.

### Changed

- Settings are reduced to provider, URL (with a provider list), key and Test Connection. Everything else
  sits in collapsed sections.
- Every tool is available in every tab, and the system prompt steers the model to the most capable one.
- Switching tabs no longer resets the conversation.
- Roughness maps are derived from the base color relative to its average brightness.

## [0.1.0] - Unreleased

### Added

- Floating prompt bar at the bottom of every 3D Viewport, drawn with the active theme's colors, with a
  Model/Scene mode pill, model and reasoning pickers, image attach, a settings popover and Send/Stop.
- Agent loop on the Anthropic Messages API with streaming, viewport screenshot feedback, an append-only
  history that replays thinking blocks intact, refusal fallback on the first-party API, per-mode
  reasoning levels mapped to each model family, and a step limit.
- 89 typed tools covering scene and objects, bmesh modelling, modifiers, UVs, materials, PBR textures,
  lights, cameras, world, render previews and turntables, armatures and skinning, IK and constraints,
  shape keys and drivers, keyframes, F-curves, NLA, walk/run/idle cycles and camera moves.
- PBR textures: an OpenAI-compatible image model produces a tileable base color, and the normal,
  roughness, height, AO and metallic maps are derived locally with numpy and wired into a Principled BSDF.
- Helper add-on integrations for Rigify (bundled), MPFB, LoopTools and Extra Mesh Objects, used when
  installed.
- A `run_python` fallback gated by per-run approval.
- A sidebar tab with the full step log and token usage.
- Packaging as per-platform zips for Windows, macOS (Apple silicon and Intel) and Linux, for Blender 4.2
  LTS and later.

[0.3.0]: https://github.com/dev-swarup/Blender-AI/releases/tag/v0.3.0
