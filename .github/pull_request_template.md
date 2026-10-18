## What this changes

Summarize the change in one or two sentences and link the issue, if any.

## How it was tested

- [ ] `ruff check` and `ruff format --check` pass
- [ ] `pytest` passes
- [ ] `blender -b --factory-startup --python tests/run_tools.py` passes (required for tool changes)
- [ ] Verified in Blender (UI changes: attach screenshots in a dark and a light theme)

## Checklist

- [ ] New tools have a docstring stating when the model should call them, a closed input schema, and a case in `tests/run_tools.py`
- [ ] `docs/tools.md` regenerated if tools changed (`blender -b --python scripts/gen_tool_docs.py`)
- [ ] CHANGELOG updated for user-visible changes
