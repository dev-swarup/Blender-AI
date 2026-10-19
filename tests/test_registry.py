from __future__ import annotations

import pytest

from blender_ai.tools import registry
from blender_ai.tools.registry import ToolError, ToolOutput, ToolRegistry, array, color, integer, number, obj, string, validate, vec3


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(obj({"name": string("n")}, ["name"]), group="objects")
    def make_thing(name: str) -> dict:
        """Make a thing with a name. Long enough description."""
        return {"made": name}

    @reg.tool(obj({}), group="mesh")
    def mesh_only() -> str:
        """Only in the mesh group, for testing filtering."""
        return "ok"

    @reg.tool(obj({}), group="integrations", requires="helper")
    def needs_helper() -> None:
        """Needs an optional helper add-on."""

    return reg


def test_schema_helpers_are_closed_objects() -> None:
    schema = obj({"a": number(minimum=0)}, ["a"])
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["a"]


@pytest.mark.parametrize(
    ("schema", "value", "ok"),
    [
        (vec3(), [1, 2, 3], True),
        (vec3(), [1, 2], False),
        (vec3(), [1, "2", 3], False),
        (color(), [0.5, 0.5, 0.5], True),
        (color(), [0.5, 0.5, 1.5], False),
        (integer(minimum=1), 2.0, True),
        (integer(minimum=1), 2.5, False),
        (integer(), True, False),
        (number(), True, False),
        (string(enum=["a", "b"]), "c", False),
        (array(string(), min_items=1), [], False),
        (obj({"x": number()}), {"x": 1, "y": 2}, False),
        (obj({"x": number()}, ["x"]), {}, False),
        ({"type": ["number", "string"]}, "s", True),
        ({"anyOf": [string(), number()]}, None, False),
    ],
)
def test_validate(schema: dict, value: object, ok: bool) -> None:
    assert (validate(schema, value) == []) is ok


def test_unknown_property_lists_known_ones() -> None:
    errors = validate(obj({"width": number(), "segments": integer()}), {"widht": 1})
    assert "widht" in errors[0] and "segments, width" in errors[0]


def test_call_validates_and_normalizes() -> None:
    reg = make_registry()
    spec = reg.get("make_thing")
    out = registry.call(spec, {"name": "box"})
    assert isinstance(out, ToolOutput) and '"made": "box"' in out.text
    with pytest.raises(ToolError, match="missing required property"):
        registry.call(spec, {})
    with pytest.raises(ToolError, match="JSON object"):
        registry.call(spec, ["nope"])


def test_mode_and_requirement_filtering_is_ordered() -> None:
    reg = make_registry()
    assert [s.name for s in reg.specs(["objects", "integrations"])] == ["make_thing", "needs_helper"]
    assert [s.name for s in reg.specs(None, lambda r: False)] == ["make_thing", "mesh_only"]
    assert [s.name for s in reg.specs(None, lambda r: True)] == ["make_thing", "mesh_only", "needs_helper"]


def test_docstring_is_required_and_names_unique() -> None:
    reg = make_registry()
    with pytest.raises(ValueError, match="docstring"):

        @reg.tool(obj({}), group="scene")
        def undocumented() -> None: ...

    with pytest.raises(ValueError, match="duplicate"):

        @reg.tool(obj({}), group="scene", name="make_thing")
        def other() -> None:
            """A different function claiming an existing name."""


def test_to_api_and_content_blocks() -> None:
    spec = make_registry().get("make_thing")
    assert spec.to_api() == {"name": "make_thing", "description": spec.description, "input_schema": spec.input_schema}
    assert spec.to_api(eager_input_streaming=True)["eager_input_streaming"] is True
    blocks = registry.to_content(ToolOutput("done", [registry.ImagePart(b"\x89PNG")]))
    assert blocks[0] == {"type": "text", "text": "done"}
    assert blocks[1]["source"]["media_type"] == "image/png"
