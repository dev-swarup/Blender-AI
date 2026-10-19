from __future__ import annotations

import base64
import enum
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


class Mode(enum.Enum):
    MODEL = "MODEL"
    SCENE = "SCENE"
    ANIMATE = "ANIMATE"
    MATERIAL = "MATERIAL"


class ToolKind(enum.Enum):
    NORMAL = "normal"
    ASYNC = "async"
    APPROVAL = "approval"
    QUESTION = "question"
    AGENT = "agent"


GROUP_ORDER = (
    "meta",
    "agents",
    "scene",
    "objects",
    "mesh",
    "modifiers",
    "uv",
    "materials",
    "textures",
    "lights_camera",
    "world",
    "render",
    "rigging",
    "animation",
    "procedural",
    "physics",
    "integrations",
    "mcp",
)


class ToolError(Exception): ...


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/png"


@dataclass
class ToolOutput:
    text: str
    images: list[ImagePart] = field(default_factory=list)


@dataclass(frozen=True)
class AsyncJob:
    label: str
    work: Callable[[], Any]
    finish: Callable[[Any], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., Any]
    group: str
    kind: ToolKind = ToolKind.NORMAL
    destructive: bool = False
    mutates: bool = True
    requires: str | None = None

    def to_api(self, *, eager_input_streaming: bool = False) -> dict[str, Any]:
        param: dict[str, Any] = {"name": self.name, "description": self.description, "input_schema": self.input_schema}
        if eager_input_streaming:
            param["eager_input_streaming"] = True
        return param


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def tool(
        self,
        schema: dict[str, Any],
        *,
        group: str,
        name: str | None = None,
        kind: ToolKind = ToolKind.NORMAL,
        destructive: bool = False,
        mutates: bool = True,
        requires: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            import inspect

            doc = inspect.getdoc(func)
            if not doc:
                raise ValueError(f"tool {func.__name__} needs a docstring; it is the model-facing description")
            self.add(
                ToolSpec(
                    name=name or func.__name__,
                    description=doc,
                    input_schema=schema,
                    func=func,
                    group=group,
                    kind=kind,
                    destructive=destructive,
                    mutates=mutates,
                    requires=requires,
                )
            )
            return func

        return decorate

    def add(self, spec: ToolSpec) -> None:
        if spec.group not in GROUP_ORDER:
            raise ValueError(f"unknown tool group {spec.group!r}")
        existing = self._specs.get(spec.name)
        if existing is not None and _origin(existing.func) != _origin(spec.func):
            raise ValueError(f"duplicate tool name {spec.name!r}")
        if spec.input_schema.get("type") != "object":
            raise ValueError(f"tool {spec.name}: input_schema must be an object schema")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def specs(self, groups: Iterable[str] | None = None, available: Callable[[str], bool] | None = None) -> list[ToolSpec]:
        wanted = None if groups is None else set(groups)
        chosen = [
            s for s in self._specs.values() if (wanted is None or s.group in wanted) and (s.requires is None or available is None or available(s.requires))
        ]
        return sorted(chosen, key=lambda s: (GROUP_ORDER.index(s.group), s.name))

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self.specs())

    def __len__(self) -> int:
        return len(self._specs)


def _origin(func: Callable[..., Any]) -> str:
    return f"{func.__module__}.{func.__qualname__}"


REGISTRY = ToolRegistry()
tool = REGISTRY.tool


def obj(properties: Mapping[str, dict[str, Any]], required: Iterable[str] = (), *, description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": dict(properties), "additionalProperties": False}
    req = list(required)
    if req:
        schema["required"] = req
    if description:
        schema["description"] = description
    return schema


def _with(schema: dict[str, Any], description: str | None, default: Any) -> dict[str, Any]:
    if description:
        schema["description"] = description
    if default is not None:
        schema["default"] = default
    return schema


def string(description: str | None = None, *, enum: Iterable[str] | None = None, default: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if enum is not None:
        schema["enum"] = list(enum)
    return _with(schema, description, default)


def number(description: str | None = None, *, minimum: float | None = None, maximum: float | None = None, default: float | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return _with(schema, description, default)


def integer(description: str | None = None, *, minimum: int | None = None, maximum: int | None = None, default: int | None = None) -> dict[str, Any]:
    schema = number(description, minimum=minimum, maximum=maximum, default=default)
    schema["type"] = "integer"
    return schema


def boolean(description: str | None = None, *, default: bool | None = None) -> dict[str, Any]:
    return _with({"type": "boolean"}, description, default)


def array(items: dict[str, Any], description: str | None = None, *, min_items: int | None = None, max_items: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": items}
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    return _with(schema, description, None)


def vec3(description: str | None = None) -> dict[str, Any]:
    return array({"type": "number"}, description, min_items=3, max_items=3)


def color(description: str | None = None, *, alpha: bool = False) -> dict[str, Any]:
    n = 4 if alpha else 3
    return array({"type": "number", "minimum": 0.0, "maximum": 1.0}, description, min_items=n, max_items=n)


def names(description: str | None = None) -> dict[str, Any]:
    return array({"type": "string"}, description, min_items=1)


def scalar(description: str | None = None) -> dict[str, Any]:
    vector = {"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}}
    return _with({"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "boolean"}, vector]}, description, None)


def free_object(description: str | None = None) -> dict[str, Any]:
    return _with({"type": "object", "additionalProperties": scalar()}, description, None)


_JSON_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
}


def validate(schema: Mapping[str, Any], value: Any, path: str = "input") -> list[str]:
    errors: list[str] = []
    _validate(schema, value, path, errors)
    return errors


def _validate(schema: Mapping[str, Any], value: Any, path: str, errors: list[str]) -> None:
    if "anyOf" in schema:
        if not any(not validate(sub, value, path) for sub in schema["anyOf"]):
            errors.append(f"{path}: does not match any allowed form")
        return

    types = schema.get("type")
    if types is not None:
        allowed = [types] if isinstance(types, str) else list(types)
        if not any(_JSON_TYPES[t](value) for t in allowed):
            errors.append(f"{path}: expected {' or '.join(allowed)}, got {type(value).__name__}")
            return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")

    if _JSON_TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: allows at most {schema['maxItems']} items")
        items = schema.get("items")
        if items:
            for i, item in enumerate(value):
                _validate(items, item, f"{path}[{i}]", errors)

    if isinstance(value, dict):
        props: Mapping[str, Any] = schema.get("properties", {})
        errors.extend(f"{path}: missing required property {key!r}" for key in schema.get("required", ()) if key not in value)
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                _validate(props[key], item, f"{path}.{key}", errors)
            elif extra is False:
                known = ", ".join(sorted(props)) or "none"
                errors.append(f"{path}: unknown property {key!r} (known: {known})")
            elif isinstance(extra, dict):
                _validate(extra, item, f"{path}.{key}", errors)


def call(spec: ToolSpec, args: Any) -> ToolOutput | AsyncJob:
    if not isinstance(args, dict):
        raise ToolError(f"{spec.name}: tool input must be a JSON object")
    errors = validate(spec.input_schema, args)
    if errors:
        raise ToolError("Invalid input:\n" + "\n".join(f"- {e}" for e in errors[:12]))
    result = spec.func(**args)
    return normalize(result)


def normalize(result: Any) -> ToolOutput | AsyncJob:
    if isinstance(result, (ToolOutput, AsyncJob)):
        return result
    if result is None:
        return ToolOutput("ok")
    return ToolOutput(result) if isinstance(result, str) else ToolOutput(json.dumps(result, ensure_ascii=False, default=_json_default))


def _json_default(value: Any) -> Any:
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        return [round(v, 5) if isinstance(v, float) else v for v in value]
    return str(value)


def to_content(output: ToolOutput) -> list[dict[str, Any]]:
    images = [{"type": "base64", "media_type": i.media_type, "data": base64.standard_b64encode(i.data).decode("ascii")} for i in output.images]
    return [{"type": "text", "text": output.text or "ok"}, *({"type": "image", "source": source} for source in images)]
