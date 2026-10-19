from __future__ import annotations

import contextlib
import io
import math
import random
import traceback
from typing import Any

import bmesh
import bpy
import mathutils

from .registry import ToolError, ToolKind, obj, string, tool

MAX_OUTPUT = 6000


def _namespace() -> dict[str, Any]:
    import numpy

    return {
        "__name__": "__blender_ai__",
        "bpy": bpy,
        "bmesh": bmesh,
        "mathutils": mathutils,
        "Vector": mathutils.Vector,
        "Matrix": mathutils.Matrix,
        "Euler": mathutils.Euler,
        "Quaternion": mathutils.Quaternion,
        "math": math,
        "random": random,
        "np": numpy,
        "C": bpy.context,
        "D": bpy.data,
        "result": None,
    }


def _trimmed_traceback(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    own = [f for f in frames if f.filename == "<run_python>"]
    lines = [f"  line {f.lineno}: {f.line or ''}".rstrip() for f in own[-6:]]
    return "\n".join([*lines, f"{type(exc).__name__}: {exc}"])


@tool(
    obj(
        {
            "code": string(
                "Python to run inside Blender with full access, like the Python console. Pre-imported: bpy, bmesh, "
                "mathutils (Vector, Matrix, Euler, Quaternion), math, random, numpy as np; C = bpy.context, "
                "D = bpy.data. Import anything else you need. print() output is returned; assign to `result` "
                "to return a value."
            ),
            "purpose": string("One short sentence on what the script does (shown to the user)."),
        },
        ["code", "purpose"],
    ),
    group="meta",
    kind=ToolKind.APPROVAL,
)
def run_python(code: str, purpose: str) -> str:
    """Run Python inside Blender with full bpy access. Use it whenever it's the better tool: geometry
    nodes, complex procedural geometry, bulk edits across many objects, drivers with custom expressions,
    particles, physics settings, custom node trees, or any property no other tool exposes. Prefer one
    well-structured script over many tiny tool calls, keep it idempotent where you can, and print what you
    need to verify. Each run is one undo step."""
    try:
        compiled = compile(code, "<run_python>", "exec")
    except SyntaxError as exc:
        raise ToolError(f"Syntax error on line {exc.lineno}: {exc.msg}") from exc
    namespace = _namespace()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compiled, namespace)
    except Exception as exc:
        printed = out.getvalue().strip()
        prefix = f"Output before the error:\n{printed[-2000:]}\n\n" if printed else ""
        raise ToolError(f"{prefix}The script raised:\n{_trimmed_traceback(exc)}") from exc
    printed = out.getvalue()
    result = namespace.get("result")
    text = (printed + ("\nresult = " + repr(result) if result is not None else "")).strip() or "ran without output"
    return text[:MAX_OUTPUT] + ("\n… (output truncated)" if len(text) > MAX_OUTPUT else "")
