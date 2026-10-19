from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

PACKAGE = "blender_ai"
SRC = Path(__file__).resolve().parent.parent / "src"


class SourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name: str, path: object = None, target: object = None) -> importlib.machinery.ModuleSpec | None:
        return importlib.util.spec_from_file_location(PACKAGE, SRC / "__init__.py", submodule_search_locations=[str(SRC)]) if name == PACKAGE else None


def install() -> None:
    if not any(isinstance(finder, SourceFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, SourceFinder())
