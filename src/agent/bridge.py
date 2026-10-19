from __future__ import annotations

import queue
import sys
import traceback
from collections.abc import Callable
from typing import Any


class Bridge:
    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[tuple[Callable[..., Any], tuple[Any, ...]]] = queue.SimpleQueue()

    def post(self, fn: Callable[..., Any], *args: Any) -> None:
        self._queue.put((fn, args))

    def drain(self, limit: int = 1000) -> int:
        ran = 0
        while ran < limit:
            try:
                fn, args = self._queue.get_nowait()
            except queue.Empty:
                break
            ran += 1
            try:
                fn(*args)
            except Exception:
                traceback.print_exc(file=sys.stderr)
        return ran

    def empty(self) -> bool:
        return self._queue.empty()
