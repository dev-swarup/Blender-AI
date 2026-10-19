from __future__ import annotations

import os
import site
import sys

_done = False


def prepare() -> None:
    global _done
    if _done:
        return
    _done = True
    if sys.platform != "win32":
        return
    for entry in list(sys.path):
        if entry and os.path.isfile(os.path.join(entry, "pywin32.pth")):
            try:
                site.addsitedir(entry)
            except Exception as exc:
                print(f"Blender AI: couldn't initialise pywin32 in {entry}: {exc}", file=sys.stderr)
