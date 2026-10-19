"""Build the installable extension zip with Blender's own builder: one zip for every platform.

    python scripts/build.py              # -> dist/blender_ai-<version>.zip
    python scripts/build.py --vendor     # download the wheels first
    python scripts/build.py --download-url https://github.com/<owner>/<repo>/releases/download/v<version>
    python scripts/build.py --blender "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"

Steps: stage src/ into build/, list every bundled wheel in the staged manifest (with the platforms those
wheels cover), then run `blender --command extension build` and validate. On install, Blender extracts
only the wheels that match the user's platform and Python version.

With --download-url it also writes dist/index.json, the listing Blender reads from a remote repository to
offer updates; the zip is expected to be published at that URL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from vendor_wheels import PLATFORM_TAGS

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src"
STAGE = ROOT / "build" / "blender_ai"
DIST = ROOT / "dist"

WINDOWS_SITE_PACKAGES_LEN = 110
WINDOWS_MAX_PATH = 259


def find_blender(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("BLENDER")]
    if sys.platform == "win32":
        base = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Blender Foundation"
        candidates += [str(p / "blender.exe") for p in sorted(base.glob("Blender *"), reverse=True)]
    elif sys.platform == "darwin":
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    candidates.append(shutil.which("blender"))
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("Blender not found; pass --blender or set BLENDER.")


def covered_platforms(wheels: list[str]) -> list[str]:
    tags = [name[:-4].split("-")[-1] for name in wheels if name.startswith("pydantic_core-")]
    return [platform for platform, markers in PLATFORM_TAGS.items() if any(m in tag for tag in tags for m in markers)]


def stage() -> tuple[list[str], list[str]]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(SOURCE, STAGE, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    wheels = sorted(p.name for p in (STAGE / "wheels").glob("*.whl"))
    platforms = covered_platforms(wheels)
    if not platforms:
        raise SystemExit("No wheels vendored for a supported platform; run scripts/vendor_wheels.py first.")
    listing = "".join(f'  "./wheels/{name}",\n' for name in wheels)
    block = "platforms = [" + ", ".join(f'"{p}"' for p in platforms) + "]\nwheels = [\n" + listing + "]\n"
    manifest = (STAGE / "blender_manifest.toml").read_text(encoding="utf-8")
    manifest = re.sub(r"(?m)^platforms = \[.*\]\n", lambda _match: block, manifest, count=1)
    (STAGE / "blender_manifest.toml").write_text(manifest, encoding="utf-8")
    return wheels, platforms


def warn_long_paths(wheels: list[str]) -> None:
    longest, member = 0, ""
    for name in wheels:
        with zipfile.ZipFile(STAGE / "wheels" / name) as zf:
            for item in zf.namelist():
                if len(item) > longest:
                    longest, member = len(item), item
    if WINDOWS_MAX_PATH - WINDOWS_SITE_PACKAGES_LEN - longest < 25:
        print(f"WARNING: {member} is {longest} characters; Windows users with long profile paths may fail to install.")


def write_index(blender: str, zip_path: Path, download_url: str) -> Path:
    subprocess.run([blender, "--factory-startup", "--command", "extension", "server-generate", "--repo-dir", str(DIST)], check=True)
    index = DIST / "index.json"
    listing = json.loads(index.read_text(encoding="utf-8"))
    for entry in listing["data"]:
        entry["archive_url"] = f"{download_url.rstrip('/')}/{zip_path.name}"
    index.write_text(json.dumps(listing, indent=2) + "\n", encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blender")
    parser.add_argument("--vendor", action="store_true", help="download wheels first")
    parser.add_argument("--download-url", help="folder URL the zip will be published at; also writes dist/index.json")
    args = parser.parse_args()
    if args.vendor:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "vendor_wheels.py")], check=True)
    blender = find_blender(args.blender)
    wheels, platforms = stage()
    print(f"Staged {len(wheels)} wheels for {', '.join(platforms)}")
    warn_long_paths(wheels)
    DIST.mkdir(exist_ok=True)
    for old in (*DIST.glob("blender_ai-*.zip"), DIST / "index.json"):
        old.unlink(missing_ok=True)
    subprocess.run([blender, "--factory-startup", "--command", "extension", "build", "--source-dir", str(STAGE), "--output-dir", str(DIST)], check=True)
    zip_path = next(DIST.glob("blender_ai-*.zip"))
    subprocess.run([blender, "--factory-startup", "--command", "extension", "validate", str(zip_path)], check=True)
    print(f"Built {zip_path.relative_to(ROOT)} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    if args.download_url:
        print(f"Wrote {write_index(blender, zip_path, args.download_url).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
