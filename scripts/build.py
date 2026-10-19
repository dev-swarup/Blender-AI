"""Build the installable extension zips with Blender's own builder.

    python scripts/build.py                          # every platform -> dist/
    python scripts/build.py --platform windows-x64   # one platform, for local testing
    python scripts/build.py --blender "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"

Steps: stage src/ into build/, write the `wheels` list for the bundled wheels into the
staged manifest, then run `blender --command extension build --split-platforms` and validate.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src"
STAGE = ROOT / "build" / "blender_ai"
DIST = ROOT / "dist"

sys.path.insert(0, str(ROOT / "scripts"))
from vendor_wheels import PLATFORM_TAGS

WINDOWS_SITE_PACKAGES_LEN = 110
WINDOWS_MAX_PATH = 259

WHEEL_PLATFORM_MARKERS = {
    "windows-x64": ("win_amd64",),
    "linux-x64": ("manylinux", "linux_x86_64"),
    "macos-arm64": ("arm64", "universal2"),
    "macos-x64": ("x86_64", "universal2"),
}


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


def wheel_matches(name: str, platforms: list[str]) -> bool:
    tag = name[:-4].split("-")[-1]
    if tag == "any":
        return True
    for platform in platforms:
        markers = WHEEL_PLATFORM_MARKERS[platform]
        if platform == "macos-x64" and "macosx" not in tag:
            continue
        if platform == "macos-arm64" and "macosx" not in tag:
            continue
        if any(m in tag for m in markers):
            return True
    return False


def stage(platforms: list[str]) -> list[str]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(SOURCE, STAGE, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.whl"))
    wheels = sorted(p for p in (SOURCE / "wheels").glob("*.whl") if wheel_matches(p.name, platforms))
    if not wheels:
        raise SystemExit("No wheels vendored; run scripts/vendor_wheels.py first.")
    (STAGE / "wheels").mkdir(exist_ok=True)
    for whl in wheels:
        shutil.copy2(whl, STAGE / "wheels" / whl.name)

    manifest = (STAGE / "blender_manifest.toml").read_text(encoding="utf-8")
    listing = "wheels = [\n" + "".join(f'  "./wheels/{w.name}",\n' for w in wheels) + "]\n"
    manifest = re.sub(r"(?m)^# The `wheels` list is written by scripts/build\.py.*\n", listing, manifest)
    manifest = re.sub(r"(?m)^platforms = \[.*\]$", "platforms = [" + ", ".join(f'"{p}"' for p in platforms) + "]", manifest)
    (STAGE / "blender_manifest.toml").write_text(manifest, encoding="utf-8")
    return [w.name for w in wheels]


def warn_long_paths(wheels: list[str]) -> None:
    longest, member = 0, ""
    for name in wheels:
        with zipfile.ZipFile(SOURCE / "wheels" / name) as zf:
            for item in zf.namelist():
                if len(item) > longest:
                    longest, member = len(item), item
    room = WINDOWS_MAX_PATH - WINDOWS_SITE_PACKAGES_LEN - longest
    if room < 25:
        print(f"WARNING: {member} is {longest} characters; Windows users with long profile paths may fail to install.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", choices=sorted(PLATFORM_TAGS), action="append")
    parser.add_argument("--blender")
    parser.add_argument("--vendor", action="store_true", help="download wheels first")
    args = parser.parse_args()

    platforms = args.platform or sorted(PLATFORM_TAGS)
    if args.vendor:
        cmd = [sys.executable, str(ROOT / "scripts" / "vendor_wheels.py")]
        for p in platforms:
            cmd += ["--platform", p]
        subprocess.run(cmd, check=True)

    blender = find_blender(args.blender)
    wheels = stage(platforms)
    print(f"Staged {len(wheels)} wheels for {', '.join(platforms)}")
    if "windows-x64" in platforms:
        warn_long_paths(wheels)
    DIST.mkdir(exist_ok=True)
    build = [blender, "--factory-startup", "--command", "extension", "build", "--source-dir", str(STAGE), "--output-dir", str(DIST)]
    if len(platforms) > 1 or args.platform:
        build.append("--split-platforms")
    subprocess.run(build, check=True)
    for zip_path in sorted(DIST.glob("blender_ai-*.zip")):
        subprocess.run([blender, "--factory-startup", "--command", "extension", "validate", str(zip_path)], check=True)
        print(f"Built {zip_path.relative_to(ROOT)} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
