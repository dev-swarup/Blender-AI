from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELS = ROOT / "src" / "wheels"
REQUIREMENTS = ROOT / "scripts" / "requirements-vendor.txt"

PYTHON_VERSIONS = ("3.11", "3.12", "3.13")

SKIP = frozenset({"requests", "urllib3", "certifi", "idna", "charset_normalizer", "numpy", "websockets"})

PLATFORM_TAGS: dict[str, tuple[str, ...]] = {
    "windows-x64": ("win_amd64",),
    "linux-x64": ("manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"),
    "macos-arm64": ("macosx_11_0_arm64", "macosx_12_0_arm64", "macosx_13_0_arm64", "macosx_14_0_arm64", "macosx_10_9_universal2", "macosx_11_0_universal2"),
    "macos-x64": ("macosx_10_9_x86_64", "macosx_10_12_x86_64", "macosx_10_13_x86_64", "macosx_11_0_x86_64", "macosx_10_9_universal2", "macosx_11_0_universal2"),
}


UV_PLATFORMS = {
    "windows-x64": "x86_64-pc-windows-msvc",
    "linux-x64": "x86_64-manylinux_2_28",
    "macos-arm64": "aarch64-apple-darwin",
    "macos-x64": "x86_64-apple-darwin",
}


def resolve(platform: str, python: str, dest: Path) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required for cross-platform resolution: pip install uv")
    locked = dest / "locked.txt"
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [uv, "pip", "compile", str(REQUIREMENTS), "--quiet", "--no-header", "--no-annotate"]
    cmd += ["--python-version", python, "--python-platform", UV_PLATFORMS[platform], "-o", str(locked)]
    subprocess.run(cmd, check=True)
    pins = [line.strip() for line in locked.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    kept = [pin for pin in pins if pin.split("==")[0].lower().replace("-", "_") not in SKIP]
    locked.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return locked


def download(platform: str, python: str, dest: Path) -> None:
    locked = resolve(platform, python, dest)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--quiet",
        "--no-deps",
        "--only-binary=:all:",
        "--implementation",
        "cp",
        "--python-version",
        python,
        "--dest",
        str(dest),
        "-r",
        str(locked),
    ]
    for tag in PLATFORM_TAGS[platform]:
        cmd += ["--platform", tag]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", choices=sorted(PLATFORM_TAGS), action="append", help="limit to these platforms")
    parser.add_argument("--clean", action="store_true", help="delete existing wheels first")
    args = parser.parse_args()

    platforms = args.platform or sorted(PLATFORM_TAGS)
    if args.clean and WHEELS.exists():
        for whl in WHEELS.glob("*.whl"):
            whl.unlink()
    WHEELS.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for platform in platforms:
            for python in PYTHON_VERSIONS:
                target = Path(tmp) / f"{platform}-{python}"
                print(f"{platform} / Python {python}")
                download(platform, python, target)
                for whl in target.glob("*.whl"):
                    if not (WHEELS / whl.name).exists():
                        shutil.copy2(whl, WHEELS / whl.name)
    names = sorted(p.name for p in WHEELS.glob("*.whl"))
    print(f"{len(names)} wheels in {WHEELS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
