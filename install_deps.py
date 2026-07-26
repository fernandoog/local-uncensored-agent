#!/usr/bin/env python3
"""
Install llama-cpp-python with a prebuilt wheel when possible.

Python 3.13 has no official CUDA wheels from abetlen → pip falls back to
compiling from source (slow / often CPU-only). This script:
  1) refuses CUDA install on 3.13+ unless --force-source
  2) prefers cu121 / cpu binary wheels with --only-binary=:all:
  3) prints exact commands to recreate a 3.12 venv on Windows
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CUDA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu121"
CPU_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
# Also try cu122/cu124 indexes if cu121 has no match
CUDA_INDEXES = (
    CUDA_INDEX,
    "https://abetlen.github.io/llama-cpp-python/whl/cu122",
    "https://abetlen.github.io/llama-cpp-python/whl/cu124",
)


def _py_tag() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _supports_prebuilt_cuda_wheels() -> bool:
    # abetlen CUDA wheels historically target cp310–cp312, not cp313+
    return sys.version_info < (3, 13)


def _pip(*args: str, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd, env=env or os.environ.copy())


def _try_pip(*args: str, env: dict[str, str] | None = None) -> bool:
    cmd = [sys.executable, "-m", "pip", *args]
    print("[try]", " ".join(cmd))
    return subprocess.call(cmd, env=env or os.environ.copy()) == 0


def _install_rest() -> None:
    req = ROOT / "requirements.txt"
    if not req.exists():
        return
    lines = [
        ln
        for ln in req.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("llama-cpp-python") and not ln.strip().startswith("#")
    ]
    if lines:
        _pip("install", *lines)


def _windows_py312_hint() -> str:
    return (
        "\n[fix] CUDA wheels need Python 3.11 or 3.12 (not 3.13).\n"
        "  winget install -e --id Python.Python.3.12\n"
        "  py -3.12 -m venv .venv\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python install_deps.py\n"
    )


def install_cuda(*, allow_source: bool) -> None:
    if not _supports_prebuilt_cuda_wheels() and not allow_source:
        print(
            f"[error] Python {_py_tag()} detected. No prebuilt CUDA wheel for this version; "
            "pip would compile from source (CPU / broken CUDA)."
        )
        if platform.system() == "Windows":
            print(_windows_py312_hint())
        else:
            print(
                "\n[fix] Create a 3.12 venv:\n"
                "  python3.12 -m venv .venv && source .venv/bin/activate\n"
                "  python install_deps.py\n"
            )
        raise SystemExit(2)

    # Prefer binary-only so pip never starts a long source build by accident
    for index in CUDA_INDEXES:
        if _try_pip(
            "install",
            "llama-cpp-python",
            "--upgrade",
            "--force-reinstall",
            "--only-binary=:all:",
            "--extra-index-url",
            index,
        ):
            print(f"[ok] CUDA wheel from {index}")
            return

    if allow_source:
        print("[warn] No binary CUDA wheel; compiling from source (CMAKE_ARGS=GGML_CUDA)")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        _pip("install", "llama-cpp-python", "--force-reinstall", "--no-cache-dir", env=env)
        return

    print("[error] No prebuilt CUDA wheel found. Refusing source build.")
    print("        Re-run with --force-source to compile, or use Python 3.12.")
    raise SystemExit(3)


def install_cpu(*, allow_source: bool) -> None:
    if _try_pip(
        "install",
        "llama-cpp-python",
        "--upgrade",
        "--force-reinstall",
        "--only-binary=:all:",
        "--extra-index-url",
        CPU_INDEX,
    ):
        print(f"[ok] CPU wheel from {CPU_INDEX}")
        return
    if _try_pip(
        "install",
        "llama-cpp-python",
        "--upgrade",
        "--force-reinstall",
        "--only-binary=:all:",
    ):
        print("[ok] CPU wheel from PyPI")
        return
    if allow_source:
        _pip("install", "llama-cpp-python", "--force-reinstall", "--no-cache-dir")
        return
    print("[error] No CPU binary wheel. Use Python 3.11/3.12 or --force-source.")
    raise SystemExit(3)


def install_metal(*, allow_source: bool) -> None:
    # Metal usually needs a local build on macOS
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_METAL=on"
    if not allow_source and not _supports_prebuilt_cuda_wheels():
        # still allow metal build on 3.13 mac — user asked about Windows CUDA;
        # metal path stays source-based by nature
        pass
    _pip("install", "llama-cpp-python", "--force-reinstall", "--no-cache-dir", env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install deps with prebuilt wheels when possible")
    parser.add_argument(
        "--force-source",
        action="store_true",
        help="Allow compiling llama-cpp-python from source (slow)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "cpu", "metal"],
        default="auto",
        help="Force backend (default: auto-detect)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from agent.gpu import detect_device, install_hint

    device = detect_device()
    backend = args.backend if args.backend != "auto" else device.backend
    print(f"[python] {sys.executable} ({_py_tag()})")
    print(f"[os] {device.os_name} arch={device.arch}")
    print(f"[device] {device.name}")
    print(f"[backend] {backend} (detected={device.backend})")
    print(f"[hint] {install_hint(device)}")

    # Fail fast on 3.13+ CUDA before mutating the environment
    if backend == "cuda" and not _supports_prebuilt_cuda_wheels() and not args.force_source:
        install_cuda(allow_source=False)

    _pip("install", "-U", "pip", "wheel", "setuptools")

    if backend == "cuda":
        install_cuda(allow_source=args.force_source)
    elif backend == "metal":
        install_metal(allow_source=args.force_source)
    else:
        install_cpu(allow_source=args.force_source)

    _install_rest()

    # Verify import + rough CUDA linkage hint
    code = (
        "import llama_cpp; "
        "print('[verify] llama_cpp', getattr(llama_cpp, '__version__', '?')); "
        "print('[verify] ok')"
    )
    subprocess.check_call([sys.executable, "-c", code])
    print("[ok] dependencies installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
