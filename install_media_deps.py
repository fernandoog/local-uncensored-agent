#!/usr/bin/env python3
"""Install professional media backends (diffusers + edge-tts + ffmpeg helpers)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQ = ROOT / "requirements-media.txt"


def main() -> int:
    py = sys.version_info
    if py >= (3, 13):
        print("[warn] Python 3.13 may lack some wheels; prefer 3.11/3.12")
    print(f"[media] installing {REQ}")
    print("[media] this downloads PyTorch + Diffusers — several GB, once")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "-r", str(REQ)]
    print("[media]", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        return proc.returncode
    print("[media] OK — photoreal images via SD-Turbo, neural TTS via edge-tts")
    print("[media] tip: first generate_image will download model weights")
    print("[media] tip: MEDIA_SD_API_URL=http://127.0.0.1:7860 uses Automatic1111/Forge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
