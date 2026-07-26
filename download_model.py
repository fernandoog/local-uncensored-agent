#!/usr/bin/env python3
"""Download GGUF for a key, or auto-pick from local device (any OS)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.config import AUTO_MODEL, assert_uncensored_model, print_disclaimer, uncensored_model_keys
from agent.download import ensure_model
from agent.gpu import detect_device, select_model_for_gpu


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download UNCENSORED-ONLY GGUF models"
    )
    parser.add_argument(
        "--model",
        default=AUTO_MODEL,
        choices=[AUTO_MODEL, *uncensored_model_keys()],
    )
    args = parser.parse_args()

    print_disclaimer()

    device = detect_device()
    print(
        f"[device] os={device.os_name} backend={device.backend} "
        f"{device.name} mem={device.vram_mb} MiB"
    )

    if args.model == AUTO_MODEL:
        selection = select_model_for_gpu(device)
        print(f"[select] {selection.reason}")
        key = selection.model_key
    else:
        key = args.model

    assert_uncensored_model(key)
    path = ensure_model(key)
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
