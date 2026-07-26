#!/usr/bin/env python3
"""CLI entry: cross-platform local agent (Windows / Linux / macOS)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import (
    AUTO_MODEL,
    AgentConfig,
    InferenceConfig,
    assert_uncensored_model,
    print_disclaimer,
    uncensored_model_keys,
)
from agent.gpu import detect_device, install_hint, select_model_for_gpu
from agent.pipeline import AgentPipeline


def build_parser() -> argparse.ArgumentParser:
    choices = [AUTO_MODEL, *uncensored_model_keys()]
    p = argparse.ArgumentParser(
        description="Local UNCENSORED-ONLY GGUF agent (Windows/Linux/macOS)"
    )
    p.add_argument(
        "--model",
        default=AUTO_MODEL,
        choices=choices,
        help="Uncensored model key, or 'auto' (default) from local device memory",
    )
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--n-ctx", type=int, default=None)
    p.add_argument("--n-gpu-layers", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    p.add_argument("--no-persist", action="store_true")
    p.add_argument("--no-download", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print_disclaimer()

    device = detect_device()
    print(
        f"[device] os={device.os_name} backend={device.backend} "
        f"name={device.name} mem_budget={device.vram_mb} MiB "
        f"ram={device.total_ram_mb} MiB source={device.source}"
    )
    print(f"[install] if llama-cpp missing: {install_hint(device)}")

    selection = select_model_for_gpu(device)
    if args.model == AUTO_MODEL:
        model_key = selection.model_key
        print(f"[select] {selection.reason}")
    else:
        model_key = args.model
        print(f"[select] manual UNCENSORED model={model_key} (device-tuned ctx/layers)")

    meta = assert_uncensored_model(model_key)
    n_ctx = selection.n_ctx if args.n_ctx is None else args.n_ctx
    n_gpu_layers = selection.n_gpu_layers if args.n_gpu_layers is None else args.n_gpu_layers
    print(f"[boot] model={model_key} file={meta['filename']} format={meta['chat_format']}")
    print(
        f"[boot] uncensored_model=True refusal_risk={meta.get('refusal_risk', '?')} "
        f"agent_mode=UNCENSORED-ONLY"
    )
    print(
        f"[boot] n_ctx={n_ctx} n_gpu_layers={n_gpu_layers} "
        f"n_batch={selection.n_batch} n_threads={selection.n_threads}"
    )
    if not args.no_download and args.model_path is None:
        print("[boot] auto-download enabled if GGUF missing (uncensored only)")
    print(f"[hint] uncensored keys: {', '.join(uncensored_model_keys())}")

    cfg = AgentConfig(
        inference=InferenceConfig(
            model_key=model_key,
            model_path=args.model_path,
            n_ctx=n_ctx,
            n_batch=selection.n_batch,
            n_gpu_layers=n_gpu_layers,
            n_threads=selection.n_threads,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_prompt_tokens=selection.max_prompt_tokens,
            flash_attn=selection.flash_attn,
        ),
        uncensored=True,
    )
    agent = AgentPipeline(config=cfg, workspace=args.workspace.resolve())
    agent.start(load_memory=not args.no_persist, auto_download=not args.no_download)
    print("[ready] commands: /status /clear /exit")
    print("[media] generated files -> <workspace>/outputs/media/")
    print("-" * 60)

    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        result = agent.step(line)
        if result.text == "__EXIT__":
            break
        # action.emit already printed the reply; keep separator for readability
        print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
