#!/usr/bin/env python3
"""One-shot hello-world smoke test (no REPL)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.config import AgentConfig, InferenceConfig
from agent.pipeline import AgentPipeline


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen25-1.5b-uncensored-es-q4"
    print(f"[test] model={model}")
    cfg = AgentConfig(
        inference=InferenceConfig(
            model_key=model,
            n_ctx=2048,
            n_gpu_layers=-1,
            max_tokens=64,
            temperature=0.2,
            max_prompt_tokens=1500,
            flash_attn=False,
        )
    )
    agent = AgentPipeline(config=cfg, workspace=ROOT)
    agent.start(load_memory=False, auto_download=True)
    result = agent.step("Reply with exactly: HELLO WORLD")
    print("[test] reply=", repr(result.text[:500] if result.text else result.text))
    print("[test] ok=", result.ok)
    return 0 if result.ok and result.text else 1


if __name__ == "__main__":
    raise SystemExit(main())
