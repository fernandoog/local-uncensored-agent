#!/usr/bin/env bash
set -euo pipefail
PROJ="$(pwd)"
echo "[wsl] pwd=$PROJ"
export PATH="$HOME/.local/bin:$PATH"

VENV="$HOME/.venvs/local-uncensored-agent"
uv venv "$VENV" --python 3.12 --clear
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[wsl] installing CPU llama-cpp-python wheel (no CUDA toolkit in this WSL)..."
uv pip uninstall -y llama-cpp-python 2>/dev/null || true
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --reinstall
uv pip install huggingface-hub pydantic rich diskcache

export PYTHONIOENCODING=utf-8
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
python -c "import llama_cpp; print('[wsl] llama_cpp', llama_cpp.__version__)"
python "$PROJ/smoke_hello.py" qwen25-1.5b-uncensored-es-q4
echo "[wsl] done"
