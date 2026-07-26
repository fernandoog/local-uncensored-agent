#!/usr/bin/env bash
# Setup macOS: Python 3.12 venv + llama-cpp Metal (Apple Silicon) or CPU (Intel).
# Usage:
#   chmod +x setup_macos.sh
#   ./setup_macos.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[setup] project=$ROOT"
echo "[setup] arch=$(uname -m)  darwin=$(sw_vers -productVersion 2>/dev/null || true)"

# Xcode Command Line Tools (needed to compile Metal / C extensions)
if ! xcode-select -p >/dev/null 2>&1; then
  echo "[setup] installing Xcode Command Line Tools..."
  xcode-select --install || true
  echo "[setup] finish the CLT installer GUI, then re-run ./setup_macos.sh"
  exit 1
fi

# Homebrew + Python 3.12
if ! command -v brew >/dev/null 2>&1; then
  echo "[setup] Homebrew not found. Install from https://brew.sh then re-run."
  exit 1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "[setup] installing python@3.12 via Homebrew..."
  brew install python@3.12
fi

PY="$(command -v python3.12)"
echo "[setup] python=$PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

if [ -d .venv ]; then
  echo "[setup] removing old .venv..."
  rm -rf .venv
fi

"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
  echo "[setup] Apple Silicon → Metal build (CMAKE_ARGS=-DGGML_METAL=on)"
  export CMAKE_ARGS="-DGGML_METAL=on"
  python install_deps.py --backend metal --force-source
else
  echo "[setup] Intel Mac → CPU wheel / build"
  python install_deps.py --backend cpu
fi

echo ""
echo "[ok] Activate with:"
echo "  source .venv/bin/activate"
echo "  python smoke_hello.py qwen25-1.5b-uncensored-es-q4"
echo "  python main.py"
