"""Auto-download GGUF weights from Hugging Face when missing."""
from __future__ import annotations

from pathlib import Path

from agent.config import MODEL_CATALOG, MODELS_DIR, ensure_dirs


def ensure_model(model_key: str, model_path: Path | None = None) -> Path:
    """
    Return a local GGUF path. If missing and the catalog entry is downloadable,
    fetch it into MODELS_DIR automatically.
    """
    ensure_dirs()

    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Explicit model path not found: {path}")

    if model_key not in MODEL_CATALOG:
        raise KeyError(f"Unknown model_key '{model_key}'. Known: {list(MODEL_CATALOG)}")

    meta = MODEL_CATALOG[model_key]
    dest = MODELS_DIR / meta["filename"]
    if dest.exists() and dest.stat().st_size > 0:
        return dest.resolve()

    repo_id = str(meta["repo_id"])
    if repo_id.startswith("local/"):
        raise FileNotFoundError(
            f"Model '{model_key}' is local-only. Place the GGUF at:\n  {dest}\n"
            "Or pass --model-path to main.py"
        )

    print(f"[download] fetching {repo_id} / {meta['filename']}")
    print(f"[download] target={dest}")
    print(f"[download] ~4+ GB — first run only")

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=meta["filename"],
        local_dir=str(MODELS_DIR),
    )
    path = Path(downloaded).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Download finished but file missing: {path}")
    print(f"[download] ready: {path}")
    return path
