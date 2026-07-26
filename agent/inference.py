"""Local GGUF inference engine (llama-cpp-python + CUDA)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from agent.config import InferenceConfig


class InferenceEngine:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._llm = None

    def load(self, *, auto_download: bool = True) -> None:
        from agent.download import ensure_model

        if self.config.model_path is not None:
            path = Path(self.config.model_path)
            if not path.exists():
                raise FileNotFoundError(f"GGUF not found: {path}")
        else:
            path = self.config.resolve_model_path()
            if not path.exists():
                if not auto_download:
                    raise FileNotFoundError(
                        f"GGUF not found: {path}\n"
                        f"Remove --no-download or run: python download_model.py --model {self.config.model_key}"
                    )
                path = ensure_model(self.config.model_key)
            self.config.model_path = path

        from llama_cpp import Llama

        kwargs = self.config.to_llama_kwargs()
        kwargs["model_path"] = str(path.resolve())

        def _try_load(params: dict[str, Any]):
            try:
                return Llama(**params)
            except TypeError:
                cleaned = {k: v for k, v in params.items() if k != "flash_attn"}
                return Llama(**cleaned)

        try:
            self._llm = _try_load(kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if not any(k in msg for k in ("out of memory", "cuda", "metal", "ggml", "alloc")):
                raise
            kwargs["flash_attn"] = False
            kwargs["n_ctx"] = min(int(kwargs.get("n_ctx", 4096)), 3072)
            layers = kwargs.get("n_gpu_layers", -1)
            if layers == -1 or layers > 28:
                kwargs["n_gpu_layers"] = 24
            elif layers > 0:
                kwargs["n_gpu_layers"] = max(8, layers // 2)
            else:
                kwargs["n_gpu_layers"] = 0
            self.config.n_gpu_layers = kwargs["n_gpu_layers"]
            self.config.n_ctx = kwargs["n_ctx"]
            self._llm = _try_load(kwargs)

    @property
    def llm(self):
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._llm

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8")))

    def truncate_messages(
        self, messages: list[dict[str, str]], max_tokens: int | None = None
    ) -> list[dict[str, str]]:
        """Drop oldest non-system messages until prompt fits max_prompt_tokens."""
        budget = max_tokens or self.config.max_prompt_tokens
        msgs = list(messages)
        while len(msgs) > 1:
            blob = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            if self.count_tokens(blob) <= budget:
                break
            # Keep system (index 0); drop next oldest
            if len(msgs) <= 2:
                # Hard-trim last user content
                last = msgs[-1]
                tokens = self.llm.tokenize(last["content"].encode("utf-8"))
                keep = max(256, budget // 2)
                trimmed = self.llm.detokenize(tokens[-keep:]).decode("utf-8", errors="ignore")
                msgs[-1] = {**last, "content": trimmed}
                break
            del msgs[1]
        return msgs

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        safe = self.truncate_messages(messages)
        params: dict[str, Any] = {
            "messages": safe,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "stream": stream,
        }
        if stop:
            params["stop"] = stop

        if stream:
            def _gen() -> Iterator[str]:
                for chunk in self.llm.create_chat_completion(**params):
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta

            return _gen()

        out = self.llm.create_chat_completion(**params)
        return out["choices"][0]["message"]["content"] or ""

    def unload(self) -> None:
        self._llm = None
