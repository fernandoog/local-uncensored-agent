"""Perception: normalize raw user input into structured agent messages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PerceivedInput:
    text: str
    intent: str
    metadata: dict[str, Any]


class Perception:
    """Map stdin / API text → ChatML-ready user message + light intent tag."""

    INTENT_HINTS = (
        (
            "tool",
            (
                "run",
                "execute",
                "lista",
                "list",
                "archivo",
                "file",
                "cmd",
                "shell",
                "busca",
                "crea",
                "crear",
                "borra",
                "borrar",
                "elimina",
                "mkdir",
                "rm",
                "directorio",
                "carpeta",
                "escribe",
                "lee",
                "calcula",
            ),
        ),
        ("memory", ("recuerda", "remember", "olvida", "forget", "historial")),
        ("chat", ()),
    )

    def perceive(self, raw: str, metadata: dict[str, Any] | None = None) -> PerceivedInput:
        text = (raw or "").strip()
        lower = text.lower()
        intent = "chat"
        for name, keys in self.INTENT_HINTS:
            if keys and any(k in lower for k in keys):
                intent = name
                break
        return PerceivedInput(text=text, intent=intent, metadata=metadata or {})

    def to_message(self, perceived: PerceivedInput) -> dict[str, str]:
        return {"role": "user", "content": perceived.text}
