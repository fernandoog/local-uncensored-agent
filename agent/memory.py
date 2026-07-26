"""Working memory: rolling buffer + rolling summary + optional persistent JSONL."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


Summarizer = Callable[[list[dict[str, str]]], str]


@dataclass
class MemoryStore:
    buffer: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    max_buffer: int = 40
    summary_trigger: int = 16
    keep_recent: int = 6
    persist_path: Path | None = None
    summarizer: Summarizer | None = None

    def load_persistent(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        from agent.refusal import looks_like_refusal

        with self.persist_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "summary":
                    content = rec.get("content", self.summary)
                    if not looks_like_refusal(content):
                        self.summary = content
                elif rec.get("type") == "message":
                    content = rec.get("content", "")
                    # Drop past refusals so they don't poison future turns
                    if looks_like_refusal(content):
                        continue
                    self.buffer.append({"role": rec["role"], "content": content})

    def _append_persist(self, record: dict) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add(self, role: str, content: str, *, persist: bool = True) -> None:
        msg = {"role": role, "content": content}
        self.buffer.append(msg)
        if persist:
            self._append_persist(
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer :]
        if len(self.buffer) >= self.summary_trigger:
            self.compact()

    def compact(self) -> None:
        if not self.summarizer or len(self.buffer) < self.keep_recent + 2:
            # Drop oldest without LLM if no summarizer
            overflow = len(self.buffer) - self.keep_recent
            if overflow > 0:
                dropped = self.buffer[:overflow]
                self.buffer = self.buffer[overflow:]
                blob = " | ".join(f"{m['role']}:{m['content'][:120]}" for m in dropped)
                self.summary = (self.summary + " " + blob).strip()[-4000:]
            return

        old = self.buffer[: -self.keep_recent]
        recent = self.buffer[-self.keep_recent :]
        piece = self.summarizer(old)
        self.summary = (self.summary + "\n" + piece).strip()[-6000:]
        self.buffer = recent
        self._append_persist(
            {
                "type": "summary",
                "content": self.summary,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    def as_messages(self, system_prompt: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if self.summary:
            msgs.append(
                {
                    "role": "system",
                    "content": f"[MEMORY SUMMARY]\n{self.summary}",
                }
            )
        msgs.extend(self.buffer)
        return msgs

    def clear(self, *, wipe_disk: bool = False) -> None:
        self.buffer.clear()
        self.summary = ""
        if wipe_disk and self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
