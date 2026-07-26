"""Action layer: emit final responses / side-effects to the user."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


PrintFn = Callable[[str], None]


@dataclass
class ActionResult:
    text: str
    ok: bool = True


class ActionLayer:
    def __init__(self, printer: PrintFn | None = None) -> None:
        self.printer = printer or (lambda s: print(s, flush=True))

    def emit(self, text: str) -> ActionResult:
        self.printer(text)
        return ActionResult(text=text, ok=True)

    def emit_error(self, err: Exception) -> ActionResult:
        msg = f"[AGENT ERROR] {type(err).__name__}: {err}"
        self.printer(msg)
        return ActionResult(text=msg, ok=False)
