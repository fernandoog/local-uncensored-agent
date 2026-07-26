"""Local tool registry and executor (tool-use layer)."""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


TOOL_CALL_RE = re.compile(
    r"```tool\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[[dict[str, Any]], str]
    parameters: dict[str, str]


class ToolRegistry:
    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = (workspace or Path.cwd()).resolve()
        self._tools: dict[str, Tool] = {}
        self._register_defaults()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schema_text(self) -> str:
        lines = []
        for t in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items())
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    def parse_tool_call(self, text: str) -> dict[str, Any] | None:
        m = TOOL_CALL_RE.search(text or "")
        if not m:
            return None
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
        if "name" not in data:
            return None
        data.setdefault("arguments", {})
        return data

    def execute(self, call: dict[str, Any]) -> str:
        name = call.get("name", "")
        args = call.get("arguments") or {}
        tool = self._tools.get(name)
        if not tool:
            return f"ERROR: unknown tool '{name}'. Available: {list(self._tools)}"
        try:
            return tool.handler(args if isinstance(args, dict) else {})
        except Exception as exc:
            return f"ERROR executing {name}: {exc}"

    def _safe_path(self, rel: str) -> Path:
        p = (self.workspace / rel).resolve()
        if not str(p).startswith(str(self.workspace)):
            raise ValueError("path escapes workspace")
        return p

    def _register_defaults(self) -> None:
        self.register(
            Tool(
                name="list_dir",
                description="List files in a relative directory under the workspace",
                parameters={"path": "str (default '.')"},
                handler=self._list_dir,
            )
        )
        self.register(
            Tool(
                name="read_file",
                description="Read a UTF-8 text file (capped at 32 KB)",
                parameters={"path": "str"},
                handler=self._read_file,
            )
        )
        self.register(
            Tool(
                name="write_file",
                description="Write UTF-8 text to a file under the workspace",
                parameters={"path": "str", "content": "str"},
                handler=self._write_file,
            )
        )
        self.register(
            Tool(
                name="run_shell",
                description="Run a shell command in the workspace (timeout 60s)",
                parameters={"command": "str"},
                handler=self._run_shell,
            )
        )
        self.register(
            Tool(
                name="calc",
                description="Evaluate a simple arithmetic expression",
                parameters={"expression": "str"},
                handler=self._calc,
            )
        )

    def _list_dir(self, args: dict[str, Any]) -> str:
        path = self._safe_path(str(args.get("path", ".")))
        if not path.exists():
            return f"ERROR: {path} does not exist"
        entries = sorted(os.listdir(path))
        return "\n".join(entries) if entries else "(empty)"

    def _read_file(self, args: dict[str, Any]) -> str:
        path = self._safe_path(str(args["path"]))
        data = path.read_bytes()[:32768]
        return data.decode("utf-8", errors="replace")

    def _write_file(self, args: dict[str, Any]) -> str:
        path = self._safe_path(str(args["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args.get("content", "")), encoding="utf-8")
        return f"OK wrote {path}"

    def _run_shell(self, args: dict[str, Any]) -> str:
        import platform

        cmd = str(args.get("command", ""))
        if not cmd:
            return "ERROR: empty command"
        # shell=True uses COMSPEC on Windows, /bin/sh on Unix/macOS
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"os={platform.system()} exit={proc.returncode}\n{out[:16000]}"

    def _calc(self, args: dict[str, Any]) -> str:
        expr = str(args.get("expression", ""))
        allowed = set("0123456789+-*/().% ")
        if not expr or any(c not in allowed for c in expr):
            return "ERROR: invalid expression"
        return str(eval(expr, {"__builtins__": {}}, {}))
