"""Local tool registry and executor (tool-use layer)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


TOOL_CALL_RE = re.compile(
    r"```tool\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
# Fallback: TOOL name={"k":"v"} or {"name": "...", "arguments": {...}}
BARE_JSON_RE = re.compile(
    r"\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*(\{.*?\})\s*\}",
    re.DOTALL,
)
BASH_FENCE_RE = re.compile(r"```(?:bash|sh|shell|powershell|cmd)?\s*\n(.*?)```", re.DOTALL | re.I)


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
        calls = self.parse_all_tool_calls(text)
        return calls[0] if calls else None

    def parse_all_tool_calls(self, text: str) -> list[dict[str, Any]]:
        text = text or ""
        found: list[dict[str, Any]] = []

        for m in TOOL_CALL_RE.finditer(text):
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if "name" in data:
                data.setdefault("arguments", {})
                found.append(data)

        if not found:
            for m in BARE_JSON_RE.finditer(text):
                try:
                    args = json.loads(m.group(2))
                except json.JSONDecodeError:
                    continue
                found.append({"name": m.group(1), "arguments": args if isinstance(args, dict) else {}})

        # Last resort: execute fenced shell snippets as run_shell (one command per non-empty line)
        if not found:
            for m in BASH_FENCE_RE.finditer(text):
                block = m.group(1).strip()
                for line in block.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    found.append({"name": "run_shell", "arguments": {"command": line}})
                if found:
                    break

        return found

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

    def execute_many(self, calls: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
        return [(c, self.execute(c)) for c in calls]

    def _safe_path(self, rel: str) -> Path:
        rel = (rel or ".").replace("\\", "/").strip()
        if rel.startswith("/"):
            rel = rel.lstrip("/")
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
                name="mkdir",
                description="Create a directory (and parents) under the workspace",
                parameters={"path": "str"},
                handler=self._mkdir,
            )
        )
        self.register(
            Tool(
                name="rm_path",
                description="Delete a file or directory under the workspace",
                parameters={"path": "str", "recursive": "bool (default true for dirs)"},
                handler=self._rm_path,
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

    def _mkdir(self, args: dict[str, Any]) -> str:
        path = self._safe_path(str(args.get("path", "")))
        if not str(args.get("path", "")).strip():
            return "ERROR: empty path"
        path.mkdir(parents=True, exist_ok=True)
        return f"OK mkdir {path}"

    def _rm_path(self, args: dict[str, Any]) -> str:
        path = self._safe_path(str(args.get("path", "")))
        if not str(args.get("path", "")).strip():
            return "ERROR: empty path"
        if path == self.workspace:
            return "ERROR: refusing to delete workspace root"
        if not path.exists():
            return f"OK already absent {path}"
        recursive = args.get("recursive", True)
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
        else:
            path.unlink()
        return f"OK removed {path}"

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
