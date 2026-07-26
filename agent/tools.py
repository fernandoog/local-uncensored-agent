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

        # Last resort: fenced source code => run_code ; shell fences => run_shell
        if not found:
            from agent.runners import extract_fenced_code

            fenced = extract_fenced_code(text)
            if fenced:
                for lang, code in fenced:
                    found.append(
                        {
                            "name": "run_code",
                            "arguments": {"language": lang, "code": code, "keep": True},
                        }
                    )
            else:
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
                name="write_script",
                description=(
                    "Create a script under scripts/ for a language "
                    "(python, javascript, bash, powershell, c, cpp, go, rust, java, ruby, perl, php, lua, r, ...)"
                ),
                parameters={"language": "str", "code": "str", "filename": "str optional"},
                handler=self._write_script,
            )
        )
        self.register(
            Tool(
                name="run_script",
                description="Execute an existing script file (language auto-detected from extension)",
                parameters={"path": "str", "language": "str optional", "timeout": "int optional"},
                handler=self._run_script,
            )
        )
        self.register(
            Tool(
                name="run_code",
                description=(
                    "Write and execute code in one shot. "
                    "language: python|javascript|bash|powershell|c|cpp|go|rust|java|..."
                ),
                parameters={
                    "language": "str",
                    "code": "str",
                    "filename": "str optional",
                    "keep": "bool default true",
                    "timeout": "int optional",
                },
                handler=self._run_code,
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
        self.register(
            Tool(
                name="generate_image",
                description=(
                    "Generate an image from a text prompt and save PNG under outputs/media. "
                    "Offline Pillow by default; set MEDIA_SD_API_URL for Automatic1111/Forge."
                ),
                parameters={
                    "prompt": "str",
                    "out": "str optional relative path",
                    "width": "int optional default 768",
                    "height": "int optional default 512",
                },
                handler=self._generate_image,
            )
        )
        self.register(
            Tool(
                name="generate_audio",
                description=(
                    "Generate sound WAV from a prompt (synth tone) or TTS if pyttsx3 + MEDIA_TTS=1. "
                    "Saves under outputs/media."
                ),
                parameters={
                    "prompt": "str (description or spoken text)",
                    "out": "str optional",
                    "seconds": "float optional default 4",
                    "mode": "auto|tone|tts optional",
                    "voice": "auto|female|male optional",
                },
                handler=self._generate_audio,
            )
        )
        self.register(
            Tool(
                name="generate_video",
                description=(
                    "Generate a short animated video/GIF from a text prompt under outputs/media."
                ),
                parameters={
                    "prompt": "str",
                    "out": "str optional (.gif or .mp4)",
                    "seconds": "float optional default 2",
                    "fps": "int optional default 8",
                    "width": "int optional",
                    "height": "int optional",
                },
                handler=self._generate_video,
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

    def _write_script(self, args: dict[str, Any]) -> str:
        from agent.runners import default_hello, normalize_lang, stamp_name

        lang = normalize_lang(str(args.get("language", "python")))
        code = str(args.get("code") or default_hello(lang))
        filename = args.get("filename")
        rel = stamp_name(lang, str(filename) if filename else None)
        if not rel.startswith("scripts/"):
            rel = f"scripts/{Path(rel).name}"
        path = self._safe_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8", newline="\n")
        return f"OK wrote script lang={lang} path={path}"

    def _run_script(self, args: dict[str, Any]) -> str:
        from agent.runners import detect_lang_from_path, run_script_path

        rel = str(args.get("path", ""))
        if not rel.strip():
            return "ERROR: empty path"
        path = self._safe_path(rel)
        if not path.exists():
            return f"ERROR: missing {path}"
        lang = str(args.get("language") or detect_lang_from_path(path))
        timeout = int(args.get("timeout", 60))
        return run_script_path(lang, path, self.workspace, timeout=timeout)

    def _run_code(self, args: dict[str, Any]) -> str:
        from agent.runners import default_hello, normalize_lang, run_script_path, stamp_name

        lang = normalize_lang(str(args.get("language", "python")))
        code = str(args.get("code") or default_hello(lang))
        filename = args.get("filename")
        keep = bool(args.get("keep", True))
        timeout = int(args.get("timeout", 60))
        rel = stamp_name(lang, str(filename) if filename else None)
        if not rel.startswith("scripts/"):
            rel = f"scripts/{Path(rel).name}"
        path = self._safe_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8", newline="\n")
        try:
            result = run_script_path(lang, path, self.workspace, timeout=timeout)
            return f"path={path}\n{result}"
        finally:
            if not keep:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _run_shell(self, args: dict[str, Any]) -> str:
        import platform

        from agent.refusal import looks_like_fake_media_shell

        cmd = str(args.get("command", ""))
        if not cmd:
            return "ERROR: empty command"
        if looks_like_fake_media_shell(cmd):
            return (
                "ERROR: fake media shell blocked. "
                "Use generate_audio / generate_image / generate_video instead of afplay paths."
            )
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

    def _media_out(self, rel: str | None, default_ext: str) -> Path:
        if rel and str(rel).strip():
            p = self._safe_path(str(rel))
        else:
            import time

            stamp = time.strftime("%Y%m%d_%H%M%S")
            p = self._safe_path(f"outputs/media/media_{stamp}{default_ext}")
        if p.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".mp4",
            ".wav",
            ".mp3",
            ".webp",
        }:
            p = p.with_suffix(default_ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _generate_image(self, args: dict[str, Any]) -> str:
        from agent.media import generate_image

        prompt = str(args.get("prompt") or args.get("text") or "").strip()
        if not prompt:
            return "ERROR: prompt required"
        out = self._media_out(args.get("out"), ".png")
        # include prompt slug in default name
        if not args.get("out"):
            from agent.media import _slug
            import time

            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = self._safe_path(f"outputs/media/img_{_slug(prompt)}_{stamp}.png")
            out.parent.mkdir(parents=True, exist_ok=True)
        result = generate_image(
            prompt,
            out=out,
            width=int(args.get("width", 768)),
            height=int(args.get("height", 512)),
            backend=args.get("backend"),
        )
        return json.dumps(result, ensure_ascii=False)

    def _generate_audio(self, args: dict[str, Any]) -> str:
        from agent.media import generate_audio, _slug
        import time

        prompt = str(args.get("prompt") or args.get("text") or "").strip()
        if not prompt:
            return "ERROR: prompt required"
        if args.get("out"):
            out = self._media_out(args.get("out"), ".wav")
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = self._safe_path(f"outputs/media/audio_{_slug(prompt)}_{stamp}.wav")
            out.parent.mkdir(parents=True, exist_ok=True)
        result = generate_audio(
            prompt,
            out=out,
            seconds=float(args.get("seconds", 4)),
            mode=args.get("mode"),
            voice=str(args.get("voice") or "auto"),
        )
        return json.dumps(result, ensure_ascii=False)

    def _generate_video(self, args: dict[str, Any]) -> str:
        from agent.media import generate_video, _slug
        import time

        prompt = str(args.get("prompt") or args.get("text") or "").strip()
        if not prompt:
            return "ERROR: prompt required"
        if args.get("out"):
            out = self._media_out(args.get("out"), ".gif")
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = self._safe_path(f"outputs/media/video_{_slug(prompt)}_{stamp}.gif")
            out.parent.mkdir(parents=True, exist_ok=True)
        result = generate_video(
            prompt,
            out=out,
            seconds=float(args.get("seconds", 2)),
            fps=int(args.get("fps", 8)),
            width=int(args.get("width", 512)),
            height=int(args.get("height", 320)),
        )
        return json.dumps(result, ensure_ascii=False)
