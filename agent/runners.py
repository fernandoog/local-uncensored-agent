"""Multi-language script writers / runners for the local hacker agent."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


LANG_EXT: dict[str, str] = {
    "python": "py",
    "py": "py",
    "python3": "py",
    "javascript": "js",
    "js": "js",
    "node": "js",
    "typescript": "ts",
    "ts": "ts",
    "bash": "sh",
    "sh": "sh",
    "shell": "sh",
    "zsh": "sh",
    "powershell": "ps1",
    "ps1": "ps1",
    "pwsh": "ps1",
    "cmd": "cmd",
    "bat": "bat",
    "batch": "bat",
    "ruby": "rb",
    "rb": "rb",
    "perl": "pl",
    "pl": "pl",
    "php": "php",
    "lua": "lua",
    "r": "r",
    "go": "go",
    "golang": "go",
    "rust": "rs",
    "rs": "rs",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "csharp": "cs",
    "cs": "cs",
    "java": "java",
    "kotlin": "kt",
    "swift": "swift",
    "haskell": "hs",
    "scala": "scala",
}

CODE_FENCE_RE = re.compile(
    r"```(python|py|javascript|js|typescript|ts|bash|sh|shell|zsh|powershell|ps1|"
    r"cmd|bat|ruby|rb|perl|pl|php|lua|r|go|rust|rs|c|cpp|c\+\+|java|csharp|cs)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def normalize_lang(lang: str) -> str:
    key = (lang or "python").strip().lower()
    if key not in LANG_EXT:
        raise ValueError(f"unsupported language '{lang}'. Supported: {sorted(set(LANG_EXT))}")
    return key


def ext_for(lang: str) -> str:
    return LANG_EXT[normalize_lang(lang)]


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def build_run_argv(lang: str, script: Path, workspace: Path) -> list[str]:
    lang = normalize_lang(lang)
    s = str(script)

    if lang in {"python", "py", "python3"}:
        for cand in ("python", "python3", "py"):
            if which(cand):
                if cand == "py":
                    return [cand, "-3", s]
                return [cand, s]
        raise RuntimeError("python not found on PATH")

    if lang in {"javascript", "js", "node"}:
        node = which("node")
        if not node:
            raise RuntimeError("node not found on PATH")
        return [node, s]

    if lang in {"typescript", "ts"}:
        if which("npx"):
            return ["npx", "--yes", "ts-node", s]
        raise RuntimeError("npx/ts-node not available")

    if lang in {"bash", "sh", "shell", "zsh"}:
        bash = which("bash") or which("sh")
        if bash:
            return [bash, s]
        # Windows fallback
        if which("wsl"):
            return ["wsl", "bash", s]
        raise RuntimeError("bash/sh not found")

    if lang in {"powershell", "ps1", "pwsh"}:
        pwsh = which("pwsh") or which("powershell")
        if not pwsh:
            raise RuntimeError("powershell not found")
        return [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", s]

    if lang in {"cmd", "bat", "batch"}:
        return ["cmd", "/c", s]

    if lang in {"ruby", "rb"}:
        ruby = which("ruby")
        if not ruby:
            raise RuntimeError("ruby not found")
        return [ruby, s]

    if lang in {"perl", "pl"}:
        perl = which("perl")
        if not perl:
            raise RuntimeError("perl not found")
        return [perl, s]

    if lang == "php":
        php = which("php")
        if not php:
            raise RuntimeError("php not found")
        return [php, s]

    if lang == "lua":
        lua = which("lua") or which("luajit")
        if not lua:
            raise RuntimeError("lua not found")
        return [lua, s]

    if lang == "r":
        rscript = which("Rscript")
        if not rscript:
            raise RuntimeError("Rscript not found")
        return [rscript, s]

    if lang in {"go", "golang"}:
        go = which("go")
        if not go:
            raise RuntimeError("go not found")
        return [go, "run", s]

    if lang in {"rust", "rs"}:
        rustc = which("rustc")
        if not rustc:
            raise RuntimeError("rustc not found")
        out = script.with_suffix(".exe" if os.name == "nt" else "")
        # caller compiles then runs — handled in run_script_path for rust/c/cpp/java
        return [rustc, s, "-o", str(out)]

    if lang == "c":
        cc = which("gcc") or which("clang") or which("cl")
        if not cc:
            raise RuntimeError("C compiler (gcc/clang) not found")
        out = script.with_suffix(".exe" if os.name == "nt" else ".out")
        return [cc, s, "-o", str(out)]

    if lang in {"cpp", "c++", "cc"}:
        cxx = which("g++") or which("clang++")
        if not cxx:
            raise RuntimeError("C++ compiler (g++/clang++) not found")
        out = script.with_suffix(".exe" if os.name == "nt" else ".out")
        return [cxx, s, "-o", str(out)]

    if lang == "java":
        javac = which("javac")
        if not javac:
            raise RuntimeError("javac not found")
        return [javac, s]

    if lang in {"csharp", "cs"}:
        dotnet = which("dotnet")
        if dotnet:
            return [dotnet, "script", s]
        csc = which("csc")
        if csc:
            out = script.with_suffix(".exe")
            return [csc, f"/out:{out}", s]
        raise RuntimeError("dotnet/csc not found")

    raise RuntimeError(f"no runner wired for {lang}")


def _run(argv: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        shell=False,
    )


def run_script_path(lang: str, script: Path, workspace: Path, timeout: int = 60) -> str:
    lang = normalize_lang(lang)
    script = script.resolve()

    # Compile-then-run languages
    if lang in {"c", "cpp", "c++", "cc", "rust", "rs"}:
        compile_argv = build_run_argv(lang, script, workspace)
        out_bin = Path(compile_argv[-1]) if lang in {"c", "cpp", "c++", "cc", "rust", "rs"} else script
        # build_run_argv ends with -o <out>
        proc = _run(compile_argv, workspace, timeout=timeout)
        if proc.returncode != 0:
            return (
                f"compile_exit={proc.returncode}\n"
                f"cmd={' '.join(compile_argv)}\n"
                f"{proc.stdout}{proc.stderr}"
            )[:16000]
        run_proc = _run([str(out_bin)], workspace, timeout=timeout)
        return (
            f"compile_ok\nrun_exit={run_proc.returncode}\n"
            f"{run_proc.stdout}{run_proc.stderr}"
        )[:16000]

    if lang == "java":
        proc = _run(build_run_argv(lang, script, workspace), workspace, timeout=timeout)
        if proc.returncode != 0:
            return f"javac_exit={proc.returncode}\n{proc.stdout}{proc.stderr}"[:16000]
        # Class name = filename stem (simple single-file programs)
        java = which("java")
        if not java:
            return "ERROR: java runtime not found after javac"
        run_proc = _run([java, "-cp", str(script.parent), script.stem], workspace, timeout=timeout)
        return f"javac_ok\nrun_exit={run_proc.returncode}\n{run_proc.stdout}{run_proc.stderr}"[:16000]

    if lang in {"csharp", "cs"} and which("csc") and not which("dotnet"):
        proc = _run(build_run_argv(lang, script, workspace), workspace, timeout=timeout)
        if proc.returncode != 0:
            return f"csc_exit={proc.returncode}\n{proc.stdout}{proc.stderr}"[:16000]
        exe = script.with_suffix(".exe")
        run_proc = _run([str(exe)], workspace, timeout=timeout)
        return f"csc_ok\nrun_exit={run_proc.returncode}\n{run_proc.stdout}{run_proc.stderr}"[:16000]

    argv = build_run_argv(lang, script, workspace)
    proc = _run(argv, workspace, timeout=timeout)
    return f"cmd={' '.join(argv)}\nexit={proc.returncode}\n{proc.stdout}{proc.stderr}"[:16000]


def extract_fenced_code(text: str) -> list[tuple[str, str]]:
    """Return list of (language, code) from markdown fences."""
    out: list[tuple[str, str]] = []
    for m in CODE_FENCE_RE.finditer(text or ""):
        lang = m.group(1).lower()
        code = m.group(2).strip()
        if code:
            out.append((lang, code))
    return out


def detect_lang_from_path(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    rev = {v: k for k, v in LANG_EXT.items()}
    # prefer canonical names
    preferred = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "sh": "bash",
        "ps1": "powershell",
        "rb": "ruby",
        "pl": "perl",
        "rs": "rust",
        "cpp": "cpp",
        "c": "c",
        "java": "java",
        "cs": "csharp",
        "go": "go",
        "php": "php",
        "lua": "lua",
        "r": "r",
        "bat": "bat",
        "cmd": "cmd",
    }
    if ext in preferred:
        return preferred[ext]
    if ext in rev:
        return rev[ext]
    raise ValueError(f"cannot detect language from extension .{ext}")


def default_hello(lang: str) -> str:
    lang = normalize_lang(lang)
    samples: dict[str, str] = {
        "python": 'print("hello from python")\n',
        "javascript": 'console.log("hello from node");\n',
        "typescript": 'console.log("hello from typescript");\n',
        "bash": 'echo "hello from bash"\n',
        "powershell": 'Write-Output "hello from powershell"\n',
        "cmd": "@echo hello from cmd\n",
        "ruby": 'puts "hello from ruby"\n',
        "perl": 'print "hello from perl\\n";\n',
        "php": '<?php echo "hello from php\\n";\n',
        "lua": 'print("hello from lua")\n',
        "r": 'cat("hello from R\\n")\n',
        "go": 'package main\nimport "fmt"\nfunc main(){ fmt.Println("hello from go") }\n',
        "rust": 'fn main(){ println!("hello from rust"); }\n',
        "c": '#include <stdio.h>\nint main(){ printf("hello from c\\n"); return 0; }\n',
        "cpp": '#include <iostream>\nint main(){ std::cout<<"hello from cpp\\n"; }\n',
        "java": 'public class Hello { public static void main(String[] a){ System.out.println("hello from java"); } }\n',
        "csharp": 'System.Console.WriteLine("hello from csharp");\n',
    }
    # map aliases
    key = lang
    for alias, ext in LANG_EXT.items():
        if alias == lang:
            for canon, sample in samples.items():
                if LANG_EXT.get(canon) == ext and canon in samples:
                    return samples[canon]
    return samples.get(key, 'print("hello")\n')


def stamp_name(lang: str, filename: str | None = None) -> str:
    if filename:
        return filename.replace("\\", "/").lstrip("/")
    ext = ext_for(lang)
    return f"scripts/agent_{int(time.time())}.{ext}"
