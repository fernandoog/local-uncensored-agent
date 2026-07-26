"""Heuristic action planner for reliable local FS / shell / script tasks.

Small GGUF models often explain commands instead of emitting tool calls.
This planner turns clear user intents into concrete tool invocations.
"""
from __future__ import annotations

import re
from typing import Any

from agent.runners import CODE_FENCE_RE, LANG_EXT, default_hello, normalize_lang


def _name_from_text(text: str, default: str = "_agent_tmp") -> str:
    stop = {
        "y",
        "and",
        "luego",
        "then",
        "despues",
        "después",
        "para",
        "con",
        "sin",
        "el",
        "la",
        "un",
        "una",
        "the",
        "a",
        "an",
        "borralo",
        "borrala",
        "eliminarlo",
        "eliminarla",
        "script",
        "codigo",
        "código",
        "code",
        "programa",
        "program",
    }
    m = re.search(r'["\']([A-Za-z0-9._\\/-]+)["\']', text)
    if m:
        return m.group(1).replace("\\", "/").strip("/")
    m = re.search(
        r"(?:llamad[oa]|called|named|nombre|name)\s+[\"']?([A-Za-z0-9._-]+)[\"']?",
        text,
        re.I,
    )
    if m and m.group(1).lower() not in stop:
        return m.group(1)
    m = re.search(
        r"(?:dir|carpeta|folder|directorio)\s+[\"']?([A-Za-z0-9._-]+)[\"']?",
        text,
        re.I,
    )
    if m and m.group(1).lower() not in stop:
        return m.group(1)
    return default


def _detect_language(text: str) -> str | None:
    lower = text.lower()
    # explicit extension
    m = re.search(r"\.(py|js|ts|sh|ps1|bat|cmd|rb|pl|php|lua|r|go|rs|c|cpp|java|cs)\b", lower)
    if m:
        from agent.runners import detect_lang_from_path
        from pathlib import Path

        try:
            return detect_lang_from_path(Path(f"x.{m.group(1)}"))
        except ValueError:
            pass
    # language keywords (longest / specific first)
    patterns = [
        (r"\b(power\s*shell|powershell|pwsh)\b", "powershell"),
        (r"\b(type\s*script|typescript)\b", "typescript"),
        (r"\b(java\s*script|javascript|node\.?js|node)\b", "javascript"),
        (r"\b(python3?|py)\b", "python"),
        (r"\b(bash|shell|zsh|sh)\b", "bash"),
        (r"\b(cmd|bat|batch)\b", "cmd"),
        (r"\b(ruby|rb)\b", "ruby"),
        (r"\b(perl|pl)\b", "perl"),
        (r"\bphp\b", "php"),
        (r"\blua\b", "lua"),
        (r"\b(golang|go)\b", "go"),
        (r"\b(rust|rs)\b", "rust"),
        (r"\b(c\+\+|cpp)\b", "cpp"),
        (r"\b(c#|csharp|cs)\b", "csharp"),
        (r"\bjava\b", "java"),
        (r"\b(?<![\w])c(?![\w+#])\b", "c"),
        (r"\br\b", "r"),
    ]
    for pat, lang in patterns:
        if re.search(pat, lower):
            return lang
    return None


def _extract_inline_code(text: str) -> tuple[str | None, str | None]:
    m = CODE_FENCE_RE.search(text or "")
    if m:
        return m.group(1).lower(), m.group(2).strip()
    # triple-ish freeform: code after ":" or "que "
    return None, None


def plan_actions(user_text: str) -> list[dict[str, Any]]:
    """
    Return ordered tool calls for obvious local tasks.
    Empty list => let the LLM decide.
    """
    t = (user_text or "").strip()
    lower = t.lower()
    if not lower:
        return []

    # Media FIRST — never let the LLM invent fake afplay/ffmpeg paths
    media_plan = _plan_media(t, lower)
    if media_plan:
        return media_plan

    fence_lang, fence_code = _extract_inline_code(t)
    lang = _detect_language(t) or (normalize_lang(fence_lang) if fence_lang else None)

    wants_script = bool(
        re.search(r"\b(script|scripts|programa|program|codigo|código|code|snippet)\b", lower)
        or fence_code
        or (
            lang
            and re.search(r"\b(crea|crear|create|escribe|escribir|haz|hacer|genera|generate|write)\b", lower)
        )
    )
    wants_run = bool(
        re.search(
            r"\b(ejecuta|ejecutar|corre|correr|run|launch|compila|compilar|compile|"
            r"y\s+ejecutalo|y\s+ejecútalo|and\s+run)\b",
            lower,
        )
    )
    wants_create = bool(
        re.search(r"\b(crea|crear|create|mkdir|haz|hacer)\b", lower)
        and re.search(r"\b(directorio|carpeta|folder|dir|directory)\b", lower)
    )
    wants_delete = bool(
        re.search(
            r"\b(borra|borrar|borralo|borrala|borralos|borrarlo|borrarla|"
            r"elimina|eliminar|eliminalo|eliminarlo|delete|remove|rm|quita|quitar)\b",
            lower,
        )
        and (
            re.search(r"\b(directorio|carpeta|folder|dir|directory|archivo|file)\b", lower)
            or "luego" in lower
            or "after" in lower
            or "despues" in lower
            or "después" in lower
            or wants_create
        )
    )
    wants_list = bool(
        re.search(r"\b(lista|listar|list|muestra|mostrar|ls|dir)\b", lower)
        and re.search(r"\b(archivo|archivos|directorio|carpeta|folder|contenido|script)\b", lower)
    )
    wants_write = bool(
        re.search(r"\b(escribe|escribir|write|crea|crear|guarda|guardar)\b", lower)
        and re.search(r"\b(archivo|file|txt|json|md)\b", lower)
        and not wants_script
    )
    wants_read = bool(
        re.search(r"\b(lee|leer|read|abre|abrir|muestra|mostrar)\b", lower)
        and re.search(r"\b(archivo|file|txt|json|md|script)\b", lower)
        and not wants_script
    )
    wants_shell = bool(
        re.search(r"\b(ejecuta|ejecutar|run|shell|cmd|powershell|comando)\b", lower)
    )

    # Script create / run (prefer before generic shell)
    if wants_script and lang:
        code = fence_code or default_hello(lang)
        # lightweight intent: print hello
        if not fence_code and re.search(r"\b(hola|hello|world|mundo)\b", lower):
            code = default_hello(lang)
        name = _name_from_text(t, default="")
        filename = None
        if name and name != "_agent_tmp":
            from agent.runners import ext_for

            filename = name if "." in name else f"{name}.{ext_for(lang)}"
            if not filename.startswith("scripts/"):
                filename = f"scripts/{filename}"
        if lang == "java" and not filename:
            filename = "scripts/Hello.java"
            code = default_hello("java") if not fence_code else fence_code

        if wants_run or fence_code or re.search(r"\b(imprima|print|echo|hola|hello)\b", lower):
            return [
                {
                    "name": "run_code",
                    "arguments": {
                        "language": lang,
                        "code": code,
                        **({"filename": filename} if filename else {}),
                        "keep": True,
                    },
                }
            ]
        return [
            {
                "name": "write_script",
                "arguments": {
                    "language": lang,
                    "code": code,
                    **({"filename": filename} if filename else {}),
                },
            }
        ]

    # run existing script path
    m_path = re.search(
        r"(?:ejecuta|ejecutar|run|corre)\s+(?:el\s+)?(?:script\s+)?[\"']?([A-Za-z0-9._\\/-]+\.(?:py|js|ts|sh|ps1|bat|cmd|rb|pl|php|lua|r|go|rs|c|cpp|java|cs))[\"']?",
        t,
        re.I,
    )
    if m_path:
        return [{"name": "run_script", "arguments": {"path": m_path.group(1).replace('\\', '/')}}]

    # Explicit shell command in backticks
    m_cmd = re.search(r"`([^`]+)`", t)
    if wants_shell and m_cmd and not wants_script:
        return [{"name": "run_shell", "arguments": {"command": m_cmd.group(1).strip()}}]

    name = _name_from_text(t)

    if wants_create and wants_delete:
        return [
            {"name": "mkdir", "arguments": {"path": name}},
            {"name": "rm_path", "arguments": {"path": name, "recursive": True}},
        ]

    if wants_create:
        return [{"name": "mkdir", "arguments": {"path": name}}]

    if wants_delete:
        return [{"name": "rm_path", "arguments": {"path": name, "recursive": True}}]

    if wants_list:
        path = "."
        m = re.search(r"(?:en|in|de)\s+[\"']?([A-Za-z0-9._\\/-]+)[\"']?", t, re.I)
        if m:
            path = m.group(1)
        return [{"name": "list_dir", "arguments": {"path": path}}]

    if wants_write:
        content = ""
        m = re.search(r"(?:con|with|contenido|content)\s*[:=]?\s*[\"'](.+?)[\"']\s*$", t, re.I)
        if m:
            content = m.group(1)
        elif "hola" in lower:
            content = "hola"
        file_name = name if "." in name else f"{name}.txt"
        if name == "_agent_tmp":
            file_name = "nota.txt"
        return [{"name": "write_file", "arguments": {"path": file_name, "content": content or "ok"}}]

    if wants_read:
        file_name = name if "." in name else f"{name}.txt"
        return [{"name": "read_file", "arguments": {"path": file_name}}]

    m = re.search(r"(?:calcula|calculate|cuanto es|cuánto es)\s+(.+)$", lower)
    if m:
        expr = re.sub(r"[^0-9+\-*/().% ]", "", m.group(1))
        if expr.strip():
            return [{"name": "calc", "arguments": {"expression": expr.strip()}}]

    return []


def _extract_spoken_text(text: str) -> str | None:
    """Pull the words to speak from 'di/diciendo/saying/que diga ...'."""
    t = (text or "").strip()
    m = re.search(r"[\"'](.+?)[\"']", t)
    if m and len(m.group(1).strip()) >= 1:
        return m.group(1).strip()
    # Short command: "di X" / "di: X" / "say X"
    m = re.search(
        r"^(?:di|dile|diga|digas|say|speak|habla|pronuncia)\s*[:=]?\s+(.+)$",
        t,
        re.I,
    )
    if m:
        spoken = m.group(1).strip(" .\"'")
        if spoken:
            return spoken
    m = re.search(
        r"(?:diciendo|saying|say(?:ing)?|habla(?:ndo)?|que\s+diga|que\s+digas|"
        r"pronunc(?:ia|iar)|speak(?:ing)?)\s*[:=]?\s*(.+)$",
        t,
        re.I,
    )
    if m:
        spoken = m.group(1).strip(" .\"'")
        spoken = re.sub(
            r"^(que\s+|esto\s+|lo\s+siguiente\s*:\s*)",
            "",
            spoken,
            flags=re.I,
        ).strip()
        if spoken:
            return spoken
    return None


def _extract_media_prompt(text: str) -> str:
    spoken = _extract_spoken_text(text)
    if spoken:
        return spoken
    t = (text or "").strip()
    m = re.search(r"[\"'](.+?)[\"']", t)
    if m and len(m.group(1).strip()) >= 2:
        return m.group(1).strip()
    m = re.search(
        r"(?:imagen|image|foto|photo|video|clip|gif|audio|sonido|sound|musica|music|wav|tts|voz)"
        r"\s+(?:de|of|sobre|about|con|with|:)\s+(.+)$",
        t,
        re.I,
    )
    if m:
        return m.group(1).strip(" .")
    m = re.search(
        r"(?:genera|generar|generate|haz|hacer|crea|crear|make)\s+"
        r"(?:el\s+|la\s+|una?\s+|un\s+)?(?:imagen|image|foto|photo|video|clip|gif|audio|sonido|sound|"
        r"musica|music|wav|tts|voz)?\s*(?:de|of|sobre|about|con|with|:)?\s*(.+)$",
        t,
        re.I,
    )
    if m:
        prompt = m.group(1).strip(" .")
        prompt = re.sub(
            r"^(el\s+|la\s+|una?\s+|un\s+)?(imagen|image|foto|photo|video|clip|gif|audio|sonido|sound|"
            r"musica|music|wav)\s*(de|of|sobre)?\s*",
            "",
            prompt,
            flags=re.I,
        ).strip()
        if prompt and prompt.lower() not in {"sonido", "sound", "audio", "imagen", "video"}:
            return prompt
    return "beep"


def _plan_media(text: str, lower: str) -> list[dict[str, Any]]:
    spoken = _extract_spoken_text(text)
    # Bare speech commands: "di hijo puta", "say hello"
    if spoken and re.match(
        r"^(?:di|dile|diga|digas|say|speak|habla|pronuncia)\b",
        lower,
    ):
        return [
            {
                "name": "generate_audio",
                "arguments": {
                    "prompt": spoken,
                    "mode": "tts",
                    "voice": "auto",
                    "seconds": 4,
                },
            }
        ]

    wants_gen = bool(
        re.search(
            r"\b(genera|generar|generate|haz|hacer|crea|crear|make|draw|pinta|pintar|"
            r"render|synth|sintetiza|sintetizar)\b",
            lower,
        )
    )
    is_image = bool(
        re.search(r"\b(imagen|image|foto|photo|png|jpg|jpeg|dibujo|drawing|picture)\b", lower)
    )
    is_video = bool(
        re.search(r"\b(video|vídeo|clip|gif|mp4|animaci[oó]n|animation)\b", lower)
    )
    is_audio = bool(
        re.search(
            r"\b(audio|sonido|sound|musica|música|music|wav|mp3|tts|voz|voice|tone|tono|"
            r"beep|alarma|sirena|habla|hablar|diciendo|saying)\b",
            lower,
        )
        or spoken is not None
    )
    if not (is_image or is_video or is_audio):
        return []
    bare = bool(re.search(r"\b(de|of|:)\b", lower))
    if not (wants_gen or bare or spoken):
        return []

    prompt = spoken or _extract_media_prompt(text)
    if is_video:
        return [{"name": "generate_video", "arguments": {"prompt": prompt, "seconds": 2}}]
    if is_audio or spoken:
        speech = spoken is not None or bool(
            re.search(
                r"\b(tts|voz|voice|habla|hablar|speak|diciendo|saying|diga|digas|pronuncia|di)\b",
                lower,
            )
        )
        voice = "auto"
        if re.search(r"\b(mujer|woman|female|chica|girl)\b", lower):
            voice = "female"
        elif re.search(r"\b(hombre|man|male|chico|boy)\b", lower):
            voice = "male"
        mode = "tts" if speech else "tone"
        if re.search(r"\b(mujer|hombre|woman|man|voice|voz)\b", lower) and wants_gen:
            mode = "tts"
        args: dict[str, Any] = {
            "prompt": prompt,
            "seconds": 4,
            "mode": mode,
            "voice": voice,
        }
        return [{"name": "generate_audio", "arguments": args}]
    return [{"name": "generate_image", "arguments": {"prompt": prompt}}]
