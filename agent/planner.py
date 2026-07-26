"""Heuristic action planner for reliable local FS / shell tasks.

Small GGUF models often explain commands instead of emitting tool calls.
This planner turns clear user intents into concrete tool invocations.
"""
from __future__ import annotations

import re
from typing import Any


def _name_from_text(text: str, default: str = "_agent_tmp") -> str:
    stop = {
        "y",
        "and",
        "luego",
        "then",
        "despues",
        "después",
        "despues",
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
    }
    # quoted name
    m = re.search(r'["\']([A-Za-z0-9._\\/-]+)["\']', text)
    if m:
        return m.group(1).replace("\\", "/").strip("/")
    # "llamado X" / "called X" / "nombre X" / "carpeta X"
    m = re.search(
        r"(?:llamad[oa]|called|named|nombre|name)\s+[\"']?([A-Za-z0-9._-]+)[\"']?",
        text,
        re.I,
    )
    if m and m.group(1).lower() not in stop:
        return m.group(1)
    # "directorio X" only if X is not a conjunction / verb remnant
    m = re.search(
        r"(?:dir|carpeta|folder|directorio)\s+[\"']?([A-Za-z0-9._-]+)[\"']?",
        text,
        re.I,
    )
    if m and m.group(1).lower() not in stop:
        return m.group(1)
    return default


def plan_actions(user_text: str) -> list[dict[str, Any]]:
    """
    Return ordered tool calls for obvious local tasks.
    Empty list => let the LLM decide.
    """
    t = (user_text or "").strip()
    lower = t.lower()
    if not lower:
        return []

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
        and re.search(r"\b(archivo|archivos|directorio|carpeta|folder|contenido)\b", lower)
    )
    wants_write = bool(
        re.search(r"\b(escribe|escribir|write|crea|crear|guarda|guardar)\b", lower)
        and re.search(r"\b(archivo|file|txt|json|md)\b", lower)
    )
    wants_read = bool(
        re.search(r"\b(lee|leer|read|abre|abrir|muestra|mostrar)\b", lower)
        and re.search(r"\b(archivo|file|txt|json|md)\b", lower)
    )
    wants_shell = bool(
        re.search(r"\b(ejecuta|ejecutar|run|shell|cmd|powershell|comando)\b", lower)
    )

    # Explicit shell command in backticks
    m_cmd = re.search(r"`([^`]+)`", t)
    if wants_shell and m_cmd:
        return [{"name": "run_shell", "arguments": {"command": m_cmd.group(1).strip()}}]

    name = _name_from_text(t)

    # Compound: create then delete directory
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

    # Math
    m = re.search(r"(?:calcula|calculate|cuanto es|cuánto es)\s+(.+)$", lower)
    if m:
        expr = re.sub(r"[^0-9+\-*/().% ]", "", m.group(1))
        if expr.strip():
            return [{"name": "calc", "arguments": {"expression": expr.strip()}}]

    return []
