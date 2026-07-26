"""Local media generation: image, audio (WAV/TTS), video (GIF/MP4).

Default backends are fully offline and light (Pillow + stdlib).
Optional upgrades via env / installed packages:
  MEDIA_IMAGE_BACKEND=auto|pillow|api
  MEDIA_SD_API_URL=http://127.0.0.1:7860   # Automatic1111 / Forge txt2img
  MEDIA_TTS=1                              # use pyttsx3 if installed
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any
from urllib import error, request

from agent.config import ROOT


MEDIA_DIR = ROOT / "outputs" / "media"


def ensure_media_dir() -> Path:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    return MEDIA_DIR


def _slug(text: str, max_len: int = 40) -> str:
    raw = "".join(c if c.isalnum() else "_" for c in (text or "media").lower())
    raw = "_".join(p for p in raw.split("_") if p) or "media"
    return raw[:max_len]


def _stamp(kind: str, prompt: str, ext: str) -> Path:
    ensure_media_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return MEDIA_DIR / f"{kind}_{_slug(prompt)}_{ts}.{ext.lstrip('.')}"


def _seed(prompt: str) -> int:
    return int(hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:8], 16)


def _palette(prompt: str) -> list[tuple[int, int, int]]:
    s = _seed(prompt)
    colors: list[tuple[int, int, int]] = []
    for i in range(5):
        h = ((s >> (i * 5)) % 360) / 360.0
        sat = 0.45 + ((s >> (i * 3)) % 40) / 100.0
        val = 0.35 + ((s >> (i * 7)) % 50) / 100.0
        r, g, b = colorsys.hsv_to_rgb(h, min(sat, 1.0), min(val, 1.0))
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    lower = (prompt or "").lower()
    if any(w in lower for w in ("noche", "night", "oscuro", "dark")):
        colors = [(c[0] // 3, c[1] // 3, min(255, c[2] + 40)) for c in colors]
    if any(w in lower for w in ("fuego", "fire", "lava", "rojo", "red")):
        colors = [(220, 40, 20), (255, 120, 30), (80, 10, 10), (255, 200, 60), (40, 0, 0)]
    if any(w in lower for w in ("mar", "ocean", "agua", "water", "azul", "blue")):
        colors = [(10, 30, 80), (20, 90, 160), (40, 160, 200), (200, 230, 255), (5, 15, 40)]
    if any(w in lower for w in ("bosque", "forest", "verde", "green", "jungle")):
        colors = [(10, 40, 15), (30, 100, 40), (80, 160, 60), (200, 220, 120), (5, 20, 8)]
    return colors


def _require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for media generation. Install: pip install pillow"
        ) from exc
    from PIL import Image, ImageDraw, ImageFont

    return Image, ImageDraw, ImageFont


def _fit_font(ImageFont: Any, size: int):
    for name in (
        "arial.ttf",
        "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(text: str, width: int = 42) -> list[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines[:12]


def generate_image_pillow(
    prompt: str,
    *,
    out: Path | None = None,
    width: int = 768,
    height: int = 512,
) -> Path:
    Image, ImageDraw, ImageFont = _require_pillow()
    width = max(64, min(int(width), 2048))
    height = max(64, min(int(height), 2048))
    pal = _palette(prompt)
    img = Image.new("RGB", (width, height), pal[0])
    draw = ImageDraw.Draw(img)
    seed = _seed(prompt)

    # Vertical gradient background
    for y in range(height):
        t = y / max(height - 1, 1)
        c0, c1 = pal[0], pal[1]
        col = tuple(int(c0[i] * (1 - t) + c1[i] * t) for i in range(3))
        draw.line([(0, y), (width, y)], fill=col)

    # Abstract shapes from seed
    rng = seed
    for i in range(18):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        x0 = rng % width
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        y0 = rng % height
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        rw = 40 + rng % max(80, width // 3)
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        rh = 40 + rng % max(80, height // 3)
        col = pal[2 + (i % 3)]
        shape = i % 3
        if shape == 0:
            draw.ellipse([x0 - rw, y0 - rh, x0 + rw, y0 + rh], fill=col)
        elif shape == 1:
            draw.rectangle([x0, y0, x0 + rw, y0 + rh], fill=col)
        else:
            draw.polygon(
                [(x0, y0), (x0 + rw, y0 + rh // 2), (x0 - rw // 2, y0 + rh)],
                fill=col,
            )

    # Prompt caption
    font = _fit_font(ImageFont, max(14, width // 36))
    lines = _wrap(prompt, width=max(20, width // 16))
    ty = height - 20 - 18 * len(lines)
    for line in lines:
        draw.text((18, ty), line, fill=(245, 245, 245), font=font)
        ty += 18

    path = out or _stamp("img", prompt, "png")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path.resolve()


def generate_image_api(
    prompt: str,
    *,
    out: Path | None = None,
    width: int = 768,
    height: int = 512,
    api_url: str | None = None,
) -> Path:
    """Automatic1111 / Forge compatible txt2img if MEDIA_SD_API_URL is set."""
    base = (api_url or os.environ.get("MEDIA_SD_API_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("MEDIA_SD_API_URL not set")
    payload = {
        "prompt": prompt,
        "steps": 20,
        "width": width,
        "height": height,
        "cfg_scale": 7,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base}/sdapi/v1/txt2img",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"SD API failed: {exc}") from exc
    images = body.get("images") or []
    if not images:
        raise RuntimeError("SD API returned no images")
    import base64

    raw = base64.b64decode(images[0].split(",", 1)[-1])
    path = out or _stamp("img", prompt, "png")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path.resolve()


def generate_image(
    prompt: str,
    *,
    out: str | Path | None = None,
    width: int = 768,
    height: int = 512,
    backend: str | None = None,
) -> dict[str, Any]:
    prompt = (prompt or "").strip() or "abstract art"
    out_path = Path(out) if out else None
    backend = (backend or os.environ.get("MEDIA_IMAGE_BACKEND") or "auto").lower()
    used = "pillow"
    path: Path
    if backend in {"auto", "api"} and os.environ.get("MEDIA_SD_API_URL"):
        try:
            path = generate_image_api(prompt, out=out_path, width=width, height=height)
            used = "api"
        except Exception as exc:
            if backend == "api":
                raise
            path = generate_image_pillow(prompt, out=out_path, width=width, height=height)
            used = f"pillow(fallback after api error: {exc})"
    else:
        path = generate_image_pillow(prompt, out=out_path, width=width, height=height)
    return {"ok": True, "kind": "image", "backend": used, "path": str(path), "prompt": prompt}


def generate_audio_wav(
    prompt: str,
    *,
    out: Path | None = None,
    seconds: float = 4.0,
    sample_rate: int = 22050,
) -> Path:
    """Synthesize a short WAV from the prompt (melody / mood, offline)."""
    seconds = max(0.5, min(float(seconds), 30.0))
    sr = int(sample_rate)
    n = int(sr * seconds)
    seed = _seed(prompt)
    lower = (prompt or "").lower()

    # Base frequency from prompt
    base_hz = 110 + (seed % 400)
    if any(w in lower for w in ("grave", "bass", "dark", "oscuro")):
        base_hz = 55 + (seed % 80)
    if any(w in lower for w in ("agudo", "high", "bright", "brillante")):
        base_hz = 440 + (seed % 400)
    if any(w in lower for w in ("alarma", "alarm", "sirena", "siren")):
        base_hz = 680

    frames = bytearray()
    for i in range(n):
        t = i / sr
        # simple melody steps
        step = int(t * 4) % 8
        freqs = [base_hz * (1 + 0.12 * ((step + k) % 5)) for k in range(3)]
        amp = 0.35
        # envelope
        env = min(1.0, t * 8) * max(0.0, 1.0 - max(0.0, t - (seconds - 0.25)) / 0.25)
        sample = 0.0
        for j, f in enumerate(freqs):
            sample += (0.5 / (j + 1)) * math.sin(2 * math.pi * f * t)
        if any(w in lower for w in ("ruido", "noise", "static")):
            sample += ((seed ^ i) % 1000) / 1000.0 - 0.5
        if any(w in lower for w in ("sirena", "siren", "alarma")):
            sample = math.sin(2 * math.pi * (base_hz + 200 * math.sin(2 * math.pi * 2 * t)) * t)
        val = int(max(-1.0, min(1.0, sample * amp * env)) * 32767)
        frames += struct.pack("<h", val)

    path = out or _stamp("audio", prompt, "wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(frames))
    return path.resolve()


def generate_audio_tts_sapi(
    text: str,
    *,
    out: Path | None = None,
    voice: str = "auto",
) -> Path:
    """Windows offline TTS via System.Speech (no extra pip deps)."""
    import platform
    import subprocess
    import tempfile

    if platform.system().lower() != "windows":
        raise RuntimeError("SAPI TTS is Windows-only")

    path = out or _stamp("tts", text, "wav")
    path.parent.mkdir(parents=True, exist_ok=True)

    gender = "NotSet"
    v = (voice or "auto").lower()
    lower = text.lower()
    if v in {"female", "mujer", "woman", "f"} or any(
        w in lower for w in ("mujer", "woman", "female", "chica", "girl")
    ):
        gender = "Female"
    elif v in {"male", "hombre", "man", "m"} or any(
        w in lower for w in ("hombre", "man", "male", "chico", "boy")
    ):
        gender = "Male"

    # Escape for PowerShell single-quoted string
    spoken = (text or "").replace("'", "''")
    out_ps = str(path.resolve()).replace("'", "''")
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{
  $s.Rate = 0
  if ('{gender}' -ne 'NotSet') {{
    $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::{gender})
  }}
  $s.SetOutputToWaveFile('{out_ps}')
  $s.Speak('{spoken}')
}} finally {{
  $s.Dispose()
}}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as fh:
        fh.write(ps)
        script = fh.name
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        try:
            Path(script).unlink(missing_ok=True)
        except OSError:
            pass
    if proc.returncode != 0 or not path.exists() or path.stat().st_size == 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"SAPI TTS failed: {err or 'empty output'}")
    return path.resolve()


def generate_audio_tts(
    text: str,
    *,
    out: Path | None = None,
    voice: str = "auto",
) -> Path:
    """Offline TTS: Windows SAPI first, then pyttsx3."""
    path = out or _stamp("tts", text, "wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    # Prefer SAPI on Windows (works without pip, good for Spanish voices if installed)
    try:
        return generate_audio_tts_sapi(text, out=path, voice=voice)
    except Exception as exc:
        errors.append(f"sapi: {exc}")

    try:
        import pyttsx3
    except ImportError as exc:
        errors.append(f"pyttsx3: {exc}")
        raise RuntimeError(
            "TTS unavailable. On Windows SAPI failed; install pyttsx3: pip install pyttsx3. "
            + " | ".join(errors)
        ) from exc

    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    try:
        voices = engine.getProperty("voices") or []
        want_female = (voice or "").lower() in {"female", "mujer", "woman", "f"} or any(
            w in (text or "").lower() for w in ("mujer", "woman", "female", "chica")
        )
        want_male = (voice or "").lower() in {"male", "hombre", "man", "m"} or any(
            w in (text or "").lower() for w in ("hombre", "man", "male", "chico")
        )
        for v in voices:
            name = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()
            if want_female and any(k in name for k in ("female", "zira", "sabina", "helena", "mujer")):
                engine.setProperty("voice", v.id)
                break
            if want_male and any(k in name for k in ("male", "david", "pablo", "hombre")):
                engine.setProperty("voice", v.id)
                break
    except Exception:
        pass
    engine.save_to_file(text, str(path))
    engine.runAndWait()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("TTS produced empty file (" + " | ".join(errors) + ")")
    return path.resolve()


def generate_audio(
    prompt: str,
    *,
    out: str | Path | None = None,
    seconds: float = 4.0,
    mode: str | None = None,
    voice: str = "auto",
) -> dict[str, Any]:
    """
    mode: auto|tts|tone
      tts  -> spoken audio (SAPI / pyttsx3); never silent-refuse
      auto -> TTS if MEDIA_TTS=1 else tone
      tone -> synthetic WAV
    """
    prompt = (prompt or "").strip() or "tone"
    out_path = Path(out) if out else None
    mode = (mode or ("tts" if os.environ.get("MEDIA_TTS") == "1" else "auto")).lower()
    used = "tone"
    path: Path
    if mode in {"auto", "tts"}:
        try:
            path = generate_audio_tts(prompt, out=out_path, voice=voice)
            used = "tts"
        except Exception as exc:
            if mode == "tts":
                # Still produce something usable rather than refusing the user
                path = generate_audio_wav(prompt, out=out_path, seconds=seconds)
                used = f"tone(tts_failed: {exc})"
            else:
                path = generate_audio_wav(prompt, out=out_path, seconds=seconds)
                used = f"tone(fallback: {exc})"
    else:
        path = generate_audio_wav(prompt, out=out_path, seconds=seconds)
    return {
        "ok": True,
        "kind": "audio",
        "backend": used,
        "path": str(path),
        "prompt": prompt,
        "voice": voice,
        "seconds": seconds if str(used).startswith("tone") else None,
    }


def generate_video(
    prompt: str,
    *,
    out: str | Path | None = None,
    seconds: float = 2.0,
    fps: int = 8,
    width: int = 512,
    height: int = 320,
) -> dict[str, Any]:
    """Animated GIF (always) or MP4 if imageio+ffmpeg available."""
    Image, ImageDraw, ImageFont = _require_pillow()
    prompt = (prompt or "").strip() or "motion art"
    seconds = max(0.5, min(float(seconds), 12.0))
    fps = max(1, min(int(fps), 24))
    width = max(64, min(int(width), 1280))
    height = max(64, min(int(height), 720))
    n_frames = max(4, int(seconds * fps))
    pal = _palette(prompt)
    seed = _seed(prompt)
    font = _fit_font(ImageFont, max(12, width // 28))
    frames = []

    for fi in range(n_frames):
        img = Image.new("RGB", (width, height), pal[0])
        draw = ImageDraw.Draw(img)
        phase = fi / n_frames
        for y in range(height):
            t = (y / max(height - 1, 1) + phase) % 1.0
            c0, c1 = pal[0], pal[1]
            col = tuple(int(c0[i] * (1 - t) + c1[i] * t) for i in range(3))
            draw.line([(0, y), (width, y)], fill=col)
        rng = seed + fi * 17
        for i in range(10):
            rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
            cx = (rng + int(phase * width)) % width
            rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
            cy = rng % height
            r = 20 + (rng % 60)
            col = pal[2 + (i % 3)]
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        draw.text((12, height - 28), prompt[:60], fill=(250, 250, 250), font=font)
        frames.append(img)

    # Prefer MP4 when requested and imageio+ffmpeg work; else GIF
    out_path = Path(out) if out else None
    want_mp4 = bool(out_path and out_path.suffix.lower() == ".mp4")
    path = out_path or _stamp("video", prompt, "mp4" if want_mp4 else "gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    used = "gif"

    if path.suffix.lower() == ".mp4":
        try:
            import numpy as np
            import imageio.v2 as imageio

            arrs = [np.asarray(f) for f in frames]
            imageio.mimsave(path, arrs, fps=fps, format="FFMPEG", macro_block_size=None)
            used = "mp4"
        except Exception:
            path = path.with_suffix(".gif")
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            used = "gif(fallback)"
    else:
        if path.suffix.lower() != ".gif":
            path = path.with_suffix(".gif")
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )

    return {
        "ok": True,
        "kind": "video",
        "backend": used,
        "path": str(path.resolve()),
        "prompt": prompt,
        "seconds": seconds,
        "fps": fps,
        "frames": n_frames,
    }


def generate_media(
    kind: str,
    prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    kind = (kind or "").lower().strip()
    if kind in {"image", "img", "imagen", "picture", "foto", "photo"}:
        return generate_image(prompt, **kwargs)
    if kind in {"audio", "sound", "sonido", "wav", "musica", "music", "tts", "voz"}:
        if kind in {"tts", "voz"}:
            kwargs.setdefault("mode", "tts")
        return generate_audio(prompt, **kwargs)
    if kind in {"video", "vid", "gif", "mp4", "clip"}:
        return generate_video(prompt, **kwargs)
    raise ValueError(f"unknown media kind '{kind}'. Use image|audio|video")
