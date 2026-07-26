"""Professional media generation: photoreal images, neural TTS, multi-frame video.

Backend priority (auto):
  Image: MEDIA_SD_API_URL → diffusers (SD-Turbo / photoreal) → Pillow fallback
  Audio: edge-tts (neural) → Windows SAPI → pyttsx3 → tone WAV
  Video: photoreal frame sequence (diffusers/API) → GIF/MP4

Env:
  MEDIA_IMAGE_BACKEND=auto|diffusers|api|pillow
  MEDIA_SD_API_URL=http://127.0.0.1:7860
  MEDIA_SD_MODEL=Lykon/dreamshaper-8   # SD1.5 uncensored-capable (not SD-Turbo)
  MEDIA_SD_DEVICE=auto|cuda|cpu
  MEDIA_TTS_BACKEND=auto|edge|sapi|pyttsx3|tone
  MEDIA_TTS_VOICE=es-ES-ElviraNeural
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Any
from urllib import error, request

from agent.config import ROOT

MEDIA_DIR = ROOT / "outputs" / "media"
_SD_PIPE = None
_SD_PIPE_KEY = None

# SD-Turbo (SD2.1) is NSFW-filtered. Default to SD1.5 DreamShaper (uncensored-capable).
DEFAULT_SD_MODEL = os.environ.get("MEDIA_SD_MODEL", "Lykon/dreamshaper-8")
PHOTO_POS = (
    "photorealistic, professional photography, ultra detailed, 8k uhd, "
    "sharp focus, natural lighting, realistic skin texture, cinematic, "
    "high quality, masterpiece"
)
PHOTO_NEG = (
    "cartoon, anime, illustration, painting, drawing, 3d render, cgi, "
    "lowres, blurry, deformed, ugly, watermark, text, logo, oversaturated, "
    "extra limbs, bad anatomy, censored, mosaic, bar censorship, clothing artifacts"
)
NSFW_HINT = re.compile(
    r"(?is)\b(nsfw|nude|nuda|desnud|porno|porn|xxx|erotic|erotica|sex|"
    r"explicit|genital|breast|tetas|culo|ass|naked)\b"
)


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


def enhance_image_prompt(prompt: str) -> str:
    p = (prompt or "").strip()
    if not p:
        return PHOTO_POS
    low = p.lower()
    extras = PHOTO_POS
    if NSFW_HINT.search(low):
        extras = (
            f"{PHOTO_POS}, uncensored, nsfw, explicit, detailed anatomy, "
            "no clothes, no censorship, no mosaic"
        )
    return f"{p}, {extras}"


def _require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow required: pip install pillow") from exc
    return Image, ImageDraw, ImageFont


def _fit_font(ImageFont: Any, size: int):
    for name in (
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pick_torch_device() -> str:
    pref = (os.environ.get("MEDIA_SD_DEVICE") or "auto").lower()
    if pref in {"cuda", "cpu", "mps"}:
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_sd_pipe(model_id: str | None = None):
    """Lazy-load diffusers txt2img pipeline (cached)."""
    global _SD_PIPE, _SD_PIPE_KEY
    model_id = model_id or DEFAULT_SD_MODEL
    device = _pick_torch_device()
    key = f"{model_id}|{device}"
    if _SD_PIPE is not None and _SD_PIPE_KEY == key:
        return _SD_PIPE, device

    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise RuntimeError(
            "Professional image backend needs: pip install -r requirements-media.txt"
        ) from exc

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    print(
        f"[media] loading UNCENSORED photoreal model={model_id} device={device} "
        "(first run downloads weights; safety_checker disabled)"
    )
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if device != "cpu" else None,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # Belt-and-suspenders: some pipelines keep a checker attr
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    try:
        if device == "cuda":
            # Keep VRAM free for the LLM when possible
            pipe.enable_model_cpu_offload()
        elif device == "mps":
            pipe.to("mps")
        else:
            pipe.to("cpu")
    except Exception:
        pipe.to(device if device != "cuda" else "cpu")

    _SD_PIPE = pipe
    _SD_PIPE_KEY = key
    return pipe, device


def generate_image_diffusers(
    prompt: str,
    *,
    out: Path | None = None,
    width: int = 768,
    height: int = 512,
    steps: int | None = None,
) -> Path:
    pipe, device = _get_sd_pipe()
    prompt_full = enhance_image_prompt(prompt)
    width = max(256, min(int(width), 1024))
    height = max(256, min(int(height), 1024))
    # DreamShaper / SD1.5: more steps; turbo models: few steps
    model = DEFAULT_SD_MODEL.lower()
    if steps is None:
        steps = 4 if "turbo" in model or "lightning" in model else 28
    guidance = 0.0 if "turbo" in model else 7.0
    kwargs: dict[str, Any] = {
        "prompt": prompt_full,
        "negative_prompt": PHOTO_NEG,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
    }
    # Some turbo pipelines reject negative_prompt / odd sizes
    try:
        result = pipe(**kwargs)
    except TypeError:
        kwargs.pop("negative_prompt", None)
        result = pipe(**kwargs)
    except Exception:
        # Fallback safer size for turbo
        kwargs["width"] = 512
        kwargs["height"] = 512
        try:
            result = pipe(**kwargs)
        except TypeError:
            kwargs.pop("negative_prompt", None)
            result = pipe(**kwargs)

    img = result.images[0]
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
    base = (api_url or os.environ.get("MEDIA_SD_API_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("MEDIA_SD_API_URL not set")
    payload = {
        "prompt": enhance_image_prompt(prompt),
        "negative_prompt": PHOTO_NEG,
        "steps": 28,
        "width": width,
        "height": height,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M Karras",
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base}/sdapi/v1/txt2img",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    images = body.get("images") or []
    if not images:
        raise RuntimeError("SD API returned no images")
    import base64

    raw = base64.b64decode(images[0].split(",", 1)[-1])
    path = out or _stamp("img", prompt, "png")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path.resolve()


def generate_image_pillow(
    prompt: str,
    *,
    out: Path | None = None,
    width: int = 768,
    height: int = 512,
) -> Path:
    """Last-resort placeholder (NOT photoreal)."""
    Image, ImageDraw, ImageFont = _require_pillow()
    width = max(64, min(int(width), 2048))
    height = max(64, min(int(height), 2048))
    seed = _seed(prompt)
    img = Image.new("RGB", (width, height), (20, 24, 32))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(height - 1, 1)
        col = (int(20 + 40 * t), int(24 + 30 * t), int(32 + 50 * t))
        draw.line([(0, y), (width, y)], fill=col)
    font = _fit_font(ImageFont, max(14, width // 40))
    draw.text((16, 16), "FALLBACK (not photoreal)", fill=(255, 180, 80), font=font)
    draw.text((16, 48), "pip install -r requirements-media.txt", fill=(220, 220, 220), font=font)
    draw.text((16, height - 40), (prompt or "")[:80], fill=(240, 240, 240), font=font)
    path = out or _stamp("img", prompt, "png")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path.resolve()


def generate_image(
    prompt: str,
    *,
    out: str | Path | None = None,
    width: int = 768,
    height: int = 512,
    backend: str | None = None,
) -> dict[str, Any]:
    prompt = (prompt or "").strip() or "cinematic portrait"
    out_path = Path(out) if out else None
    backend = (backend or os.environ.get("MEDIA_IMAGE_BACKEND") or "auto").lower()
    errors: list[str] = []
    order: list[str]
    if backend == "auto":
        order = []
        if os.environ.get("MEDIA_SD_API_URL"):
            order.append("api")
        order.extend(["diffusers", "pillow"])
    else:
        order = [backend]

    for name in order:
        try:
            if name == "api":
                path = generate_image_api(prompt, out=out_path, width=width, height=height)
            elif name == "diffusers":
                path = generate_image_diffusers(prompt, out=out_path, width=width, height=height)
            elif name == "pillow":
                path = generate_image_pillow(prompt, out=out_path, width=width, height=height)
            else:
                continue
            return {
                "ok": True,
                "kind": "image",
                "backend": name,
                "quality": "photoreal" if name in {"api", "diffusers"} else "fallback",
                "path": str(path),
                "prompt": prompt,
                "enhanced_prompt": enhance_image_prompt(prompt) if name != "pillow" else prompt,
            }
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
    raise RuntimeError("Image generation failed: " + " | ".join(errors))


# ----- Audio (professional neural TTS) -----


def generate_audio_wav(
    prompt: str,
    *,
    out: Path | None = None,
    seconds: float = 4.0,
    sample_rate: int = 22050,
) -> Path:
    seconds = max(0.5, min(float(seconds), 30.0))
    sr = int(sample_rate)
    n = int(sr * seconds)
    seed = _seed(prompt)
    base_hz = 110 + (seed % 400)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        step = int(t * 4) % 8
        freqs = [base_hz * (1 + 0.12 * ((step + k) % 5)) for k in range(3)]
        env = min(1.0, t * 8) * max(0.0, 1.0 - max(0.0, t - (seconds - 0.25)) / 0.25)
        sample = sum((0.5 / (j + 1)) * math.sin(2 * math.pi * f * t) for j, f in enumerate(freqs))
        val = int(max(-1.0, min(1.0, sample * 0.35 * env)) * 32767)
        frames += struct.pack("<h", val)
    path = out or _stamp("audio", prompt, "wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(frames))
    return path.resolve()


def _edge_voice(voice: str, text: str) -> str:
    explicit = os.environ.get("MEDIA_TTS_VOICE")
    if explicit:
        return explicit
    v = (voice or "auto").lower()
    low = (text or "").lower()
    female = v in {"female", "mujer", "woman", "f"} or any(
        w in low for w in ("mujer", "woman", "female", "chica")
    )
    male = v in {"male", "hombre", "man", "m"} or any(
        w in low for w in ("hombre", "man", "male", "chico")
    )
    # Spanish neural voices (high quality)
    if female:
        return "es-ES-ElviraNeural"
    if male:
        return "es-ES-AlvaroNeural"
    return "es-ES-ElviraNeural"


def generate_audio_edge_tts(
    text: str,
    *,
    out: Path | None = None,
    voice: str = "auto",
) -> Path:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts not installed: pip install edge-tts") from exc

    path = out or _stamp("tts", text, "mp3")
    if path.suffix.lower() not in {".mp3", ".wav"}:
        path = path.with_suffix(".mp3")
    path.parent.mkdir(parents=True, exist_ok=True)
    voice_id = _edge_voice(voice, text)

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(str(path))

    asyncio.run(_run())
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("edge-tts produced empty file")
    return path.resolve()


def generate_audio_tts_sapi(
    text: str,
    *,
    out: Path | None = None,
    voice: str = "auto",
) -> Path:
    import platform
    import subprocess

    if platform.system().lower() != "windows":
        raise RuntimeError("SAPI TTS is Windows-only")

    path = out or _stamp("tts", text, "wav")
    if path.suffix.lower() != ".wav":
        path = path.with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)

    gender = "NotSet"
    v = (voice or "auto").lower()
    lower = text.lower()
    if v in {"female", "mujer", "woman", "f"} or any(
        w in lower for w in ("mujer", "woman", "female", "chica")
    ):
        gender = "Female"
    elif v in {"male", "hombre", "man", "m"} or any(
        w in lower for w in ("hombre", "man", "male", "chico")
    ):
        gender = "Male"

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
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        Path(script).unlink(missing_ok=True)
    if proc.returncode != 0 or not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"SAPI TTS failed: {(proc.stderr or proc.stdout or '').strip()}")
    return path.resolve()


def generate_audio_tts_pyttsx3(
    text: str,
    *,
    out: Path | None = None,
    voice: str = "auto",
) -> Path:
    import pyttsx3

    path = out or _stamp("tts", text, "wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.save_to_file(text, str(path))
    engine.runAndWait()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("pyttsx3 produced empty file")
    return path.resolve()


def generate_audio(
    prompt: str,
    *,
    out: str | Path | None = None,
    seconds: float = 4.0,
    mode: str | None = None,
    voice: str = "auto",
) -> dict[str, Any]:
    prompt = (prompt or "").strip() or "tone"
    out_path = Path(out) if out else None
    mode = (mode or "auto").lower()
    prefer = (os.environ.get("MEDIA_TTS_BACKEND") or "auto").lower()

    if mode == "tone":
        path = generate_audio_wav(prompt, out=out_path, seconds=seconds)
        return {
            "ok": True,
            "kind": "audio",
            "backend": "tone",
            "quality": "synth",
            "path": str(path),
            "prompt": prompt,
            "voice": voice,
            "seconds": seconds,
        }

    # Professional speech order
    order: list[str] = []
    if prefer != "auto":
        order = [prefer]
    else:
        order = ["edge", "sapi", "pyttsx3"]
    if mode == "tts":
        # force speech backends only first
        pass

    errors: list[str] = []
    for name in order:
        try:
            if name == "edge":
                path = generate_audio_edge_tts(prompt, out=out_path, voice=voice)
            elif name == "sapi":
                path = generate_audio_tts_sapi(prompt, out=out_path, voice=voice)
            elif name == "pyttsx3":
                path = generate_audio_tts_pyttsx3(prompt, out=out_path, voice=voice)
            else:
                continue
            return {
                "ok": True,
                "kind": "audio",
                "backend": name,
                "quality": "neural" if name == "edge" else "system",
                "path": str(path),
                "prompt": prompt,
                "voice": voice,
                "seconds": None,
            }
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    path = generate_audio_wav(prompt, out=out_path, seconds=seconds)
    return {
        "ok": True,
        "kind": "audio",
        "backend": f"tone(fallback: {' | '.join(errors)})",
        "quality": "synth",
        "path": str(path),
        "prompt": prompt,
        "voice": voice,
        "seconds": seconds,
    }


# ----- Video (photoreal multi-frame) -----


def generate_video(
    prompt: str,
    *,
    out: str | Path | None = None,
    seconds: float = 2.0,
    fps: int = 8,
    width: int = 512,
    height: int = 320,
) -> dict[str, Any]:
    prompt = (prompt or "").strip() or "cinematic scene"
    seconds = max(0.5, min(float(seconds), 4.0))
    fps = max(4, min(int(fps), 8))
    # Cap frames: each photoreal frame is expensive
    n_frames = max(4, min(int(seconds * fps), 6))
    width = max(256, min(int(width), 640))
    height = max(256, min(int(height), 640))

    frames = []
    backend = "pillow"
    quality = "fallback"
    errors: list[str] = []

    # Try photoreal frames via diffusers/API
    motion = [
        "wide establishing shot",
        "slow camera push in",
        "slight pan left, cinematic",
        "close detail, shallow depth of field",
        "tracking shot, photorealistic",
        "final hero frame, dramatic light",
    ]
    try:
        use_api = bool(os.environ.get("MEDIA_SD_API_URL"))
        for i in range(n_frames):
            frame_prompt = f"{prompt}, {motion[i % len(motion)]}, frame {i+1}"
            tmp = _stamp("frame", f"{prompt}_{i}", "png")
            if use_api:
                p = generate_image_api(frame_prompt, out=tmp, width=width, height=height)
                backend = "api"
            else:
                p = generate_image_diffusers(frame_prompt, out=tmp, width=width, height=height)
                backend = "diffusers"
            Image, _, _ = _require_pillow()
            frames.append(Image.open(p).convert("RGB"))
            quality = "photoreal"
    except Exception as exc:
        errors.append(str(exc))
        # Fallback animated gradient
        Image, ImageDraw, ImageFont = _require_pillow()
        font = _fit_font(ImageFont, max(12, width // 28))
        frames = []
        for fi in range(n_frames):
            img = Image.new("RGB", (width, height), (18, 20, 28))
            draw = ImageDraw.Draw(img)
            phase = fi / n_frames
            for y in range(height):
                t = (y / max(height - 1, 1) + phase) % 1.0
                draw.line(
                    [(0, y), (width, y)],
                    fill=(int(20 + 60 * t), int(24 + 40 * t), int(40 + 80 * t)),
                )
            draw.text((12, 12), "FALLBACK video", fill=(255, 180, 80), font=font)
            draw.text((12, height - 28), prompt[:50], fill=(240, 240, 240), font=font)
            frames.append(img)
        backend = f"pillow({errors[0][:80] if errors else 'fallback'})"
        quality = "fallback"

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

    # Cleanup temp frame PNGs we created under media dir with frame_ prefix — optional
    return {
        "ok": True,
        "kind": "video",
        "backend": f"{backend}/{used}",
        "quality": quality,
        "path": str(path.resolve()),
        "prompt": prompt,
        "seconds": seconds,
        "fps": fps,
        "frames": len(frames),
        "hint": None
        if quality == "photoreal"
        else "Install professional backends: pip install -r requirements-media.txt",
    }


def generate_media(kind: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
    kind = (kind or "").lower().strip()
    if kind in {"image", "img", "imagen", "picture", "foto", "photo"}:
        return generate_image(prompt, **kwargs)
    if kind in {"audio", "sound", "sonido", "wav", "musica", "music", "tts", "voz"}:
        if kind in {"tts", "voz"}:
            kwargs.setdefault("mode", "tts")
        return generate_audio(prompt, **kwargs)
    if kind in {"video", "vid", "gif", "mp4", "clip"}:
        return generate_video(prompt, **kwargs)
    raise ValueError(f"unknown media kind '{kind}'")
