"""Cross-platform device detection (Windows / Linux / macOS) + model picker."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Literal

from agent.config import MODEL_CATALOG

Backend = Literal["cuda", "metal", "cpu"]
OsName = Literal["windows", "linux", "darwin", "other"]


@dataclass
class DeviceInfo:
    os_name: OsName
    name: str
    vram_mb: int
    total_ram_mb: int
    backend: Backend
    index: int = 0
    source: str = "unknown"
    arch: str = ""

    @property
    def has_accelerator(self) -> bool:
        return self.backend in ("cuda", "metal") and self.vram_mb > 0


@dataclass
class ModelSelection:
    model_key: str
    device: DeviceInfo
    n_ctx: int
    n_batch: int
    n_gpu_layers: int
    n_threads: int
    max_prompt_tokens: int
    flash_attn: bool
    reason: str


# Back-compat alias used by older call sites
GpuInfo = DeviceInfo


def detect_os() -> OsName:
    sysname = platform.system().lower()
    if sysname == "windows":
        return "windows"
    if sysname == "linux":
        return "linux"
    if sysname == "darwin":
        return "darwin"
    return "other"


def _total_ram_mb() -> int:
    # psutil-free detection
    try:
        if hasattr(os, "sysconf"):
            if "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        pass

    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5)
            return int(int(out.strip()) / (1024 * 1024))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys / (1024 * 1024))
        except (AttributeError, OSError, ValueError):
            pass

    return 8192  # conservative default


def _detect_nvidia() -> DeviceInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None

    best: DeviceInfo | None = None
    ram = _total_ram_mb()
    os_name = detect_os()
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            vram = int(float(parts[2]))
        except ValueError:
            continue
        cand = DeviceInfo(
            os_name=os_name,
            name=name,
            vram_mb=vram,
            total_ram_mb=ram,
            backend="cuda",
            index=idx,
            source="nvidia-smi",
            arch=platform.machine(),
        )
        if best is None or cand.vram_mb > best.vram_mb:
            best = cand
    return best


def _detect_apple_metal() -> DeviceInfo | None:
    if platform.system() != "Darwin":
        return None
    ram = _total_ram_mb()
    chip = platform.processor() or platform.machine()
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
        ).strip()
        if out:
            chip = out
    except (OSError, subprocess.SubprocessError):
        pass

    # Unified memory: usable fraction for GGUF + KV (leave OS headroom)
    # Apple Silicon: Metal path in llama.cpp
    is_apple_silicon = platform.machine().lower() in {"arm64", "aarch64"}
    usable = int(ram * (0.55 if is_apple_silicon else 0.35))
    backend: Backend = "metal" if is_apple_silicon else "cpu"
    return DeviceInfo(
        os_name="darwin",
        name=chip or "Apple",
        vram_mb=usable,
        total_ram_mb=ram,
        backend=backend,
        source="sysctl-unified-memory",
        arch=platform.machine(),
    )


def detect_device() -> DeviceInfo:
    """
    Detect best local accelerator:
    - NVIDIA CUDA (Windows/Linux, rare on Mac)
    - Apple Metal (macOS Apple Silicon, unified memory budget)
    - CPU fallback using system RAM budget
    """
    os_name = detect_os()
    ram = _total_ram_mb()

    nvidia = _detect_nvidia()
    if nvidia is not None:
        return nvidia

    if os_name == "darwin":
        apple = _detect_apple_metal()
        if apple is not None:
            return apple

    # CPU-only: use ~40% of system RAM as model budget
    return DeviceInfo(
        os_name=os_name,
        name=f"CPU ({platform.processor() or platform.machine() or 'generic'})",
        vram_mb=int(ram * 0.40),
        total_ram_mb=ram,
        backend="cpu",
        source="system-ram",
        arch=platform.machine(),
    )


def detect_gpu() -> DeviceInfo:
    """Alias kept for callers expecting detect_gpu()."""
    return detect_device()


def recommended_n_threads(device: DeviceInfo | None = None) -> int:
    cpus = os.cpu_count() or 4
    device = device or detect_device()
    if device.backend == "cpu":
        return max(2, cpus)
    return max(2, min(8, cpus))


def _budget_mb(device: DeviceInfo) -> int:
    mem = device.vram_mb
    if mem <= 0:
        return 0
    if device.backend == "cuda":
        headroom = 1800 if mem < 10000 else 2200
        return max(0, mem - headroom)
    if device.backend == "metal":
        # Unified memory already discounted in detect; small extra KV reserve
        return max(0, mem - 1024)
    # CPU RAM budget already fractional
    return max(0, mem - 512)


def _ctx_for_budget(mem_mb: int, weight_mb: int, backend: Backend) -> tuple[int, int]:
    leftover = max(0, mem_mb - weight_mb - (900 if backend != "cpu" else 400))
    if leftover >= 3500:
        return 8192, 7000
    if leftover >= 2500:
        return 6144, 5200
    if leftover >= 1800:
        return 4096, 3200
    if leftover >= 1200:
        return 3072, 2400
    return 2048, 1600


def select_model_for_gpu(gpu: DeviceInfo | None = None) -> ModelSelection:
    """Choose UNCENSORED open-weight GGUF + knobs from detected device."""
    from agent.config import PREFER_UNCENSORED

    device = gpu or detect_device()
    budget = _budget_mb(device)
    mem = device.vram_mb

    def fits(meta: dict[str, Any]) -> bool:
        weight = int(meta.get("weight_mb", 99999))
        min_vram = int(meta.get("min_vram_mb", weight + 1800))
        if device.backend == "cpu":
            return weight <= budget
        return weight <= budget and mem >= min_vram * 0.85

    def collect(*, require_uncensored: bool) -> list[tuple[int, str, dict[str, Any]]]:
        out: list[tuple[int, str, dict[str, Any]]] = []
        for key, meta in MODEL_CATALOG.items():
            if not meta.get("auto_select", False):
                continue
            if str(meta.get("repo_id", "")).startswith("local/"):
                continue
            if require_uncensored and not meta.get("uncensored", False):
                continue
            if not fits(meta):
                continue
            weight = int(meta.get("weight_mb", 99999))
            unc = 1_000_000 if meta.get("uncensored") else 0
            # Prefer truly low-refusal models when uncensored mode is on
            risk = int(meta.get("refusal_risk", 50))
            score = (
                unc
                + (100 - risk) * 100_000
                + int(meta.get("quality_score", 0)) * 10_000
                + weight
            )
            out.append((score, key, meta))
        return out

    candidates = collect(require_uncensored=PREFER_UNCENSORED)
    if not candidates and PREFER_UNCENSORED:
        candidates = collect(require_uncensored=False)

    n_threads = recommended_n_threads(device)

    if not candidates:
        fallbacks = [
            (int(m.get("weight_mb", 99999)), k, m)
            for k, m in MODEL_CATALOG.items()
            if m.get("auto_select")
            and m.get("uncensored", False)
            and not str(m.get("repo_id", "")).startswith("local/")
        ] or [
            (int(m.get("weight_mb", 99999)), k, m)
            for k, m in MODEL_CATALOG.items()
            if m.get("auto_select") and not str(m.get("repo_id", "")).startswith("local/")
        ]
        if not fallbacks:
            raise RuntimeError("No auto-selectable models in MODEL_CATALOG")
        fallbacks.sort(key=lambda x: x[0])
        _, key, meta = fallbacks[0]
        weight = int(meta["weight_mb"])
        n_gpu_layers = 0 if device.backend == "cpu" else 20
        return ModelSelection(
            model_key=key,
            device=device,
            n_ctx=2048,
            n_batch=256,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            max_prompt_tokens=1600,
            flash_attn=False,
            reason=(
                f"OS={device.os_name} backend={device.backend} device={device.name} "
                f"mem={mem} MiB. Tight fit -> uncensored '{key}' (~{weight} MB)."
            ),
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, key, meta = candidates[0]
    weight = int(meta["weight_mb"])
    n_ctx, max_prompt = _ctx_for_budget(mem, weight, device.backend)
    n_batch = 512 if mem >= 6000 and device.backend != "cpu" else 256

    if device.backend == "cpu":
        n_gpu_layers = 0
        flash = False
    elif device.backend == "metal":
        n_gpu_layers = -1 if mem >= 3500 else 24
        flash = False
    else:
        n_gpu_layers = -1 if mem >= 4500 else 24
        flash = mem >= 6000

    reason = (
        f"OS={device.os_name} backend={device.backend} device={device.name} "
        f"mem={mem} MiB budget~{budget} MiB -> UNCENSORED model={key} "
        f"({meta.get('filename')}, ~{weight} MB, license={meta.get('license', 'n/a')}). "
        f"{meta.get('reason', '')}"
    )
    return ModelSelection(
        model_key=key,
        device=device,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        max_prompt_tokens=max_prompt,
        flash_attn=flash,
        reason=reason,
    )


def apply_selection(config: Any, selection: ModelSelection) -> None:
    config.model_key = selection.model_key
    config.n_ctx = selection.n_ctx
    config.n_batch = selection.n_batch
    config.n_gpu_layers = selection.n_gpu_layers
    config.n_threads = selection.n_threads
    config.max_prompt_tokens = selection.max_prompt_tokens
    config.flash_attn = selection.flash_attn


def parse_vram_override(text: str) -> int | None:
    m = re.fullmatch(r"(\d+)\s*(mb|mi[bB])?", text.strip(), re.I)
    if not m:
        return None
    return int(m.group(1))


def install_hint(device: DeviceInfo | None = None) -> str:
    device = device or detect_device()
    if device.backend == "cuda":
        return (
            "pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cu121"
        )
    if device.backend == "metal":
        return 'CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir'
    return "pip install llama-cpp-python"
