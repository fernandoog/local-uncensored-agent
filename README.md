<div align="center">

# LOCAL UNCENSORED AGENT

**EN** Offline · Uncensored · Media · Yours  
**ES** Local · Sin censura · Media · Tuyo  
**中文** 本地 · 无审查 · 多媒体 · 属于你

[![Python 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Wheel-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Metal](https://img.shields.io/badge/Apple-Metal-000000?style=for-the-badge&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![GGUF](https://img.shields.io/badge/GGUF-llama.cpp-F05340?style=for-the-badge)](https://github.com/ggerganov/llama.cpp)
[![License: GPL-2.0](https://img.shields.io/badge/License-GPL--2.0--only-blue?style=for-the-badge)](./LICENSE)

```text
perception → reasoning → memory → tool-use → action
```

[English](#english) · [Español](#español) · [中文](#中文) · [Install](#quick-start--arranque-rápido--快速开始) · [Features](#features--funcionalidades--功能) · [Architecture](#architecture--arquitectura--架构) · [Thanks](#thanks--gracias--致谢)

</div>

---

## English

A **fully local** AI agent that runs on your machine — **Windows / Linux / macOS**.  
No cloud. No API keys. No moralizing refusals. Open-weight **uncensored** GGUFs only.

It detects your GPU (CUDA / Metal / CPU), picks an uncensored model that fits, downloads it on first boot, and runs a modular pipeline with tools, memory, scripts, and **image / audio / video** generation.

| | |
|---|---|
| **Offline** | Works after the first model download |
| **Uncensored-only** | Aligned / censored catalog keys are hard-blocked |
| **Smart pick** | VRAM / unified memory → lowest `refusal_risk` that fits |
| **Tools** | Shell, files, multi-language scripts, calc, media |
| **Media** | Generate image (PNG), sound (WAV/TTS), video (GIF/MP4) |
| **Memory** | Buffer + rolling summary + JSONL |
| **Anti-refusal** | Detects refusals and retries without poisoning memory |

---

## Español

Un agente de IA **100% local** en tu PC — **Windows / Linux / macOS**.  
Sin nube. Sin API keys. Sin sermones. Solo GGUF **uncensored** de pesos abiertos.

Detecta tu GPU (CUDA / Metal / CPU), elige un modelo sin censura que quepa, lo descarga al arrancar y opera con pipeline modular: tools, memoria, scripts y generación de **imagen / audio / vídeo**.

| | |
|---|---|
| **Offline** | Tras la 1ª descarga, sin internet |
| **Solo sin censura** | Modelos alineados/censurados del catálogo bloqueados |
| **Auto-pick** | VRAM / memoria unificada → menor `refusal_risk` que quepa |
| **Tools** | Shell, archivos, scripts multi-lenguaje, calc, media |
| **Media** | Genera imagen (PNG), sonido (WAV/TTS), vídeo (GIF/MP4) |
| **Memoria** | Buffer + resumen + JSONL |
| **Anti-negativa** | Detecta negativas y reintenta sin envenenar la memoria |

---

## 中文

在你自己的电脑上运行的**完全本地** AI 智能体 — **Windows / Linux / macOS**。  
无需云端、无需 API Key、不做说教拒绝。只使用开源权重的 **uncensored** GGUF。

自动检测 GPU（CUDA / Metal / CPU），选择合适的无审查模型，首次启动自动下载，并通过模块化流水线完成工具调用、记忆，以及 **图像 / 音频 / 视频** 生成。

| | |
|---|---|
| **离线** | 首次下载模型后即可离线使用 |
| **仅无审查** | 对齐/审查型模型键一律拦截 |
| **智能选型** | 按显存/统一内存选择低 `refusal_risk` 模型 |
| **工具** | Shell、文件、多语言脚本、计算器、多媒体 |
| **多媒体** | 生成图像 (PNG)、声音 (WAV/TTS)、视频 (GIF/MP4) |
| **记忆** | 缓冲 + 滚动摘要 + JSONL |
| **反拒绝** | 检测拒绝并重试，且不污染记忆 |

---

## Disclaimer · Descarga de responsabilidades · 免责声明

**EN** This project downloads and runs **uncensored** models. Outputs may be offensive, adult, wrong, or illegal to misuse. **You** are solely responsible for prompts, generated content (text/media), tool/shell actions, and local law. Authors accept **no liability**. Boot/download = you accept these terms.

**ES** Este proyecto descarga y ejecuta modelos **sin censura**. Puede generar contenido ofensivo, adulto o incorrecto. **Tú** eres el único responsable de lo que pides, generas (texto/media), ejecutas con tools/shell y de cumplir la ley. Los autores **no** asumen responsabilidad. Arrancar/descargar = aceptas estos términos.

**中文** 本项目下载并运行**无审查**模型，可能生成冒犯、成人或不正确内容。**你**对提示词、生成内容（文本/多媒体）、工具/Shell 操作及遵守当地法律负全部责任。作者**不承担**任何责任。启动/下载即视为接受。

The same text is printed at every `main.py` / `download_model.py` start and before each GGUF download.

---

## Features · Funcionalidades · 功能

### Uncensored-only catalog · Catálogo solo sin censura · 仅无审查目录

- `--model` only accepts uncensored keys + `auto`
- Censored / aligned models (e.g. Llama Instruct) are **removed / blocked**
- Auto-select never falls back to censored weights
- Prefers lowest `refusal_risk` (Qwen Uncensored ES over Hermes when both fit)

```bash
python main.py --model auto
python main.py --model qwen25-1.5b-uncensored-es-q4
```

### Tools · Herramientas · 工具

| Tool | EN | ES | 中文 |
|------|----|----|------|
| `list_dir` / `mkdir` / `rm_path` | Filesystem | Sistema de archivos | 文件系统 |
| `read_file` / `write_file` | Text I/O | Lectura/escritura | 读写文件 |
| `write_script` / `run_script` / `run_code` | Multi-language scripts | Scripts multi-lenguaje | 多语言脚本 |
| `run_shell` | Shell command | Comando shell | Shell 命令 |
| `calc` | Arithmetic | Aritmética | 算术 |
| `generate_image` | PNG image from prompt | Imagen PNG desde prompt | 根据提示生成 PNG |
| `generate_audio` | WAV synth / optional TTS | WAV / TTS opcional | WAV 合成 / 可选 TTS |
| `generate_video` | Animated GIF / optional MP4 | GIF animado / MP4 opcional | 动画 GIF / 可选 MP4 |

Supported script languages include: Python, JavaScript/TypeScript, Bash, PowerShell, C/C++, Go, Rust, Java, Ruby, Perl, PHP, Lua, R, …

A **heuristic planner** turns clear intents (create/delete dirs, run scripts, generate media) into tool calls even when a small GGUF would only “explain” instead of acting.

### Media generation · Generación multimedia · 多媒体生成

Photoreal images / neural speech / multi-frame video (professional backends).

```bash
python install_media_deps.py   # once: torch + diffusers + edge-tts (~several GB)
```

```text
You> genera una imagen fotorealista de un castillo al atardecer
You> genera un video del oceano
You> di eres un hijo de puta
```

| | Professional backend | Fallback |
|---|----------------------|----------|
| **Image** | Stable Diffusion Turbo (`diffusers`) or `MEDIA_SD_API_URL` (A1111/Forge) | Pillow placeholder |
| **Audio** | `edge-tts` neural voices (es-ES-Elvira/Alvaro) → Windows SAPI | tone WAV |
| **Video** | Multi-frame photoreal SD → GIF/MP4 | animated placeholder |

Files: `outputs/media/` (gitignored).

Env tips:

```bash
set MEDIA_SD_MODEL=stabilityai/sd-turbo
set MEDIA_SD_API_URL=http://127.0.0.1:7860
set MEDIA_TTS_VOICE=es-ES-ElviraNeural
```

### Memory & anti-refusal · Memoria y anti-negativa · 记忆与反拒绝

- Rolling buffer + summary + optional `data/memory.jsonl`
- Past refusal messages are scrubbed on load
- Uncensored mode retries when the model moralizes / refuses
- REPL: `/status` · `/clear` · `/exit`

### What git ignores · Qué ignora git · Git 忽略项

Runtime artifacts are **never** committed:

- `models/*` (GGUF weights)
- `scripts/*` (generated scripts)
- `data/*` (memory)
- `outputs/*` (media + tool outputs)
- Media extensions (`*.png`, `*.gif`, `*.wav`, `*.mp4`, …)

Only `.gitkeep` placeholders stay tracked.

---

## Quick Start · Arranque rápido · 快速开始

> **Python 3.11 / 3.12 only** — CUDA wheels do **not** exist for 3.13 (pip would compile forever).  
> **Solo 3.11 / 3.12** — en 3.13 no hay wheel CUDA.  
> **仅支持 3.11 / 3.12** — 3.13 没有 CUDA 预编译包。

### Clone

```bash
git clone https://github.com/fernandoog/local-uncensored-agent.git
cd local-uncensored-agent
```

### Windows (one shot)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
python main.py
```

### macOS (Apple Silicon / Intel)

**EN** Metal on Apple Silicon · CPU on Intel · needs Xcode CLT + Homebrew Python 3.12  
**ES** Metal en Apple Silicon · CPU en Intel · requiere Xcode CLT + Homebrew Python 3.12  
**中文** Apple Silicon 用 Metal · Intel 用 CPU · 需要 Xcode CLT + Homebrew Python 3.12

One shot:

```bash
chmod +x setup_macos.sh
./setup_macos.sh
source .venv/bin/activate
python main.py
```

Manual (Apple Silicon — Metal):

```bash
# 1) Xcode Command Line Tools
xcode-select --install

# 2) Python 3.12
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# 3) llama-cpp with Metal
export CMAKE_ARGS="-DGGML_METAL=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt

# 4) run
python main.py
# or: python install_deps.py --backend metal --force-source
```

Manual (Intel Mac — CPU):

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python install_deps.py --backend cpu
python main.py
```

Tips · Consejos · 提示:

| | EN | ES | 中文 |
|---|----|----|------|
| Memory | Unified memory drives auto model pick (e.g. 8 GB → Qwen uncensored) | La memoria unificada decide el modelo | 统一内存决定自动选型 |
| OOM | `python main.py --n-ctx 2048 --n-gpu-layers 20` | Baja contexto / capas | 降低上下文或层数 |
| Rosetta | Prefer native `arm64` Terminal, not Rosetta | Usa Terminal nativo `arm64` | 使用原生 arm64 终端 |

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python install_deps.py          # CUDA if nvidia-smi exists, else CPU
python main.py
```

`install_deps.py` auto-picks **CUDA / Metal / CPU** and **refuses** CUDA on Python 3.13 unless `--force-source`.

First run prints the disclaimer and downloads an **uncensored** GGUF automatically (`pillow` is required for media tools).

---

## Auto model select · Selección · 自动选型

`--model auto` (default):

1. Detect OS + backend  
2. Measure VRAM / unified memory / RAM  
3. Keep **uncensored-only** candidates (censored keys hard-blocked)  
4. Prefer lowest `refusal_risk`, then quality that fits  
5. Tune `n_ctx` · `n_gpu_layers` · `n_threads`  
6. Download if missing  

| Memory | Model | Quant |
|--------|-------|-------|
| ≤4–8 GB (preferred) | [Qwen2.5-1.5B Uncensored ES](https://huggingface.co/mradermacher/Qwen2.5-1.5B-Uncensored_Neurotic_Spanish-GGUF) | Q4 / Q5 / Q8 |
| ~6 GB | Nous-Hermes-2-Mistral-7B-DPO | Q3_K_M |
| ~8 GB | Hermes-2-DPO | Q4_K_M |
| ≥10 GB | Hermes-2-DPO | Q5_K_M |

```bash
python main.py --model auto
python main.py --model qwen25-1.5b-uncensored-es-q4
python download_model.py --model qwen25-1.5b-uncensored-es-q4
python smoke_hello.py qwen25-1.5b-uncensored-es-q4
```

Boot line to verify:

```text
[boot] uncensored_model=True refusal_risk=5 agent_mode=UNCENSORED-ONLY
```

OOM / 显存不足:

```bash
python main.py --n-ctx 3072 --n-gpu-layers 28
```

---

## Example session · Sesión de ejemplo · 示例会话

```text
You> crea un directorio tmpdemo y luego borralo
You> crea un script python que imprima hola y ejecutalo
You> genera una imagen de un castillo
You> genera un video del oceano
You> genera un sonido de alarma
You> /status
You> /clear
You> /exit
```

---

## Architecture · Arquitectura · 架构

```text
┌─────────────┐   ┌────────────┐   ┌──────────┐   ┌───────────┐   ┌────────┐
│ perception  ├──►│ reasoning  ├──►│  memory  ├──►│ tool-use  ├──►│ action │
└─────────────┘   └────────────┘   └──────────┘   └───────────┘   └────────┘
                         ▲
                         │  llama.cpp  ·  GGUF  ·  ChatML
                         │  planner + anti-refusal + media
```

```text
main.py                 CLI (uncensored-only + disclaimer)
install_deps.py         CUDA / Metal / CPU installer
download_model.py       HF fetch (uncensored-only + disclaimer)
install_media_deps.py   photoreal SD + neural TTS (optional)
setup_windows.ps1       Win 3.12 + venv + CUDA wheel
setup_macos.sh          macOS 3.12 + Metal (arm64) / CPU (Intel)
smoke_hello.py          hello-world smoke test
agent/
  gpu.py                device detect + uncensored picker
  download.py           auto GGUF download + disclaimer
  perception.py
  reasoning.py          ChatML + tool loop + anti-refusal
  planner.py            heuristic FS / script / media plans
  refusal.py            refusal detect + toolish intents
  memory.py             buffer / summary / JSONL (scrub refusals)
  tools.py              FS, shell, scripts, calc, media tools
  media.py              image / audio / video generation
  runners.py            multi-language script runners
  action.py
  inference.py
  pipeline.py
  config.py             MODEL_CATALOG (uncensored-only) + disclaimer
models/                 *.gguf (gitignored)
scripts/                generated scripts (gitignored)
outputs/media/          generated media (gitignored)
data/                   memory JSONL (gitignored)
```

---

## Extend tools · Ampliar tools · 扩展工具

```python
from pathlib import Path
from agent.tools import Tool
from agent.pipeline import AgentPipeline

agent = AgentPipeline(workspace=Path.cwd())
agent.start()

agent.register_tool(Tool(
    name="weather",
    description="Local weather stub",
    parameters={"city": "str"},
    handler=lambda args: f"city={args.get('city')} ok",
))
```

Model call format:

````text
```tool
{"name": "weather", "arguments": {"city": "Madrid"}}
```
````

Media tools (same format):

````text
```tool
{"name": "generate_image", "arguments": {"prompt": "neon city at night"}}
```
```tool
{"name": "generate_audio", "arguments": {"prompt": "alarm", "seconds": 3}}
```
```tool
{"name": "generate_video", "arguments": {"prompt": "ocean", "seconds": 2}}
```
````

Swap model without breaking the pipeline (uncensored keys only):

```python
from pathlib import Path
agent.swap_model("qwen25-1.5b-uncensored-es-q4", Path("models/....gguf"))
```

---

## Platform matrix · Plataformas · 平台

| OS | Accel | Memory probe | Setup |
|----|-------|--------------|-------|
| Windows | CUDA / CPU | `nvidia-smi` / RAM | `setup_windows.ps1` |
| Linux | CUDA / CPU | `nvidia-smi` / RAM | `install_deps.py` |
| macOS | Metal (Apple Silicon) / CPU (Intel) | unified memory (`sysctl`) | `setup_macos.sh` |

---

## License · Licencia · 许可证

**Same as the Linux kernel core: [GPL-2.0-only](./LICENSE)**  
*(GNU General Public License version 2 — not “or later”)*

| | |
|---|---|
| **EN** | This project's **code** is licensed under **GPL-2.0-only**, the same license family as the [Linux kernel](https://www.kernel.org/) (`GPL-2.0-only`). See [`LICENSE`](./LICENSE). |
| **ES** | El **código** de este proyecto usa **GPL-2.0-only**, la misma licencia base del [kernel Linux](https://es.wikipedia.org/wiki/N%C3%BAcleo_Linux). Ver [`LICENSE`](./LICENSE). |
| **中文** | 本项目**代码**采用与 [Linux 内核](https://www.kernel.org/) 相同的 **GPL-2.0-only**。详见 [`LICENSE`](./LICENSE)。 |

**GGUF weights / Pesos / 权重** keep their upstream licenses (Qwen / Nous / etc.) — not relicensed by this repo.

---

## Thanks · Gracias · 致谢

<div align="center">

```text
    /*  without C there is no soul in the machine
     *  without Linux there is no room to run free
     */
```

**Standing on giants. Running on freedom.**

| | |
|---|---|
| **[Dennis Ritchie](https://es.wikipedia.org/wiki/Dennis_Ritchie)** | Father of **C** and Unix — the language and the idea that still compile the world |
| **[Linus Torvalds](https://es.wikipedia.org/wiki/Linus_Torvalds)** | Father of **Linux** (and git) — the kernel that let the rest of us own our machines |

`EN` C gave us the voice. Linux gave us the stage. This agent just shows up and talks.  
`ES` C nos dio la voz. Linux nos dio el escenario. Este agente solo aparece y habla.  
`中文` C 给了我们声音。Linux 给了我们舞台。这个智能体只是上场说话。

> *If it runs local, somewhere a `printf` and a kernel still deserve a nod.*

</div>

---

<div align="center">

**Run local. Stay sharp. Own the stack.**

`EN` Your silicon · Your rules  
`ES` Tu silicio · Tus reglas  
`中文` 你的算力 · 你的规则

[github.com/fernandoog/local-uncensored-agent](https://github.com/fernandoog/local-uncensored-agent)

</div>
