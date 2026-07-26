<div align="center">

# LOCAL UNCENSORED AGENT

**EN** Offline · Uncensored · Yours  
**ES** Local · Sin censura · Tuyo  
**中文** 本地 · 无审查 · 属于你

[![Python 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Wheel-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Metal](https://img.shields.io/badge/Apple-Metal-000000?style=for-the-badge&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![GGUF](https://img.shields.io/badge/GGUF-llama.cpp-F05340?style=for-the-badge)](https://github.com/ggerganov/llama.cpp)
[![License: GPL-2.0](https://img.shields.io/badge/License-GPL--2.0--only-blue?style=for-the-badge)](./LICENSE)

```text
perception → reasoning → memory → tool-use → action
```

[English](#english) · [Español](#español) · [中文](#中文) · [Install](#quick-start--arranque-rápido--快速开始) · [Architecture](#architecture--arquitectura--架构) · [Thanks](#thanks--gracias--致谢)

</div>

---

## English

A **fully local** AI agent that runs on your machine — **Windows / Linux / macOS**.  
No cloud. No API keys. No moralizing refusals. Open-weight **uncensored** GGUFs only.

It detects your GPU (CUDA / Metal / CPU), picks a model that fits, downloads it on first boot, and talks through a modular pipeline with tools + memory.

| | |
|---|---|
| **Offline** | Works after the first model download |
| **Uncensored** | Aligned Instruct models are excluded from auto-select |
| **Smart pick** | VRAM / unified memory → best GGUF |
| **Tools** | Shell, files, calc — extend in 5 lines |
| **Memory** | Buffer + rolling summary + JSONL |

---

## Español

Un agente de IA **100% local** en tu PC — **Windows / Linux / macOS**.  
Sin nube. Sin API keys. Sin sermones. Solo GGUF **uncensored** de pesos abiertos.

Detecta tu GPU (CUDA / Metal / CPU), elige el modelo que cabe, lo descarga al arrancar y opera con pipeline modular: tools + memoria.

| | |
|---|---|
| **Offline** | Tras la 1ª descarga, sin internet |
| **Sin censura** | Instruct alineados fuera del auto-select |
| **Auto-pick** | VRAM / memoria unificada → mejor GGUF |
| **Tools** | Shell, archivos, calc — ampliables |
| **Memoria** | Buffer + resumen + JSONL |

---

## 中文

在你自己的电脑上运行的**完全本地** AI 智能体 — **Windows / Linux / macOS**。  
无需云端、无需 API Key、不做说教拒绝。只使用开源权重的 **uncensored** GGUF。

自动检测 GPU（CUDA / Metal / CPU），选择合适模型，首次启动自动下载，并通过模块化流水线完成工具调用与记忆。

| | |
|---|---|
| **离线** | 首次下载模型后即可离线使用 |
| **无审查** | 对齐/审查型 Instruct 不参与自动选择 |
| **智能选型** | 按显存/统一内存选择最佳 GGUF |
| **工具** | Shell、文件、计算器 — 可轻松扩展 |
| **记忆** | 缓冲 + 滚动摘要 + JSONL |

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
| Memory | Unified memory drives auto model pick (e.g. 8 GB Mac → small Qwen; 16 GB+ → Hermes) | La memoria unificada decide el modelo | 统一内存决定自动选型 |
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

First run downloads the GGUF automatically.

---

## Auto model select · Selección · 自动选型

`--model auto` (default):

1. Detect OS + backend  
2. Measure VRAM / unified memory / RAM  
3. Keep **uncensored-only** candidates  
4. Tune `n_ctx` · `n_gpu_layers` · `n_threads`  
5. Download if missing  

| Memory | Model | Quant |
|--------|-------|-------|
| ≤4–6 GB | [Qwen2.5-1.5B Uncensored ES](https://huggingface.co/mradermacher/Qwen2.5-1.5B-Uncensored_Neurotic_Spanish-GGUF) | Q4 / Q5 / Q8 |
| ~6 GB | Nous-Hermes-2-Mistral-7B-DPO | Q3_K_M |
| **~8 GB** | **Hermes-2-DPO** | **Q4_K_M** |
| ≥10 GB | Hermes-2-DPO | Q5_K_M |

```bash
python main.py --model auto
python main.py --model qwen25-1.5b-uncensored-es-q4
python smoke_hello.py qwen25-1.5b-uncensored-es-q4
```

REPL: `/status` · `/clear` · `/exit`

OOM / 显存不足:

```bash
python main.py --n-ctx 3072 --n-gpu-layers 28
```

---

## Architecture · Arquitectura · 架构

```text
┌─────────────┐   ┌────────────┐   ┌──────────┐   ┌───────────┐   ┌────────┐
│ perception  ├──►│ reasoning  ├──►│  memory  ├──►│ tool-use  ├──►│ action │
└─────────────┘   └────────────┘   └──────────┘   └───────────┘   └────────┘
                         ▲
                         │  llama.cpp  ·  GGUF  ·  ChatML
```

```text
main.py                 CLI
install_deps.py         CUDA / Metal / CPU installer
download_model.py       HF fetch
setup_windows.ps1       Win 3.12 + venv + CUDA wheel
setup_macos.sh          macOS 3.12 + Metal (arm64) / CPU (Intel)
smoke_hello.py          hello-world smoke test
agent/
  gpu.py                device detect + picker
  download.py           auto GGUF download
  perception.py
  reasoning.py
  memory.py
  tools.py
  action.py
  inference.py
  pipeline.py
  config.py             MODEL_CATALOG (uncensored-first)
models/                 *.gguf (gitignored)
data/memory.jsonl       persistent memory
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

Swap model without breaking the pipeline:

```python
from pathlib import Path
agent.swap_model("nous-hermes-2-mistral-7b-dpo", Path("models/....gguf"))
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
