"""Agent pipeline: perception → reasoning → memory → tool-use → action."""
from __future__ import annotations

from pathlib import Path

from agent.action import ActionLayer, ActionResult
from agent.config import AgentConfig, MEMORY_FILE, ensure_dirs
from agent.inference import InferenceEngine
from agent.memory import MemoryStore
from agent.perception import Perception
from agent.reasoning import ReasoningEngine
from agent.tools import Tool, ToolRegistry


class AgentPipeline:
    def __init__(self, config: AgentConfig | None = None, workspace: Path | None = None) -> None:
        ensure_dirs()
        self.config = config or AgentConfig()
        self.perception = Perception()
        self.engine = InferenceEngine(self.config.inference)
        self.tools = ToolRegistry(workspace=workspace or Path.cwd())
        self.memory = MemoryStore(
            summary_trigger=self.config.inference.summary_trigger_messages,
            keep_recent=self.config.inference.summary_keep_recent,
            persist_path=MEMORY_FILE,
        )
        self.action = ActionLayer()
        self.reasoner = ReasoningEngine(
            self.config, self.engine, self.memory, self.tools
        )

    def start(self, *, load_memory: bool = True, auto_download: bool = True) -> None:
        from agent.config import AUTO_MODEL
        from agent.gpu import detect_gpu, select_model_for_gpu

        if self.config.inference.model_key == AUTO_MODEL:
            selection = select_model_for_gpu(detect_gpu())
            self.config.inference.model_key = selection.model_key
            self.config.inference.n_ctx = selection.n_ctx
            self.config.inference.n_batch = selection.n_batch
            self.config.inference.n_gpu_layers = selection.n_gpu_layers
            self.config.inference.n_threads = selection.n_threads
            self.config.inference.max_prompt_tokens = selection.max_prompt_tokens
            self.config.inference.flash_attn = selection.flash_attn
            print(f"[select] {selection.reason}")

        self.engine.load(auto_download=auto_download)
        self.memory.summarizer = self.reasoner.summarize_chunk
        if load_memory:
            self.memory.load_persistent()

    def step(self, raw_input: str) -> ActionResult:
        perceived = self.perception.perceive(raw_input)
        if not perceived.text:
            return self.action.emit("")
        if perceived.text.lower() in {"/exit", "/quit", "exit", "quit"}:
            return ActionResult(text="__EXIT__", ok=True)
        if perceived.text.lower() == "/status":
            return self.action.emit(str(self.reasoner.diagnose()))
        if perceived.text.lower() == "/clear":
            self.memory.clear(wipe_disk=False)
            return self.action.emit("memory buffer cleared")
        try:
            reply = self.reasoner.think(perceived.text)
            return self.action.emit(reply)
        except Exception as exc:
            return self.action.emit_error(exc)

    def register_tool(self, tool: Tool) -> None:
        """Extend tool-use without touching the ChatML / inference pipeline."""
        self.tools.register(tool)

    def swap_model(self, model_key: str, model_path: Path | None = None) -> None:
        """Hot-swap GGUF; perception/memory/tools stay intact."""
        from agent.config import MODEL_CATALOG

        if model_key not in MODEL_CATALOG:
            raise KeyError(f"Unknown model_key. Known: {list(MODEL_CATALOG)}")
        self.engine.unload()
        self.config.inference.model_key = model_key
        self.config.inference.model_path = model_path
        self.engine.load()
