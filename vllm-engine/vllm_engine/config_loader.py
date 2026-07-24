from pathlib import Path
from typing import List, Dict, Any
import yaml
from pydantic import BaseModel, Field


class VLLMConfig(BaseModel):
    """Pydantic Schema for Industry-Standard vLLM Engine Configuration."""
    model: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct", description="HuggingFace model repository ID")
    host: str = Field(default="0.0.0.0", description="Serving bind host")
    port: int = Field(default=8000, description="Serving bind port")
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=1.0, description="VRAM percentage reserved for model & KV-cache")
    max_model_len: int = Field(default=4096, gt=0, description="Maximum sequence length limit")
    block_size: int = Field(default=16, description="PagedAttention KV-cache block size in tokens")
    max_num_seqs: int = Field(default=256, gt=0, description="Max active sequence batch size")
    enable_chunked_prefill: bool = Field(default=True, description="Enable chunked prefill to prevent decode TPOT spikes")
    max_num_batched_tokens: int = Field(default=2048, description="Max tokens per prefill chunk")
    enable_prefix_caching: bool = Field(default=True, description="Enable Radix Tree prompt prefix caching")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Number of GPUs for Tensor Parallelism")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Number of GPUs for Pipeline Parallelism")

    def to_cli_args(self) -> List[str]:
        """Converts configuration options into exact vLLM entrypoint CLI flags."""
        args = [
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--block-size", str(self.block_size),
            "--max-num-seqs", str(self.max_num_seqs),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--pipeline-parallel-size", str(self.pipeline_parallel_size),
        ]
        if self.enable_chunked_prefill:
            args.extend(["--enable-chunked-prefill", "True"])
            args.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        return args


def load_vllm_config(config_path: Path) -> VLLMConfig:
    """Reads YAML config and returns a validated VLLMConfig object."""
    if not config_path.exists():
        raise FileNotFoundError(f"vLLM config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    return VLLMConfig(**raw_data)
