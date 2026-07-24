import time
import json
import random
import asyncio
from typing import List, Dict, Any, Tuple
import requests
from vllm_engine.benchmarks.benchmark_serving import RequestBenchmarkResult


from pathlib import Path

def generate_synthetic_prompts(num_prompts: int, prompt_tokens: int = 128) -> List[str]:
    """Generates synthetic prompts with approximately desired token length."""
    base_words = ["algorithm", "optimization", "pagedattention", "cuda", "vllm", "kernel", "throughput", "latency", "prefill", "decode", "batching"]
    prompts = []
    for i in range(num_prompts):
        words_needed = prompt_tokens  # ~1 word per token for synthetic testing
        words = [random.choice(base_words) for _ in range(words_needed)]
        prompt = f"System test query #{i+1}: " + " ".join(words)
        prompts.append(prompt)
    return prompts


def load_dataset_prompts(dataset_path: Path, max_samples: int = 100) -> List[str]:
    """Loads real prompts from an SFT/DPO dataset JSONL file."""
    prompts = []
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                prompt = data.get("prompt") or data.get("instruction")
                if isinstance(data.get("messages"), list) and len(data["messages"]) > 0:
                    prompt = data["messages"][0].get("content", "")
                if prompt:
                    prompts.append(prompt)
                if len(prompts) >= max_samples:
                    break
            except json.JSONDecodeError:
                continue

    return prompts if prompts else generate_synthetic_prompts(max_samples)



def send_streaming_request(
    api_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 128,
    timeout: float = 60.0
) -> RequestBenchmarkResult:
    """Sends a single streaming request to standard OpenAI chat completions endpoint."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0
    }

    start_time = time.perf_counter()
    first_token_time: float = 0.0
    tokens_received = 0
    full_text = ""

    try:
        response = requests.post(api_url, headers=headers, json=payload, stream=True, timeout=timeout)
        if response.status_code != 200:
            return RequestBenchmarkResult(
                prompt_len=len(prompt) // 4,
                output_len=0,
                success=False,
                error_message=f"HTTP {response.status_code}: {response.text[:100]}",
                start_time=start_time,
                e2e_latency=time.perf_counter() - start_time
            )

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time == 0.0:
                                first_token_time = time.perf_counter()
                            tokens_received += 1
                            full_text += content
                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        e2e_latency = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time > 0 else e2e_latency

        if tokens_received > 1:
            tpot = (end_time - first_token_time) / (tokens_received - 1)
        else:
            tpot = 0.0

        return RequestBenchmarkResult(
            prompt_len=len(prompt) // 4,
            output_len=tokens_received,
            success=True,
            start_time=start_time,
            ttft=ttft,
            tpot=tpot,
            e2e_latency=e2e_latency
        )

    except Exception as e:
        return RequestBenchmarkResult(
            prompt_len=len(prompt) // 4,
            output_len=0,
            success=False,
            error_message=str(e),
            start_time=start_time,
            e2e_latency=time.perf_counter() - start_time
        )
