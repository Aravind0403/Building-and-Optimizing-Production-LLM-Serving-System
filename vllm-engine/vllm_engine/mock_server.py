import time
import json
import asyncio
import random
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse

app = FastAPI(title="Mock vLLM OpenAI API & Prometheus Server")

# Metrics state
metrics_state = {
    "requests_total": 0,
    "requests_running": 0,
    "requests_waiting": 0,
    "gpu_cache_usage": 0.15,
    "cpu_cache_usage": 0.05,
    "preemptions_total": 0,
    "prompt_tokens_total": 0,
    "generation_tokens_total": 0,
    "ttft_sum": 0.0,
    "tpot_sum": 0.0,
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    # Simulate dynamic KV cache fluctuations
    if metrics_state["requests_running"] > 0:
        metrics_state["gpu_cache_usage"] = min(0.95, 0.20 + (metrics_state["requests_running"] * 0.12))
    else:
        metrics_state["gpu_cache_usage"] = 0.08

    prometheus_text = f"""# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running {metrics_state['requests_running']}

# HELP vllm:num_requests_waiting Number of requests waiting in queue.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting {metrics_state['requests_waiting']}

# HELP vllm:gpu_cache_usage_perc GPU KV-cache memory usage percentage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc {metrics_state['gpu_cache_usage']:.3f}

# HELP vllm:cpu_cache_usage_perc CPU KV-cache memory usage percentage.
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc {metrics_state['cpu_cache_usage']:.3f}

# HELP vllm:num_preemptions_total Total number of request preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total {metrics_state['preemptions_total']}

# HELP vllm:prompt_tokens_total Total prompt tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total {metrics_state['prompt_tokens_total']}

# HELP vllm:generation_tokens_total Total generation tokens produced.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total {metrics_state['generation_tokens_total']}
"""
    return Response(content=prometheus_text, media_type="text/plain")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    model = data.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 30)

    metrics_state["requests_total"] += 1
    metrics_state["requests_running"] += 1

    prompt_tokens = 32
    metrics_state["prompt_tokens_total"] += prompt_tokens

    async def event_generator():
        try:
            # Simulate Prefill TTFT delay (~30-60ms)
            await asyncio.sleep(random.uniform(0.03, 0.06))

            words = ["vLLM", "PagedAttention", "continuous", "batching", "brings", "high", "throughput", "serving", "to", "production", "GPU", "clusters."]
            tokens_generated = 0

            for i in range(min(max_tokens, len(words))):
                token_text = words[i] + " "
                chunk = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None if i < len(words) - 1 else "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                tokens_generated += 1
                metrics_state["generation_tokens_total"] += 1
                # Simulate Decode TPOT delay (~15-25ms/token)
                await asyncio.sleep(random.uniform(0.015, 0.025))

            yield "data: [DONE]\n\n"
        finally:
            metrics_state["requests_running"] = max(0, metrics_state["requests_running"] - 1)

    if stream:
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        metrics_state["requests_running"] = max(0, metrics_state["requests_running"] - 1)
        return JSONResponse({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "vLLM serving platform ready."}}]
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
