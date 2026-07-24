# Learning Dossier 04: Kubernetes vLLM Deployment & Observability Architecture

> **Folder Path**: `docs/dossier/04_kubernetes_vllm_observability.md`  
> **Session Topic**: vLLM Kubernetes Deployment Manifests, Prometheus Metrics Scraping, and Grafana Dashboards  
> **Date**: July 21, 2026  

---

## 1. What is the Kubernetes Deployment Architecture for vLLM?

In **AetherControl**, vLLM is deployed as a Kubernetes Deployment paired with Prometheus scraping annotations and `trainsight` pre-flight InitContainer checks.

```
+-------------------------------------------------------------------------------+
|                             KUBERNETES POD                                    |
|                                                                               |
|  [ InitContainer: trainsight-validator ]                                      |
|        │ (Runs dataset quality inspection on CPU)                             |
|        └──► PASS (Exit 0)                                                     |
|                 │                                                             |
|                 v                                                             |
|  [ Container: vllm-container (Port 8000) ]                                    |
|        ├── Serving Qwen2.5-1.5B via OpenAI API Server                         |
|        ├── PagedAttention & Prefix Caching ENABLED                            |
|        └── Metrics Endpoint: http://0.0.0.0:8000/metrics                      |
|                 │                                                             |
|                 ▼                                                             |
|  [ Prometheus Scraper ] ───► [ Grafana Dashboard (TTFT, KV-Cache, TPOT) ]    |
+-------------------------------------------------------------------------------+
```

---

## 2. Key vLLM Prometheus Metrics & What They Measure

| Prometheus Metric Name | Unit | What It Measures & Why It Matters |
| :--- | :--- | :--- |
| `vllm:time_to_first_token_seconds_bucket` | Seconds (Histogram) | **Time-To-First-Token (TTFT)**: Measures the prefill phase latency. High TTFT indicates large prompt context or prefill compute bottlenecks. |
| `vllm:time_per_output_token_seconds_bucket` | Seconds (Histogram) | **Time-per-Output-Token (TPOT)**: Measures the decode phase latency per generated token. |
| `vllm:gpu_cache_usage_perc` | Percentage ($0.0 - 1.0$) | **KV-Cache Memory Occupancy %**: Measures how much of the allocated PagedAttention block pool is currently filled with active request KV-tensors. |
| `vllm:num_preemptions_total` | Counter | **Total Preemptions**: Increments when vLLM runs out of free KV-cache blocks and is forced to preempt (evict/re-compute) active requests. |
| `vllm:avg_prompt_throughput_toks_per_s` | Tokens/sec | **Prefill Throughput**: Rate of prompt processing across the engine. |
| `vllm:avg_generation_throughput_toks_per_s` | Tokens/sec | **Decode Throughput**: Rate of output token generation across the engine. |

---

## 3. Kubernetes Health & Readiness Probes

To ensure zero-downtime serving and prevent broken pods from receiving user traffic:

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 15
```

- **Readiness Probe**: Queries `/health`. Kubernetes only routes HTTP ingress traffic to the pod after model weights and PagedAttention KV-cache pools are fully initialized in VRAM.
- **Liveness Probe**: Detects CUDA deadlocks or engine crashes, automatically restarting the pod if `/health` fails 5 times continuously.
