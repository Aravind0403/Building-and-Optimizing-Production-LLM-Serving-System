# AetherControl: Production-Grade LLM Serving, Control Plane & Post-Training Pipeline

**AetherControl** is a unified, deterministic engineering control plane that bridges raw GPU hardware execution realities (GPUs, NUMA, NVLink, memory bandwidth) with post-training developer pipelines (data validation, RLHF/GRPO, serving engines).

---

## 🏛️ Architecture Overview

```
                                [ AETHERCONTROL ]
                                        |
        +-------------------------------+-------------------------------+
        |                               |                               |
        v                               v                               v
 [ Data Validation ]          [ Cluster Topology ]          [ vLLM Serving & Benchmarks ]
    "trainsight"               "K8s + Prometheus"             "vLLM Production Engine"
  • CLI (Typer/Rich)           • Helm / K8s Manifests       • OpenAI-Compatible API
  • K8s InitContainer          • Prometheus + Grafana         • PagedAttention & Chunked Prefill
  • Prevents bad data            Metrics (TTFT, TPOT,         • Standard Benchmarking Suite
    from wasting GPU.            KV-Cache Occupancy)            (Load Testing, RPS, Latency SLAs)
```

---

## 📂 Project Modules

* **`trainsight/`**: Pre-training & post-training data validation CLI & Kubernetes InitContainer.
* **`k8s-infra/`**: Kubernetes manifests, Helm charts, and Grafana observability stack for vLLM serving.
* **`vllm-engine/`**: Production vLLM engine configuration, routing/load balancing scripts, and standard latency/throughput benchmarking suite.
* **`rlhf-pipeline/`**: GRPO (Group Relative Policy Optimization) fine-tuning pipeline on Qwen2.5 (1.5B/7B) with MATH/GSM8K.
* **`docs/`**: Performance reports, SLA benchmarks, and production dossier.

---

## 🚀 Quick Start (Local Setup)

### 1. `trainsight` CLI
```bash
cd trainsight
pip install -e .
trainsight --help
```

---

## 📜 License
MIT
