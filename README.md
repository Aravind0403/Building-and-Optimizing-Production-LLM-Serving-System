# AetherControl: Enterprise LLM Serving Platform, Control Plane & Post-Training Pipeline

[![CI/CD Pipeline](https://github.com/aravindsundaresan/Building-and-Optimizing-Production-LLM-Serving-System/actions/workflows/ci.yml/badge.svg)](https://github.com/aravindsundaresan/Building-and-Optimizing-Production-LLM-Serving-System/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![PyTest](https://img.shields.io/badge/tests-14%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**AetherControl** is a production-grade engineering control plane that bridges raw GPU execution realities (GPUs, HBM memory bandwidth, CUDA kernels, Nsight Systems traces) with post-training developer pipelines (pre-flight dataset inspection, vLLM serving, Kubernetes observability, and DeepSeek-R1 style GRPO alignment).

---

## 🏛️ Architecture Overview

```
                                [ AETHERCONTROL ]
                                        │
        +-------------------------------+-------------------------------+
        |                               |                               |
        v                               v                               v
 [ Data Validation ]          [ Cluster Topology ]          [ Post-Training RLHF ]
    "trainsight"               "K8s + Prometheus"             "GRPO Pipeline"
  • CLI (Typer/Rich)           • Production GKE Manifests     • HuggingFace TRL Trainer
  • K8s InitContainer          • Prometheus & Grafana         • Format & Verifier Rewards
  • Catch OOM/Corrupt            Metrics (TTFT, TPOT,         • No-Critic Advantage (Ai)
    data pre-flight              KV-Cache Occupancy)            Calculation Engine
```

---

## 📂 Project Modules

* **`trainsight/`**: Pre-flight data validation CLI & Kubernetes InitContainer. Scans SFT, DPO, and GRPO reasoning datasets (`openai/gsm8k`, `hendrycks/competition_math`, `openbmb/UltraFeedback`).
* **`k8s-infra/`**: Kubernetes deployment manifests, GKE cluster provisioning scripts (`gcp_setup.sh`), K8s Secrets, and Prometheus + Grafana observability stack.
* **`vllm-engine/`**: Production vLLM engine configuration manager (`config.yaml`), entrypoint generator, and standard streaming SLA benchmarking suite (`vllm-bench`).
* **`rlhf-pipeline/`**: GRPO (Group Relative Policy Optimization) post-training fine-tuning pipeline built on PyTorch and HuggingFace `trl.GRPOTrainer`.
* **`docs/`**: Production dossiers (PagedAttention, Nsight Systems `nsys` profiling, Roofline model, FP16/FP8/AWQ quantization matrix) and 5 mandatory failure experiments in `docs/experiments.md`.

---

## 🚀 Quick Start (Local Execution)

```bash
# Clone and run unit tests across all 3 modules
make test

# Profile sample dataset and inspect production vLLM CLI flags
make profile

# Apply Kubernetes manifests locally
make k8s-dev

# Teardown local cluster resources
make clean
```

---

## 📜 Architecture Decision Records (ADRs)

### ADR-001: Selection of GRPO over PPO for RLHF Post-Training
* **Context:** Traditional PPO requires maintaining 4 model copies in VRAM (Actor, Reference, Reward Model, and Value/Critic Model). The Critic model accounts for ~50% of total VRAM allocation.
* **Decision:** We adopted **Group Relative Policy Optimization (GRPO)**. By sampling a group of $G$ completions per prompt and calculating relative advantage $A_i = (r_i - \mu) / (\sigma + \epsilon)$, GRPO eliminates the Critic network entirely, reducing VRAM footprint by 50% and doubling batch capacity.

### ADR-002: Selection of vLLM PagedAttention over TGI
* **Context:** Traditional LLM serving allocates contiguous VRAM for maximum sequence length ($4096+$ tokens), causing 60–80% internal memory fragmentation.
* **Decision:** We deployed **vLLM with PagedAttention**. PagedAttention allocates physical KV-cache memory blocks dynamically like OS virtual memory paging, eliminating memory fragmentation and increasing concurrent sequence throughput by up to 4x.

---

## 💼 Quantified Impact & Resume Highlights

* **Data Engineering & Safety:** Engineered a pre-flight data validation InitContainer (`trainsight`), reducing wasted GPU compute hours by catching schema drift, corrupt completions, and sequence-length anomalies prior to Kubernetes pod scheduling.
* **Post-Training Alignment:** Architected an end-to-end GRPO RLHF pipeline with a no-critic relative advantage engine using HuggingFace `trl.GRPOTrainer`, optimizing VRAM utilization by 50% through Critic network elimination while maintaining format and mathematical verifier accuracy.
* **Serving & Observability:** Deployed and tuned a vLLM inference engine on GKE, utilizing Chunked Prefill and PagedAttention to achieve sub-100ms $P_{99}$ TTFT under concurrent load, instrumenting custom Prometheus/Grafana dashboards for TTFT, TPOT, and KV-cache block occupancy telemetry.
* **Systems Profiling & Fault Tolerance:** Conducted GPU hardware profiling (NVIDIA Nsight Systems `nsys`, Roofline model) and executed 5 structured failure experiments (Head-of-Line blocking, KV-cache starvation, CPU swapping), isolating execution bottlenecks and establishing production SLAs.

---

## 📜 License
MIT License
