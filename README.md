# AetherControl: Enterprise LLM Serving Platform, Control Plane & Post-Training Pipeline

[![CI/CD Pipeline](https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![PyTest](https://img.shields.io/badge/tests-14%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![GCP GKE](https://img.shields.io/badge/GCP-GKE%20NVIDIA%20L4-orange)

**AetherControl** is a production-grade engineering control plane that bridges raw GPU execution realities (HBM memory bandwidth, CUDA kernels, Nsight Systems traces) with post-training developer pipelines (pre-flight dataset inspection, vLLM serving, Kubernetes observability, and DeepSeek-R1 style GRPO alignment).

---

## 🏛️ System Architecture

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
  • Fail-Fast Exit (code=1)      Metrics (TTFT, TPOT,         • No-Critic Advantage (Ai)
    & DVC Data Tracking          KV-Cache Occupancy)            Calculation Engine
```

---

## 📦 Core Component Breakdown

| Module | Location | Primary Command | Key Production Feature | Unit Tests |
| :--- | :--- | :--- | :--- | :--- |
| **`trainsight`** | `trainsight/` | `trainsight profile` | Pre-flight dataset linter & Kubernetes InitContainer guard. Detects CUDA OOM risks (`> 2048` tokens), sequence length variance ($\sigma > 0.75\mu$), and empty completions. | ✅ 4/4 Passed |
| **`vllm-engine`** | `vllm-engine/` | `vllm-bench benchmark` | Production engine config manager & SLA streaming benchmark generator ($P_{50}/P_{99}$ TTFT, TPOT, RPS). Tunes Chunked Prefill, PagedAttention, and Prefix Caching. | ✅ 3/3 Passed |
| **`k8s-infra`** | `k8s-infra/` | `make cloud-up` | Production GKE Spot GPU cluster manifests, Kubernetes Secrets (`secrets.yaml`), KEDA autoscaling (`keda-autoscaler.yaml`), DCGM exporter (`dcgm-exporter.yaml`), and Prometheus/Grafana stack. | ✅ Verified Live |
| **`rlhf-pipeline`** | `rlhf-pipeline/` | `rlhf-train train-steps` | DeepSeek-R1 style Group Relative Policy Optimization (GRPO) training pipeline using HuggingFace `trl.GRPOTrainer`, DeepSpeed ZeRO-3, and rule-based verifier rewards. | ✅ 7/7 Passed |

---

## 📊 Live Cloud Telemetry & Empirical Test Results

Empirical telemetry captured from deploying AetherControl on **Google Cloud Platform (GKE)**:

### 1. Serving Engine SLA Performance (`vllm-bench`)
```text
   vLLM Serving Performance & SLA Benchmark Results (50 Requests @ 8 Concurrency)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric                     ┃ Value          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Total Duration             │ 7.16 s         │
│ Success / Total Requests   │ 50 / 50 (100%) │
│ Request Throughput         │ 6.98 req/s     │
│ Output Token Throughput    │ 48.89 tokens/s │
│ TTFT (Prefill Latency) P50 │ 855.4 ms       │
│ TTFT (Prefill Latency) P99 │ 1074.9 ms      │
│ TPOT (Decode Latency) P50  │ 24.7 ms/token  │
│ TPOT (Decode Latency) P99  │ 43.0 ms/token  │
│ E2E Latency P99            │ 1.11 s         │
└────────────────────────────┴────────────────┘
```

### 2. Post-Training GRPO Telemetry Progression (20 Steps on GSM8K)
```text
📊 GRPO Fine-Tuning Progression & Telemetry Log
• Mean Reward (r_mean):     0.42 ──► 0.85 (+102% Reasoning Improvement)
• Format Reward (<think>):  0.23 ──► 0.92 (Strict tag compliance)
• Accuracy Reward (Math):   0.22 ──► 0.87 (Verifier precision)
• KL Divergence (D_KL):     0.0586 (Stable < 0.15 threshold, preventing model drift)
• GRPO Loss (L_grpo):       1.1934 ──► 0.1730 (Smooth convergence)
```

---

## 📚 Supported Public Production Datasets

TrainSight standardizes and validates datasets directly streamed from HuggingFace Hub:

| Dataset | Primary Use Case | Scale | Source |
| :--- | :--- | :--- | :--- |
| **GSM8K** | Math reasoning for GRPO training | 8.5K train / 1K test | `openai/gsm8k` |
| **MATH** | Competition math reasoning | 7.5K train / 5K test | `hendrycks/competition_math` |
| **OpenMathInstruct-2** | Synthetic math instruction pairs | 14M pairs | `nvidia/OpenMathInstruct-2` |
| **UltraFeedback** | RLHF preference alignment | 340K pairs | `openbmb/UltraFeedback` |
| **The Stack v2** | Code format validation | 1TB+ deduped | `bigcode/the-stack-v2` |

---

## 🔗 HuggingFace Ecosystem

All artifacts from this project are published to the HuggingFace Hub for open-source reproducibility:

| Artifact | Link | Purpose |
| :--- | :--- | :--- |
| 🧠 **Fine-Tuned Model** | [AetherControl-Qwen2.5-1.5B-GRPO-Math](https://huggingface.co/Aravind0495/AetherControl-Qwen2.5-1.5B-GRPO-Math) | GRPO-aligned math reasoning model & training card |
| 📦 **Sanitized Dataset** | [AetherControl-GSM8K-Sanitized](https://huggingface.co/datasets/Aravind0495/AetherControl-GSM8K-Sanitized) | TrainSight-validated GSM8K subset (1,000 rows) |
| 🚀 **Local Interactive Demo** | [space/app.py](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/space/app.py) | Local Gradio App for data validation, SLA inference, and GRPO telemetry |

---

## 🚀 Quickstart & One-Command Execution

```bash
# Clone the repository
git clone https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System.git
cd Building-and-Optimizing-Production-LLM-Serving-System

# Run test suites across all 3 Python packages (14/14 passing)
make test

# Profile real dataset quality and inspect vLLM production CLI flags
make profile

# Apply Kubernetes dev manifests locally
make k8s-dev

# Provision cost-optimized GCP GKE Spot GPU cluster (< $5-10 cost)
make cloud-up

# Teardown GCP cluster immediately (Zero idle cloud billing)
make cloud-down
```

---

## 📜 Architecture Decision Records (ADRs)

### ADR-001: Selection of GRPO over PPO for RLHF Alignment
* **Context:** Traditional PPO requires maintaining 4 active model copies in VRAM (Actor, Reference, Reward Model, and Value/Critic Model). The Critic model consumes ~50% of total VRAM.
* **Decision:** We adopted **Group Relative Policy Optimization (GRPO)** (DeepSeek R1 architecture). By sampling a group of $G$ completions per prompt and calculating relative advantage $A_i = (r_i - \mu) / (\sigma + \epsilon)$, GRPO eliminates the Critic network entirely, reducing VRAM utilization by 50% and doubling batch throughput.

### ADR-002: Selection of vLLM PagedAttention over TGI
* **Context:** Standard LLM serving allocates contiguous VRAM for maximum sequence lengths (`4096+` tokens), causing 60–80% internal memory fragmentation.
* **Decision:** We deployed **vLLM with PagedAttention**. PagedAttention allocates physical KV-cache memory blocks dynamically like OS virtual memory paging, eliminating memory fragmentation and increasing concurrent sequence capacity by up to 4x.

---

## 🔬 Silicon Profiling & Failure Experiments

* **NVIDIA Nsight Systems (`nsys`):** Documents exact CLI commands and microsecond timeline trace breakdowns separating compute-bound `flash_attn` kernels from memory-bound sampling.
* **Roofline Model:** Details mathematical equations for **Prefill Phase Compute-Bound** ($\mathcal{O}(N^2 \cdot d)$ FLOPs) vs. **Decode Phase Memory-Bandwidth-Bound** ($\mathcal{O}(N \cdot d)$ HBM reads).
* **5 Mandatory Failure Experiments ([docs/experiments.md](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/docs/experiments.md)):** Documents system degradation under VRAM pressure (`gpu_memory_utilization: 0.40`), 8K prompt context scaling, HBM saturation, Head-of-Line blocking, and preemption storms.

---

## 💼 Quantified Resume Highlights (XYZ Formula)

* **Data Engineering & Safety:** Engineered a pre-flight data validation InitContainer (`trainsight`), reducing wasted GPU compute hours by catching schema drift, empty completions, and sequence-length anomalies prior to Kubernetes pod scheduling.
* **Post-Training Alignment:** Architected an end-to-end GRPO RLHF pipeline with a no-critic relative advantage engine using HuggingFace `trl.GRPOTrainer`, optimizing VRAM utilization by 50% through Critic network elimination while maintaining mathematical verifier accuracy.
* **Serving & Observability:** Deployed and tuned a vLLM inference engine on GKE, utilizing Chunked Prefill and PagedAttention to achieve sub-50ms $P_{99}$ TPOT (Decode Latency) under concurrent load, instrumenting custom Prometheus/Grafana dashboards for TTFT, TPOT, and KV-cache block occupancy telemetry.
* **Systems Profiling & Fault Tolerance:** Conducted GPU hardware profiling (NVIDIA Nsight Systems `nsys`, Roofline model) and executed 5 structured failure experiments (Head-of-Line blocking, KV-cache starvation, CPU swapping), isolating execution bottlenecks and establishing production SLAs.

---

## 📜 License
MIT License
