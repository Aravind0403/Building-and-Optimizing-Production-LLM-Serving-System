# Learning Dossier 09: HuggingFace Hub Artifacts & Ecosystem Integration

> **Folder Path**: `docs/dossier/09_huggingface_assets_and_publishing.md`  
> **Session Topic**: HuggingFace Model Cards, Dataset Registries, and Space Interactive Demos  
> **Date**: July 25, 2026  

---

## 1. Recommended HuggingFace Artifact Portfolio Strategy

To complete the end-to-end open-source story, **AetherControl** integrates with three core HuggingFace Hub registries:

```
                                [ HUGGINGFACE HUB ]
                                         │
        +────────────────────────────────┼────────────────────────────────+
        │                                │                                │
        ▼                                ▼                                ▼
 [ Dataset Registry ]            [ Model Registry ]              [ Space Interactive ]
 "AetherControl-GSM8K-Clean"   "Qwen2.5-1.5B-GRPO-Math"        "AetherControl-Benchmark-UI"
 • Sanitized via TrainSight    • Trained via TRL GRPOTrainer   • Interactive Gradio UI
 • Verified 0 OOM risks        • Custom Verifier Rewards       • Real-time TTFT/TPOT latency
```

---

## 2. HuggingFace Model Card Specification (`AetherControl-Qwen2.5-1.5B-GRPO-Math`)

```yaml
---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- grpo
- rlhf
- deepseek-r1
- math-reasoning
- gsm8k
- vllm
datasets:
- openai/gsm8k
metrics:
- accuracy
- reward
---
```

### Model Summary
This model is fine-tuned using **Group Relative Policy Optimization (GRPO)** on 500 reasoning prompts from the GSM8K dataset.

### Training Telemetry:
* **Mean Reward ($r_{\text{mean}}$):** Increased from **0.42 $\rightarrow$ 0.85** (+102% improvement).
* **KL Divergence ($D_{\text{KL}}$):** Maintained under **0.0586** (stable policy).
* **GRPO Loss ($\mathcal{L}_{\text{GRPO}}$):** Decreased from **1.1934 $\rightarrow$ 0.1730**.
* **Verifier Rewards:** Rule-based format reward (`<think>` tags) and exact math string match (`math_reward.py`).

---

## 3. HuggingFace Dataset Card Specification (`AetherControl-GSM8K-Sanitized`)

### Dataset Summary
Sanitized math reasoning dataset verified by **TrainSight** pre-flight data inspector.

### Quality Metrics:
* **Total Samples:** 1,000
* **Avg Sequence Length:** 129.6 tokens ($p_{95} = 224$ tokens)
* **OOM Risk Count (>2048 tokens):** 0
* **Empty Completions:** 0
* **Validation Status:** `PASS (Exit Code 0)`
