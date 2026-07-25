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

# AetherControl-Qwen2.5-1.5B-GRPO-Math

This model is fine-tuned using **Group Relative Policy Optimization (GRPO)** on 500 reasoning prompts from the GSM8K dataset as part of the **AetherControl** platform.

---

## 📊 Training Telemetry Progression (20 Steps)

```text
                📊 GRPO Fine-Tuning Progression & Telemetry Log                 
┏━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Step    ┃ Mean Reward ┃ Format      ┃ Accuracy    ┃ KL          ┃ GRPO Loss  ┃
┃         ┃ (r_mean)    ┃ Reward      ┃ Reward      ┃ Div (D_KL)  ┃ (L_grpo)   ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Step 01 │ 0.42        │ 0.23        │ 0.22        │ 0.0376      │ 1.1934     │
│ Step 05 │ 0.50        │ 0.34        │ 0.32        │ 0.0581      │ 0.9610     │
│ Step 10 │ 0.64        │ 0.54        │ 0.48        │ 0.0595      │ 0.7322     │
│ Step 15 │ 0.72        │ 0.75        │ 0.70        │ 0.0536      │ 0.4367     │
│ Step 20 │ 0.85        │ 0.92        │ 0.87        │ 0.0586      │ 0.1730     │
└─────────┴─────────────┴─────────────┴─────────────┴─────────────┴────────────┘
```

* **Mean Reward ($r_{\text{mean}}$):** Increased from **0.42 $\rightarrow$ 0.85** (+102% improvement).
* **KL Divergence ($D_{\text{KL}}$):** Maintained under **0.0586** (stable policy).
* **GRPO Loss ($\mathcal{L}_{\text{GRPO}}$):** Decreased from **1.1934 $\rightarrow$ 0.1730**.
* **Verifier Rewards:** Rule-based format reward (`<think>` tags) and exact math string match (`math_reward.py`).

---

## 🏛️ System Repository
Github Repository: [https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System](https://github.com/Aravind0403/Building-and-Optimizing-Production-LLM-Serving-System)
