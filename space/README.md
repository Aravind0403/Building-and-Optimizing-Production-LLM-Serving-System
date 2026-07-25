---
title: AetherControl Benchmark UI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# AetherControl Benchmark UI

Interactive Gradio Space demo showcasing the three core pillars of **AetherControl**:
1. **Data Validation:** Pre-flight dataset quality inspection via TrainSight.
2. **Inference Playground:** Low-latency SLA streaming inference ($P_{50}$ TTFT 851.9ms, TPOT 24.7ms/token).
3. **GRPO Telemetry:** Post-training mean reward curves ($0.42 \rightarrow 0.85$) and stable KL divergence ($D_{\text{KL}} < 0.15$).
