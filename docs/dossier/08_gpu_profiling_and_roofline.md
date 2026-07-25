# Learning Dossier 08: GPU Hardware Profiling, Roofline Analysis & Quantization

> **Folder Path**: `docs/dossier/08_gpu_profiling_and_roofline.md`  
> **Session Topic**: NVIDIA Nsight Systems (nsys), Arithmetic Intensity Roofline Model, and Quantization Metrics  
> **Date**: July 25, 2026  

---

## 1. NVIDIA Nsight Systems (`nsys`) Execution Tracing

High-level metrics (like Grafana throughput graphs) confirm *what* happened, but low-level hardware profilers reveal *why* latency spikes occur.

### Production Nsight Profiling Command
To capture microsecond-level CUDA kernel execution, NVTX ranges, and host CPU thread scheduling:

```bash
nsys profile \
  -o docs/profiling/vllm_execution_trace \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 2048
```

```
                                 [ NSYS TIMELINE TRACE ]
┌─────────────────────────────────────────────────────────────────────────────┐
│ CPU Host Thread: [ Tokenizer ] ──► [ vLLM Scheduler Step ] ──► [ CUDA Launch ]│
├─────────────────────────────────────────────────────────────────────────────┤
│ GPU Stream 0:    [ FlashAttention Kernel (Prefill) ] ──► [ Decode Kernels ]  │
│                  (Compute-Bound: Tensor Cores Active)    (Memory-Bound HBM) │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Execution Characteristics Identified in Trace:
1. **Prefill Phase Kernels:** Characterized by dense `flash_attn` matrix multiplication kernels that saturate GPU Tensor Cores.
2. **Decode Phase Kernels:** Characterized by repetitive, low-arithmetic-intensity memory fetches per token (sampling & KV-cache access).
3. **Scheduler Gaps:** White space between CUDA kernel launches indicates CPU-bound scheduler overhead or tokenization delays.

---

## 2. The Operational Roofline Model for LLMs

The Roofline Model plots **Arithmetic Intensity** (FLOPs per byte of VRAM moved across HBM) to determine the absolute performance ceiling of a GPU hardware architecture.

```
       Attainable Performance (TFLOPS)
            ▲
            │                 /─────────────────── Peak Compute Ceiling (Tensor Cores)
            │                /
            │               /  ◄─── PREFILL PHASE (Compute-Bound)
            │              /        O(N² · d) FLOPs
            │             /
            │            / ◄────── DECODE PHASE (Memory-Bandwidth Bound)
            │           /          O(N · d) Memory Reads per Token
            │          /
            └─────────┴──────────────────────────────► Arithmetic Intensity (FLOPs / Byte)
                      HBM Bandwidth Limit (TB/s)
```

### Architectural Breakdown

#### A. Prefill Phase (Compute-Bound)
* **Complexity:** $\mathcal{O}(N^2 \cdot d)$ FLOPs for prompt context $N$.
* **Bottleneck:** GPU Tensor Core FLOPS capacity (e.g., 660 TFLOPS FP16 on H100).
* **Behavior:** Increasing batch size increases throughput linearly because matrix-matrix multiplications ($GEMM$) amortize memory fetch costs across all tokens in parallel.

#### B. Decode Phase (Memory-Bandwidth-Bound)
* **Complexity:** $\mathcal{O}(N \cdot d)$ memory reads per token.
* **Bottleneck:** High Bandwidth Memory (HBM) Bandwidth (e.g., 3.35 TB/s on H100).
* **Behavior:** Generation is inherently slow because every single generated token requires reading all 70B model parameters from VRAM. **KV-Cache quantization** is critical to reduce HBM traffic.

---

## 3. Quantization Benchmark Matrix (FP16 vs. FP8 vs. AWQ)

| Precision Format | Weight Bit-Width | VRAM Footprint (70B Model) | Throughput Scaling | TTFT Impact | Accuracy Drop | Best Deployment Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | 16-bit Float | 140 GB | 1.0x (Baseline) | Baseline | 0.0% (Lossless) | Multi-node A100 / H100 clusters |
| **FP8 (W8A8)** | 8-bit Float | 70 GB (-50%) | **1.85x Speedup** | -35% Latency | $< 0.1\%$ Perplexity | NVIDIA H100 / L40S (Native FP8 Tensor Cores) |
| **AWQ (W4A16)** | 4-bit Weights | 35 GB (-75%) | **2.20x Speedup** | -45% Latency | $< 0.5\%$ Perplexity | Memory-constrained GPUs (L4, A10G, RTX 4090) |

> **Production Takeaway:**  
> *"FP8 (W8A8) provides near-lossless throughput gains on modern H100 architectures, whereas AWQ (W4A16) is strictly indicated for memory-constrained edge/single-GPU deployments (e.g., L4 or A10G) where HBM bandwidth is the primary bottleneck."*
