# Learning Dossier 03: DPO Fundamentals, Jaccard Similarity & SFT Benchmarks

> **Folder Path**: `docs/dossier/03_dpo_jaccard_and_benchmarks.md`  
> **Session Topic**: DPO Mechanics, Jaccard Similarity Intuition, SFT Hardware Benchmarks, and Data Corruption Examples  
> **Date**: July 21, 2026  

---

## 1. What is DPO (Direct Preference Optimization)?

DPO is Stage 2 of post-training alignment.

- **In Stage 1 (SFT)**: We give the model 1 prompt and 1 good response. The model learns **how to generate language**.
- **In Stage 2 (DPO)**: We give the model 1 prompt and **2 alternative responses**:
  1. **Chosen ($y_w$)**: The preferred, higher-quality response.
  2. **Rejected ($y_l$)**: The dispreferred, lower-quality response.

### Why DPO Replaced RLHF PPO
Older RLHF required training a separate 7B+ parameter **Reward Model** network plus running unstable PPO Actor-Critic loops. DPO mathematically proves that the policy network itself can act as its own reward model, making preference alignment **3x faster, stable, and lightweight**.

---

## 2. What is Jaccard Similarity, and How Does It Help Us?

### Mathematical Definition
Jaccard Similarity measures the similarity between two sets of words by calculating the size of their intersection divided by the size of their union:

$$\text{Jaccard Similarity}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Number of Shared Unique Words}}{\text{Total Unique Words Across Both Texts}}$$

### Code Implementation (`dpo_inspector.py`)
```python
def _jaccard_similarity(self, text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0
```

### Core Intuition: Why High Jaccard Similarity Means "Corrupted Data"
The goal of DPO is to teach an AI model **"What makes an answer GOOD vs BAD"**.

If a dataset provides two nearly identical answers:
* **Chosen**: `"The capital of France is Paris."`
* **Rejected**: `"The capital city of France is Paris."`

These two sentences share **86% of the exact same words** (High Jaccard Similarity). They mean the exact same thing!
If you tell an AI model *"Answer 1 is Good, but Answer 2 (which is 99% identical) is Bad"*, the model is confused because there is no actual quality difference to learn from.

#### The Mathematical Problem on GPU:
In DPO, the GPU calculates the difference in log-probabilities:

$$\text{Margin} = \log \pi(\text{Chosen}) - \log \pi(\text{Rejected})$$

* When Jaccard Similarity is **High (90%-100%)**, $\text{Chosen} \approx \text{Rejected}$, so $\text{Margin} \approx 0$.
* **The loss gradient $\nabla_\theta \mathcal{L} \approx 0$**.
* **The GPU updates zero weights!** It runs expensive math on your A100 GPU for hours, but **the model learns nothing**.
* That is why `trainsight` flags high Jaccard similarity as **Corrupted Data**—it wastes GPU compute on zero-gradient pairs!

---

## 3. What Makes a DPO Pair "Ready for Training"?

A pair is **Ready for Training** when there is a **strong quality difference** between Chosen and Rejected, which naturally results in **Low/Moderate Jaccard Similarity (10% to 50%)**.

### Example of a Great DPO Pair (Ready for Training):
* **Prompt**: `"How do I bake a chocolate cake?"`
* **Chosen (Detailed & Helpful)**: `"1. Preheat oven to 350°F. 2. Mix 2 cups flour, 1 cup cocoa powder, and 2 eggs. 3. Pour into pan and bake for 30 minutes. 4. Let cool before frosting."`
* **Rejected (Lazy & Unhelpful)**: `"Just mix some flour and cocoa and cook it."`

### Why this pair is Ready for Training:
1. **Jaccard Similarity is Low (~15%)**: The two answers use different structures and amounts of detail.
2. **Huge Quality Gap**: Chosen is clear and step-by-step; Rejected is lazy and vague.
3. **Strong Gradient Signal**: The GPU calculates a large positive margin $\text{Margin} \gg 0$. The model receives a **strong signal** to learn step-by-step formatting!

### Summary Matrix

| Dataset Metric | High Jaccard Similarity (90%-100%) | Low Jaccard Similarity (10%-40%) |
| :--- | :--- | :--- |
| **Word Overlap** | Nearly identical words | Distinct words & structures |
| **Quality Gap** | No difference | Huge quality difference |
| **GPU Gradient Signal** | $\approx 0$ (Zero gradient, no learning) | Strong gradient (Active learning) |
| **`trainsight` Status** | ❌ **CORRUPTED (Drop pair)** | ✅ **READY FOR TRAINING** |

---

## 4. What Benchmarks Are We Comparing the SFT Inspector Against?

The SFT Inspector compares dataset characteristics against **GPU Hardware Constraints & Batching Efficiency Baselines**:

| Benchmark Parameter | Threshold / Baseline | Engineering Rationale |
| :--- | :--- | :--- |
| **Max Sequence Length** | `2048` / `4096` tokens | Benchmarked against target GPU VRAM limits and model context window. Exceeding this triggers **OOM (Out-Of-Memory)** crashes. |
| **Sequence Variance** | `std_len > 0.75 * avg_len` | Benchmarked against **PyTorch / vLLM Batching Efficiency**. High variance forces batch padding, creating $>30\%$ GPU idle bubbles. |
| **Empty Completions** | `0` tolerance | Benchmarked against cross-entropy loss contracts. Missing completions force loss to penalize predictions, degrading model tone. |
| **Duplicate Prompts** | `< 2%` threshold | Benchmarked against instruction-tuning papers (Alpaca/UltraFeedback) to prevent gradient over-fitting. |

---

## 5. Step-by-Step Word Calculation Example (Case 4)

### Sample Texts
* **Chosen**: `"Place egg in boiling water for 6-7 mins for soft-boiled or 10 mins for hard-boiled. Cool in ice water."`
* **Rejected**: `"Put egg in water."`

### Word Sets
* **Chosen Word Set ($A$)**: `{'place', 'egg', 'in', 'boiling', 'water', 'for', '6-7', 'mins', 'soft-boiled', 'or', '10', 'hard-boiled', 'cool', 'ice'}` ($|A| = 14$)
* **Rejected Word Set ($B$)**: `{'put', 'egg', 'in', 'water'}` ($|B| = 4$)

### Intersection & Union
* **Intersection ($A \cap B$)**: `{'egg', 'in', 'water'}` ($|A \cap B| = 3$)
* **Union ($A \cup B$)**: `{'place', 'egg', 'in', 'boiling', 'water', 'for', '6-7', 'mins', 'soft-boiled', 'or', '10', 'hard-boiled', 'cool', 'ice', 'put'}` ($|A \cup B| = 15$)

$$\text{Jaccard Similarity} = \frac{3}{15} = 0.20 = \mathbf{20.0\%}$$

Since $20.0\% \ll 90.0\%$, the pair is **VALID & READY FOR TRAINING**.
