# TrainSight: Data Quality Inspector & K8s InitContainer

`trainsight` is a pre-training and post-training data validation CLI and Kubernetes InitContainer that catches dataset corruption, sequence length anomalies, CUDA OOM risks, and duplicate prompts **before** launching expensive GPU training runs.

---

## 📚 Supported Production Datasets

TrainSight standardizes and validates datasets across SFT, DPO, and GRPO post-training pipelines:

| Dataset | Use Case | Size | Source |
| :--- | :--- | :--- | :--- |
| **GSM8K** | Math reasoning for GRPO training | 8.5K train / 1K test | `openai/gsm8k` |
| **MATH** | Competition math problems | 7.5K train / 5K test | `hendrycks/competition_math` |
| **OpenMathInstruct-2** | Synthetic math instruction data | 14M pairs | `nvidia/OpenMathInstruct-2` |
| **UltraFeedback** | RLHF preference data | 340K pairs | `openbmb/UltraFeedback` |
| **The Stack v2** | Code data for format validation | 1TB+ deduped | `bigcode/the-stack-v2` |

---

## 💻 Usage

```bash
# Profile an SFT dataset
trainsight profile --dataset sample_data/sample_sft.jsonl --type sft

# Profile a DPO preference dataset
trainsight profile --dataset sample_data/sample_dpo.jsonl --type dpo

# Profile a GSM8K math reasoning dataset
trainsight profile --dataset sample_data/sample_gsm8k.jsonl --type sft
```
