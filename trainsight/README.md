# trainsight (The Validator)

`trainsight` is a pre-training and post-training data validation CLI that catches data corruptions, sequence length anomalies, label inconsistencies, and preference margin collapses before launching expensive GPU training runs.

---

## 💻 Usage

```bash
# Profile an SFT dataset
trainsight profile --dataset data/sft_data.jsonl --type sft

# Profile a DPO / Preference dataset
trainsight profile --dataset data/dpo_data.jsonl --type dpo
```
