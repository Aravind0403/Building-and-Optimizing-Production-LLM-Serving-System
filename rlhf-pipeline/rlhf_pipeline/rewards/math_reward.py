import re
from typing import Tuple, Dict, Any


class GRPOReasoningReward:
    """Rule-based reward evaluator for DeepSeek-R1 style reasoning models (GSM8K/MATH)."""

    def __init__(self, accuracy_weight: float = 1.0, format_weight: float = 0.5):
        self.accuracy_weight = accuracy_weight
        self.format_weight = format_weight

    def extract_reasoning_and_answer(self, completion: str) -> Tuple[str, str, bool]:
        """Extracts text inside <think>...</think> and <answer>...</answer> tags."""
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)

        think_text = think_match.group(1).strip() if think_match else ""
        answer_text = answer_match.group(1).strip() if answer_match else ""

        # Format is valid if both think and answer tags exist in proper order
        format_valid = bool(think_match and answer_match and think_match.start() < answer_match.start())
        return think_text, answer_text, format_valid

    def evaluate_format_reward(self, completion: str) -> float:
        """Returns format reward (+0.5 if valid <think>/<answer> tags exist, 0.0 otherwise)."""
        _, _, format_valid = self.extract_reasoning_and_answer(completion)
        return self.format_weight if format_valid else 0.0

    def evaluate_accuracy_reward(self, completion: str, ground_truth: str) -> float:
        """Returns accuracy reward (+1.0 if extracted answer matches ground truth)."""
        _, extracted_answer, _ = self.extract_reasoning_and_answer(completion)
        
        # Fallback: if no <answer> tag, search for trailing numbers
        if not extracted_answer:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", completion)
            extracted_answer = numbers[-1] if numbers else ""

        cleaned_extracted = extracted_answer.strip().lower()
        cleaned_target = ground_truth.strip().lower()

        if cleaned_extracted == cleaned_target:
            return self.accuracy_weight

        # Check numeric equivalence
        try:
            if float(cleaned_extracted) == float(cleaned_target):
                return self.accuracy_weight
        except ValueError:
            pass

        return 0.0

    def compute_total_reward(self, completion: str, ground_truth: str) -> Dict[str, float]:
        """Computes combined format + accuracy reward scores."""
        r_format = self.evaluate_format_reward(completion)
        r_accuracy = self.evaluate_accuracy_reward(completion, ground_truth)
        r_total = r_format + r_accuracy

        return {
            "total_reward": r_total,
            "format_reward": r_format,
            "accuracy_reward": r_accuracy,
        }
