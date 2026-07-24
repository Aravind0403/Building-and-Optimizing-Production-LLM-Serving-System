import numpy as np
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from rlhf_pipeline.rewards.math_reward import GRPOReasoningReward


class GRPOTrainingConfig(BaseModel):
    """Configuration parameters for GRPO fine-tuning pipeline."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    group_size: int = Field(default=4, ge=2, description="Number of completions sampled per prompt (Group Size G)")
    kl_coeff: float = Field(default=0.04, ge=0.0, description="KL divergence penalty coefficient beta")
    clip_eps: float = Field(default=0.2, gt=0.0, description="PPO/GRPO ratio clipping parameter epsilon")
    learning_rate: float = Field(default=1e-6, description="Policy model learning rate")
    max_completion_length: int = Field(default=512, description="Max generated tokens per completion")


class GRPOGroupResult(BaseModel):
    """Calculated advantages and rewards for a single sampled prompt group."""
    prompt: str
    ground_truth: str
    completions: List[str]
    rewards: List[float]
    advantages: List[float]
    group_mean_reward: float
    group_std_reward: float


class GRPOTrainer:
    """Group Relative Policy Optimization (GRPO) Trainer Engine."""

    def __init__(self, config: GRPOTrainingConfig):
        self.config = config
        self.reward_evaluator = GRPOReasoningReward()

    def compute_group_advantages(self, rewards: List[float], eps: float = 1e-8) -> Tuple[List[float], float, float]:
        """Computes group-relative normalized advantages A_i = (r_i - mean) / (std + eps)."""
        arr = np.array(rewards, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if std < eps:
            advantages = [0.0 for _ in rewards]
        else:
            advantages = [float((r - mean) / (std + eps)) for r in rewards]

        return advantages, mean, std

    def process_prompt_group(self, prompt: str, ground_truth: str, completions: List[str]) -> GRPOGroupResult:
        """Evaluates rewards and computes relative advantages across a sampled completion group."""
        if len(completions) != self.config.group_size:
            raise ValueError(f"Expected {self.config.group_size} completions per group, got {len(completions)}")

        rewards = []
        for comp in completions:
            eval_res = self.reward_evaluator.compute_total_reward(comp, ground_truth)
            rewards.append(eval_res["total_reward"])

        advantages, mean_r, std_r = self.compute_group_advantages(rewards)

        return GRPOGroupResult(
            prompt=prompt,
            ground_truth=ground_truth,
            completions=completions,
            rewards=rewards,
            advantages=advantages,
            group_mean_reward=mean_r,
            group_std_reward=std_r,
        )
