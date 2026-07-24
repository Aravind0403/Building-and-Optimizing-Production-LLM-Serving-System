import pytest
from rlhf_pipeline.grpo_trainer import GRPOTrainer, GRPOTrainingConfig


def test_grpo_advantage_computation():
    trainer = GRPOTrainer(GRPOTrainingConfig(group_size=4))
    rewards = [1.5, 0.5, 0.0, 1.5]

    advantages, mean_r, std_r = trainer.compute_group_advantages(rewards)
    assert len(advantages) == 4
    assert abs(mean_r - 0.875) < 1e-3
    assert advantages[0] > 0
    assert advantages[2] < 0


def test_process_prompt_group():
    trainer = GRPOTrainer(GRPOTrainingConfig(group_size=4))
    prompt = "Solve 2x = 6"
    ground_truth = "3"
    completions = [
        "<think>2x=6 -> x=3</think> <answer>3</answer>",
        "<think>x=3</think> <answer>3</answer>",
        "<think>wrong</think> <answer>5</answer>",
        "Answer 3"
    ]

    res = trainer.process_prompt_group(prompt, ground_truth, completions)
    assert res.prompt == prompt
    assert len(res.rewards) == 4
    assert len(res.advantages) == 4
    assert res.rewards[0] == 1.5
