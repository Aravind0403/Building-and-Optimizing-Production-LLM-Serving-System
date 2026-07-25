import pytest
from rlhf_pipeline.grpo_trl_trainer import reward_function_format, reward_function_accuracy


def test_reward_function_format():
    prompts = ["Solve 2x = 6"]
    completions = ["<think>2x=6 -> x=3</think> <answer>3</answer>"]

    scores = reward_function_format(prompts, completions)
    assert len(scores) == 1
    assert scores[0] == 0.5


def test_reward_function_accuracy():
    prompts = ["Solve 2x = 6"]
    completions = ["<think>2x=6 -> x=3</think> <answer>3</answer>"]
    answers = ["3"]

    scores = reward_function_accuracy(prompts, completions, answers)
    assert len(scores) == 1
    assert scores[0] == 1.0
