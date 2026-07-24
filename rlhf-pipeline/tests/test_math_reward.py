import pytest
from rlhf_pipeline.rewards.math_reward import GRPOReasoningReward


def test_evaluate_format_reward():
    evaluator = GRPOReasoningReward()
    valid_text = "<think>Steps here</think> <answer>42</answer>"
    invalid_text = "The answer is 42"

    assert evaluator.evaluate_format_reward(valid_text) == 0.5
    assert evaluator.evaluate_format_reward(invalid_text) == 0.0


def test_evaluate_accuracy_reward():
    evaluator = GRPOReasoningReward()
    text = "<think>Calculation</think> <answer>15</answer>"

    assert evaluator.evaluate_accuracy_reward(text, "15") == 1.0
    assert evaluator.evaluate_accuracy_reward(text, "20") == 0.0


def test_compute_total_reward():
    evaluator = GRPOReasoningReward()
    text = "<think>2+2=4</think> <answer>4</answer>"
    res = evaluator.compute_total_reward(text, "4")

    assert res["total_reward"] == 1.5
    assert res["format_reward"] == 0.5
    assert res["accuracy_reward"] == 1.0
