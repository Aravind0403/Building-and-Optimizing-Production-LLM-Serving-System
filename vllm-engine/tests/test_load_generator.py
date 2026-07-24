import pytest
from vllm_engine.benchmarks.load_generator import generate_synthetic_prompts, send_streaming_request, load_dataset_prompts

def test_generate_synthetic_prompts():
    prompts = generate_synthetic_prompts(num_prompts=5, prompt_tokens=50)
    assert len(prompts) == 5
    for p in prompts:
        assert isinstance(p, str)
        assert len(p.split()) >= 50

def test_load_dataset_prompts(tmp_path):
    sample_file = tmp_path / "sample.jsonl"
    sample_file.write_text('{"prompt": "What is PyTorch?"}\n{"instruction": "Explain CUDA."}\n')
    prompts = load_dataset_prompts(sample_file, max_samples=2)
    assert len(prompts) == 2
    assert prompts[0] == "What is PyTorch?"
    assert prompts[1] == "Explain CUDA."


def test_send_streaming_request_unreachable_host():
    # Tests graceful failure when target vLLM server is unreachable
    result = send_streaming_request(
        api_url="http://127.0.0.1:59999/v1/chat/completions",
        model_name="test-model",
        prompt="Hello world",
        timeout=1.0
    )
    assert result.success is False
    assert "Connection" in result.error_message or "HTTP" in result.error_message or "refused" in result.error_message.lower()
