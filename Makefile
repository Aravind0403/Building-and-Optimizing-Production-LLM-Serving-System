.PHONY: all test profile k8s-dev clean

all: test profile

test:
	@echo "🧪 Running unit tests across all AetherControl modules..."
	(cd trainsight && pytest)
	(cd vllm-engine && pytest)
	(cd rlhf-pipeline && pytest)
	@echo "✅ All unit tests passed!"

profile:
	@echo "🔍 Running TrainSight dataset quality profiling..."
	(cd trainsight && trainsight profile --dataset sample_data/sample_gsm8k.jsonl --type sft)
	@echo "⚡ Displaying vLLM engine production CLI arguments..."
	(cd vllm-engine && vllm-bench show-config)

k8s-dev:
	@echo "☸️ Applying local Kubernetes dev manifests..."
	kubectl apply -f k8s-infra/manifests/secrets.yaml
	kubectl apply -f k8s-infra/manifests/local-dev-deployment.yaml
	kubectl apply -f k8s-infra/manifests/prometheus-grafana.yaml
	@echo "✨ Kubernetes manifests applied successfully!"

clean:
	@echo "🧹 Cleaning up local cluster and background processes..."
	-KIND_EXPERIMENTAL_PROVIDER=podman kind delete cluster --name vllm-local
	-podman machine stop
	@echo "✨ Teardown complete!"
