import time
from pathlib import Path
import concurrent.futures
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vllm_engine.config_loader import load_vllm_config, VLLMConfig
from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner
from vllm_engine.benchmarks.load_generator import generate_synthetic_prompts, send_streaming_request

app = typer.Typer(help="vLLM Serving Engine CLI & Benchmark Suite")
console = Console()


@app.command()
def show_config(
    config_file: Path = typer.Option(Path(__file__).parent / "config.yaml", "--config", "-c", help="Path to config.yaml")
):
    """Displays current vLLM engine configuration and corresponding CLI launch flags."""
    try:
        config = load_vllm_config(config_file)
        console.print(Panel(f"[bold cyan]Model:[/bold cyan] {config.model}\n"
                            f"[bold cyan]VRAM Utilization:[/bold cyan] {config.gpu_memory_utilization * 100}%\n"
                            f"[bold cyan]Max Context Len:[/bold cyan] {config.max_model_len}\n"
                            f"[bold cyan]Chunked Prefill:[/bold cyan] {config.enable_chunked_prefill}\n"
                            f"[bold cyan]Prefix Caching:[/bold cyan] {config.enable_prefix_caching}",
                            title="vLLM Production Configuration"))

        cli_args = config.to_cli_args()
        console.print(f"\n[bold green]Generated Entrypoint Command:[/bold green]")
        console.print(f"[yellow]python3 -m vllm.entrypoints.openai.api_server {' '.join(cli_args)}[/yellow]\n")
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")


@app.command()
def benchmark(
    host: str = typer.Option("http://localhost:8000", "--host", "-h", help="vLLM server endpoint URL"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model name"),
    num_requests: int = typer.Option(20, "--num-requests", "-n", help="Total benchmark requests"),
    concurrency: int = typer.Option(4, "--concurrency", "-k", help="Concurrent worker threads"),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max tokens to generate per request"),
):
    """Runs standard industry SLA benchmark against a vLLM server."""
    console.print(f"\n[bold cyan]🚀 Starting vLLM SLA Benchmark...[/bold cyan]")
    console.print(f"Target: [bold yellow]{host}[/bold yellow] | Model: [bold yellow]{model}[/bold yellow]")
    console.print(f"Requests: {num_requests} | Concurrency: {concurrency}\n")

    prompts = generate_synthetic_prompts(num_requests)
    runner = ServingBenchmarkRunner(host, model)

    results = []
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_streaming_request, runner.api_url, model, prompt, max_tokens)
            for prompt in prompts
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_duration = time.perf_counter() - start_time
    summary = runner.summarize_results(results, total_duration)

    table = Table(title="vLLM Serving Performance & SLA Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Duration", f"{summary.total_duration_s:.2f} s")
    table.add_row("Success / Total Requests", f"{summary.successful_requests} / {summary.total_requests}")
    table.add_row("Request Throughput", f"{summary.request_throughput_rps:.2f} req/s")
    table.add_row("Output Token Throughput", f"{summary.output_token_throughput_tps:.2f} tokens/s")
    table.add_row("TTFT (Prefill Latency) P50", f"{summary.ttft_p50 * 1000:.1f} ms")
    table.add_row("TTFT (Prefill Latency) P99", f"{summary.ttft_p99 * 1000:.1f} ms")
    table.add_row("TPOT (Decode Latency) P50", f"{summary.tpot_p50 * 1000:.1f} ms/token")
    table.add_row("TPOT (Decode Latency) P99", f"{summary.tpot_p99 * 1000:.1f} ms/token")
    table.add_row("E2E Latency P99", f"{summary.e2e_p99:.2f} s")

    console.print(table)


if __name__ == "__main__":
    app()
