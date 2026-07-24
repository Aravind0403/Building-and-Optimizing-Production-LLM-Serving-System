import time
import asyncio
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import requests


@dataclass
class RequestBenchmarkResult:
    """Individual HTTP inference request latency breakdown."""
    prompt_len: int
    output_len: int
    success: bool
    error_message: str = ""
    start_time: float = 0.0
    ttft: float = 0.0          # Time To First Token (seconds)
    tpot: float = 0.0          # Time Per Output Token (seconds)
    e2e_latency: float = 0.0   # End to End Latency (seconds)


@dataclass
class BenchmarkSummary:
    """Aggregated SLA performance report across all test requests."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_s: float
    request_throughput_rps: float
    output_token_throughput_tps: float
    
    # TTFT Percentiles (Prefill Latency)
    ttft_p50: float
    ttft_p90: float
    ttft_p95: float
    ttft_p99: float
    
    # TPOT Percentiles (Decode Latency)
    tpot_p50: float
    tpot_p90: float
    tpot_p95: float
    tpot_p99: float

    # E2E Percentiles
    e2e_p50: float
    e2e_p90: float
    e2e_p95: float
    e2e_p99: float


class ServingBenchmarkRunner:
    """Standard OpenAI-compatible API load testing & SLA benchmark runner."""

    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url.rstrip("/") + "/v1/chat/completions"
        self.model_name = model_name

    def calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Utility to calculate P50, P90, P95, P99 percentiles."""
        if not values:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def summarize_results(self, results: List[RequestBenchmarkResult], total_duration_s: float) -> BenchmarkSummary:
        """Aggregates raw request latency metrics into a structured BenchmarkSummary."""
        successful = [r for r in results if r.success]
        failed_count = len(results) - len(successful)

        if not successful:
            return BenchmarkSummary(
                total_requests=len(results),
                successful_requests=0,
                failed_requests=failed_count,
                total_duration_s=total_duration_s,
                request_throughput_rps=0.0,
                output_token_throughput_tps=0.0,
                ttft_p50=0.0, ttft_p90=0.0, ttft_p95=0.0, ttft_p99=0.0,
                tpot_p50=0.0, tpot_p90=0.0, tpot_p95=0.0, tpot_p99=0.0,
                e2e_p50=0.0, e2e_p90=0.0, e2e_p95=0.0, e2e_p99=0.0,
            )

        ttfts = [r.ttft for r in successful if r.ttft > 0]
        tpots = [r.tpot for r in successful if r.tpot > 0]
        e2es = [r.e2e_latency for r in successful]

        ttft_stats = self.calculate_percentiles(ttfts)
        tpot_stats = self.calculate_percentiles(tpots)
        e2e_stats = self.calculate_percentiles(e2es)

        total_tokens = sum(r.output_len for r in successful)
        rps = len(successful) / total_duration_s if total_duration_s > 0 else 0.0
        tps = total_tokens / total_duration_s if total_duration_s > 0 else 0.0

        return BenchmarkSummary(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=failed_count,
            total_duration_s=total_duration_s,
            request_throughput_rps=rps,
            output_token_throughput_tps=tps,
            ttft_p50=ttft_stats["p50"],
            ttft_p90=ttft_stats["p90"],
            ttft_p95=ttft_stats["p95"],
            ttft_p99=ttft_stats["p99"],
            tpot_p50=tpot_stats["p50"],
            tpot_p90=tpot_stats["p90"],
            tpot_p95=tpot_stats["p95"],
            tpot_p99=tpot_stats["p99"],
            e2e_p50=e2e_stats["p50"],
            e2e_p90=e2e_stats["p90"],
            e2e_p95=e2e_stats["p95"],
            e2e_p99=e2e_stats["p99"],
        )
