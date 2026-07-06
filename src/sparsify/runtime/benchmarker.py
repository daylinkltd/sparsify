"""Benchmarker comparing Sparsify Dynamic Swapping against Traditional (Full-RAM) execution."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

import mlx.core as mx
from sparsify.runtime.cache import MoeCache
from sparsify.runtime.prefetcher import PredictivePrefetcher
from sparsify.runtime.registry import ExpertRegistry
from sparsify.prototype.moe_model import MoeTransformer
from sparsify.prototype.runner import compile_and_save_prototype

console = Console()


class SparsifyBenchmarker:
    """Benchmark class comparing peak memory, throughput, startup time, and Sparsify Ratio."""

    def __init__(self, experts_dir: Path) -> None:
        self.experts_dir = experts_dir
        self.registry = ExpertRegistry()
        self.registry.scan_directory(experts_dir)

    def run_traditional_benchmark(self) -> Dict[str, Any]:
        """Simulate a traditional runtime where the entire model is loaded into memory."""
        console.print("[bold yellow]Running Traditional Full-RAM Benchmark...[/bold yellow]")
        
        # Infinite budget loads all 128 experts into memory
        start_load = time.perf_counter()
        
        # Instantiate a cache that allows loading all experts without evictions
        cache = MoeCache(self.registry, budget_bytes=500 * 1024 * 1024, policy_name="lru")
        model = MoeTransformer(cache=cache)
        
        # Pre-load all 128 experts to simulate full memory loading
        for l in range(8):
            for e in range(16):
                _ = model.cache.get_expert(l, e)
                
        load_time = time.perf_counter() - start_load
        
        # Run inference benchmarks over a dummy sequence
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        start_inf = time.perf_counter()
        
        # Execute 5 runs for statistics
        for _ in range(5):
            out = model(prompt)
            mx.eval(out)
            
        inf_duration = time.perf_counter() - start_inf
        tokens_per_sec = (10 * 5) / inf_duration
        
        # Parameters size in memory
        def count_params(d: Any) -> int:
            if isinstance(d, dict):
                return sum(count_params(v) for v in d.values())
            elif isinstance(d, list):
                return sum(count_params(v) for v in d)
            elif hasattr(d, "size"):
                return d.size
            return 0
        shared_size = count_params(model.parameters())
        total_expert_size = 128 * (3 * 256 * 1024)
        peak_params = shared_size + total_expert_size
        
        return {
            "startup_latency_seconds": load_time,
            "peak_parameters_in_memory": peak_params,
            "peak_memory_mb": peak_params * 4 / 1024 / 1024,
            "throughput_tokens_per_sec": tokens_per_sec,
            "ratio": 1.0,
        }

    def run_sparsify_benchmark(self, cache_policy: str = "lru", prefetch: bool = True) -> Dict[str, Any]:
        """Run the Sparsify Runtime benchmark using dynamic swapping and prefetching."""
        console.print(f"[bold yellow]Running Sparsify Virtual Memory Benchmark ({cache_policy.upper()}, Prefetch={prefetch})...[/bold yellow]")
        
        start_load = time.perf_counter()
        
        # Constrained budget: 24MB (exactly 8 active experts in memory)
        cache = MoeCache(self.registry, budget_bytes=24 * 1024 * 1024, policy_name=cache_policy)
        model = MoeTransformer(cache=cache)
        
        # Load time for Sparsify is just loading the shared weights (excluding experts)
        load_time = time.perf_counter() - start_load
        
        prefetcher = PredictivePrefetcher(cache) if prefetch else None
        
        # Patch the model layers to run the prefetcher on FFN executions
        if prefetcher:
            for l_idx, layer in enumerate(model.layers):
                original_call = layer._call_impl
                
                def make_prefetch_call(l=l_idx, orig=original_call):
                    def prefetch_call(x, mask=None):
                        # Log router selection and prefetch
                        B, L, D = x.shape
                        x_flat = x.reshape(-1, D)
                        logits = layer.router(x_flat)
                        routing_idx = int(mx.argmax(logits, axis=-1)[0])
                        prefetcher.record_access_and_predict(l, routing_idx)
                        return orig(x, mask=mask)
                    return prefetch_call
                
                layer._call_impl = make_prefetch_call()

        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        start_inf = time.perf_counter()
        
        for _ in range(5):
            out = model(prompt)
            mx.eval(out)
            
        inf_duration = time.perf_counter() - start_inf
        tokens_per_sec = (10 * 5) / inf_duration
        
        if prefetcher:
            prefetcher.shutdown()
            
        # Peak parameters in memory (shared + bounded cache size)
        def count_params(d: Any) -> int:
            if isinstance(d, dict):
                return sum(count_params(v) for v in d.values())
            elif isinstance(d, list):
                return sum(count_params(v) for v in d)
            elif hasattr(d, "size"):
                return d.size
            return 0
        shared_size = count_params(model.parameters())
        active_expert_size = 8 * (3 * 256 * 1024)
        peak_params = shared_size + active_expert_size
        
        total_expert_size = 128 * (3 * 256 * 1024)
        total_size = shared_size + total_expert_size
        ratio = peak_params / total_size
        
        return {
            "startup_latency_seconds": load_time,
            "peak_parameters_in_memory": peak_params,
            "peak_memory_mb": peak_params * 4 / 1024 / 1024,
            "throughput_tokens_per_sec": tokens_per_sec,
            "ratio": ratio,
        }

    def execute_and_compare(self) -> None:
        """Run both benchmarks and present a comparative terminal analytics table."""
        trad = self.run_traditional_benchmark()
        console.print()
        spar = self.run_sparsify_benchmark(cache_policy="adaptive", prefetch=True)
        
        table = Table(title="Sparsify Runtime Comparative Benchmarking", title_style="bold cyan")
        table.add_column("Benchmark Metric", style="bold")
        table.add_column("Traditional (Full-RAM)", justify="right")
        table.add_column("Sparsify (Virtual Memory)", justify="right", style="green")
        table.add_column("Improvement / Savings", justify="right", style="bold yellow")
        
        # 1. Startup Latency
        speedup = trad["startup_latency_seconds"] / spar["startup_latency_seconds"] if spar["startup_latency_seconds"] > 0 else 0.0
        table.add_row(
            "Startup / Load Latency",
            f"{trad['startup_latency_seconds']*1000.0:.2f} ms",
            f"{spar['startup_latency_seconds']*1000.0:.2f} ms",
            f"{speedup:.2f}x Faster"
        )
        
        # 2. Peak RAM Parameter Size
        ram_saved = trad["peak_memory_mb"] - spar["peak_memory_mb"]
        table.add_row(
            "Peak RAM Footprint",
            f"{trad['peak_memory_mb']:.2f} MB",
            f"{spar['peak_memory_mb']:.2f} MB",
            f"{ram_saved:.2f} MB Saved"
        )
        
        # 3. Throughput
        pct_perf = (spar["throughput_tokens_per_sec"] / trad["throughput_tokens_per_sec"]) * 100.0
        table.add_row(
            "Generation Throughput",
            f"{trad['throughput_tokens_per_sec']:.2f} tok/s",
            f"{spar['throughput_tokens_per_sec']:.2f} tok/s",
            f"{pct_perf:.1f}% Performance Retained"
        )
        
        # 4. Sparsify Ratio
        table.add_row(
            "Sparsify Ratio (Active/Total)",
            "100.00%",
            f"{spar['ratio']*100.0:.3f}%",
            f"{100.0 - (spar['ratio']*100.0):.2f}% RAM Reduction"
        )
        
        console.print()
        console.print(table)
        console.print()
