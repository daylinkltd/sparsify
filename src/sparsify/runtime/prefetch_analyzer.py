"""Predictability Analyzer building expert transition matrices and benchmarking prefetching."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

import mlx.core as mx
from sparsify.runtime.cache import MoeCache
from sparsify.runtime.prefetcher import PredictivePrefetcher
from sparsify.runtime.registry import ExpertRegistry
from sparsify.prototype.moe_model import MoeTransformer


class PrefetchPredictabilityAnalyzer:
    """Markov transition matrix builder and forecasting accuracy checker."""

    def __init__(self, experts_dir: Path) -> None:
        self.experts_dir = Path(experts_dir)
        self.registry = ExpertRegistry()
        self.registry.scan_directory(experts_dir)

        # Traces of activated experts per layer
        self.activation_traces: Dict[int, List[int]] = {i: [] for i in range(8)}

    def collect_activation_traces(self) -> None:
        """Run text generation to gather trace sequences of activated experts across all layers."""
        cache = MoeCache(self.registry, budget_bytes=100 * 1024 * 1024, policy_name="lru")
        model = MoeTransformer(cache=cache)

        # Hook layers to log routing selections
        for l_idx, layer in enumerate(model.layers):
            original_call = layer._call_impl
            def make_logging_call(l=l_idx, orig=original_call):
                def log_call(x, mask=None):
                    B, L, D = x.shape
                    x_flat = x.reshape(-1, D)
                    logits = layer.router(x_flat)
                    routing_idx = int(mx.argmax(logits, axis=-1)[0])
                    # Log the trace activation
                    self.activation_traces[l].append(routing_idx)
                    return orig(x, mask=mask)
                return log_call
            layer._call_impl = make_logging_call()

        # Run 50 tokens generation
        prompt = mx.array([[1, 2, 3, 4, 5]])
        for _ in range(50):
            logits = model(prompt)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1]).item())
            prompt = mx.concat([prompt, mx.array([[next_token]])], axis=1)

    def calculate_forecasting_accuracy(self) -> Tuple[float, float, Dict[int, np.ndarray]]:
        """Compute transition matrices and calculate cumulative top-1 and top-3 prediction accuracy."""
        transition_matrices: Dict[int, np.ndarray] = {}
        top1_hits = 0
        top3_hits = 0
        total_evals = 0

        for l_idx, trace in self.activation_traces.items():
            if len(trace) < 2:
                continue

            # Build transition frequency matrix (16 experts x 16 experts)
            freq_matrix = np.zeros((16, 16), dtype=np.int32)
            
            # Online prediction validation loop
            for t in range(1, len(trace)):
                prev = trace[t-1]
                actual = trace[t]

                # Predict based on history up to t-1
                if freq_matrix[prev].sum() > 0:
                    probs = freq_matrix[prev].astype(np.float32) / freq_matrix[prev].sum()
                    # Sorted indices of highest probabilities
                    sorted_preds = np.argsort(probs)[::-1]
                    
                    # Top-1 match
                    if sorted_preds[0] == actual:
                        top1_hits += 1
                    # Top-3 match
                    if actual in sorted_preds[:3]:
                        top3_hits += 1
                        
                    total_evals += 1

                # Update transition frequency matrix
                freq_matrix[prev, actual] += 1

            transition_matrices[l_idx] = freq_matrix

        top1_acc = (top1_hits / total_evals) if total_evals > 0 else 0.0
        top3_acc = (top3_hits / total_evals) if total_evals > 0 else 0.0

        return top1_acc, top3_acc, transition_matrices


class PrefetchBenchmarker:
    """Benchmark comparing cache hits and throughput across prefetch configurations."""

    def __init__(self, registry: ExpertRegistry) -> None:
        self.registry = registry

    def run_benchmark_run(self, prefetch_mode: str) -> Dict[str, Any]:
        """Run standard generation loop under different prefetching policies.

        prefetch_mode: 'none', 'top1', 'top3'
        """
        # Bounded cache capacity: 8 experts (24MB)
        cache = MoeCache(self.registry, budget_bytes=24 * 1024 * 1024, policy_name="adaptive")
        model = MoeTransformer(cache=cache)

        # Setup prefetchers
        prefetcher = PredictivePrefetcher(cache) if prefetch_mode != "none" else None

        # Intercept forward layer routing calls
        for l_idx, layer in enumerate(model.layers):
            original_call = layer._call_impl
            
            def make_prefetch_call(l=l_idx, orig=original_call):
                def p_call(x, mask=None):
                    B, L, D = x.shape
                    x_flat = x.reshape(-1, D)
                    logits = layer.router(x_flat)
                    routing_idx = int(mx.argmax(logits, axis=-1)[0])
                    
                    if prefetcher:
                        # Record access and check hit
                        prefetcher.record_access_and_predict(l, routing_idx)
                        
                        # Top-3 Prefetch simulation
                        if prefetch_mode == "top3":
                            # Preload top 3 highest probability transitions in parallel
                            key = (l, routing_idx)
                            candidates = prefetcher.transitions.get(key)
                            if candidates:
                                sorted_candidates = sorted(candidates.keys(), key=lambda k: candidates[k], reverse=True)
                                for c_idx in sorted_candidates[:3]:
                                    prefetcher.trigger_background_prefetch(l, c_idx)
                                    
                    return orig(x, mask=mask)
                return p_call
            layer._call_impl = make_prefetch_call()

        # Run 20 generation steps
        prompt = mx.array([[10, 20, 30, 40, 50]])
        start_time = time.perf_counter()
        
        for _ in range(20):
            logits = model(prompt)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1]).item())
            prompt = mx.concat([prompt, mx.array([[next_token]])], axis=1)

        duration = time.perf_counter() - start_time
        if prefetcher:
            prefetcher.shutdown()

        total_reqs = cache.hits + cache.misses
        hit_ratio = cache.hits / total_reqs if total_reqs > 0 else 0.0

        return {
            "throughput_tokens_sec": 20 / duration,
            "cache_hit_ratio": hit_ratio,
            "ssd_read_mb": cache.bytes_read_total / 1024 / 1024,
            "cache_evictions": cache.evictions,
        }
