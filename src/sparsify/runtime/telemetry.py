"""Telemetry module to record, calculate, and format runtime performance metrics."""
from __future__ import annotations

import time
from typing import Any, Dict, List


class TelemetryRecorder:
    """Telemetry system tracking memory usage, SSD bandwidth, cache hits, prefetching, and throughput."""

    def __init__(self, total_model_bytes: int) -> None:
        self.total_model_bytes = total_model_bytes
        self.start_time = 0.0
        self.first_token_latency = 0.0
        self.total_tokens = 0
        
        # Cache & Disk IO stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.total_ssd_read_bytes = 0
        
        # Load times history
        self.load_latencies: List[float] = []
        
        # Prefetcher stats
        self.prefetch_attempts = 0
        self.prefetch_hits = 0
        
        # Working footprint
        self.active_memory_footprint_bytes = 0
        self.active_experts_count = 0

    def start_generation(self) -> None:
        """Mark the beginning of token generation."""
        self.start_time = time.perf_counter()
        self.first_token_latency = 0.0
        self.total_tokens = 0

    def record_token(self) -> None:
        """Increment generated token count and track first token latency if applicable."""
        self.total_tokens += 1
        if self.total_tokens == 1 and self.start_time > 0:
            self.first_token_latency = time.perf_counter() - self.start_time

    def get_metrics(self) -> Dict[str, Any]:
        """Compute and return aggregated metrics."""
        elapsed = time.perf_counter() - self.start_time if self.start_time > 0 else 0.0
        tokens_per_sec = self.total_tokens / elapsed if elapsed > 0 else 0.0
        
        # Calculate SSD bandwidth (MB/s)
        ssd_bandwidth_mbs = (self.total_ssd_read_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
        
        # Cache hit ratio
        total_cache_requests = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        
        # Average load latency
        avg_load_lat = sum(self.load_latencies) / len(self.load_latencies) if self.load_latencies else 0.0
        
        # Prefetch hit ratio
        prefetch_ratio = self.prefetch_hits / self.prefetch_attempts if self.prefetch_attempts > 0 else 0.0
        
        # Sparsify ratio
        sparsify_ratio = self.active_memory_footprint_bytes / self.total_model_bytes if self.total_model_bytes > 0 else 0.0

        return {
            "total_model_size_mb": self.total_model_bytes / (1024 * 1024),
            "active_memory_footprint_mb": self.active_memory_footprint_bytes / (1024 * 1024),
            "sparsify_ratio": sparsify_ratio,
            "ssd_bandwidth_mbs": ssd_bandwidth_mbs,
            "cache_hit_ratio": hit_ratio,
            "expert_load_latency_ms": avg_load_lat * 1000.0,
            "active_experts_count": self.active_experts_count,
            "cache_evictions": self.cache_evictions,
            "tokens_per_sec": tokens_per_sec,
            "first_token_latency_ms": self.first_token_latency * 1000.0,
            "prefetch_hit_ratio": prefetch_ratio,
            "prefetch_attempts": self.prefetch_attempts,
            "prefetch_hits": self.prefetch_hits,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": elapsed,
        }
