"""Unit and integration tests for the Sparsify Virtual Memory Runtime."""
from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from sparsify.runtime.registry import ExpertRegistry
from sparsify.runtime.cache import LRUPolicy, LFUPolicy, ARCPolicy, MoeCache
from sparsify.runtime.prefetcher import PredictivePrefetcher
from sparsify.runtime.telemetry import TelemetryRecorder
from sparsify.runtime.benchmarker import SparsifyBenchmarker
from sparsify.prototype.runner import compile_and_save_prototype


def test_registry_and_cache_eviction_policies(tmp_path) -> None:
    """Test registry indexing and correctness of LRU, LFU, and ARC cache policies."""
    # Create mock directory and serialize prototype experts
    _, _ = compile_and_save_prototype(tmp_path)

    registry = ExpertRegistry(registry_file=tmp_path / "cache.json")
    count = registry.scan_directory(tmp_path)
    assert count == 128

    # 1. Test LRU Policy Eviction
    lru = LRUPolicy(capacity=3)
    assert lru.access((0, 1)) is None
    assert lru.access((0, 2)) is None
    assert lru.access((0, 3)) is None
    # Accessing 4th item must evict (0, 1) as it is the least recently used
    assert lru.access((0, 4)) == (0, 1)

    # 2. Test LFU Policy Eviction
    lfu = LFUPolicy(capacity=3)
    lfu.access((0, 1))
    lfu.access((0, 2))
    lfu.access((0, 3))
    # Record hits on 2 and 3
    lfu.record_hit((0, 2))
    lfu.record_hit((0, 3))
    # Accessing 4th item must evict (0, 1) because its hit count is lowest (1 vs 2)
    assert lfu.access((0, 4)) == (0, 1)

    # 3. Test ARC Policy Eviction
    arc = ARCPolicy(capacity=4)
    arc.access((0, 1))
    arc.access((0, 2))
    arc.access((0, 3))
    arc.access((0, 4))
    # Verify ghost cache and Adaptive replacement triggers evictions
    evicted = arc.access((0, 5))
    assert evicted is not None


def test_predictive_prefetcher(tmp_path) -> None:
    """Test Markov prefetch forecasting and background preloading."""
    _, _ = compile_and_save_prototype(tmp_path)
    
    registry = ExpertRegistry(registry_file=tmp_path / "cache.json")
    registry.scan_directory(tmp_path)
    cache = MoeCache(registry, budget_bytes=16 * 1024 * 1024, policy_name="lru")
    
    prefetcher = PredictivePrefetcher(cache)
    
    # Simulate a pattern: expert 3 is always followed by expert 7 in Layer 2
    for _ in range(5):
        prefetcher.record_access_and_predict(layer_id=2, expert_id=3)
        prefetcher.record_access_and_predict(layer_id=2, expert_id=7)
        
    # Check if prediction for current expert=3 is expert=7
    predicted = prefetcher.predict_next_expert(layer_id=2, current_expert_id=3)
    assert predicted == 7
    
    # Triggering access on 3 should prefetch 7 automatically
    prefetcher.record_access_and_predict(layer_id=2, expert_id=3)
    
    # Wait for background thread tasks to complete
    prefetcher.executor.shutdown(wait=True)
    
    # Verify expert 7 was successfully loaded in memory by prefetch thread
    assert (2, 7) in cache._loaded_experts


def test_benchmarker(tmp_path) -> None:
    """Test full benchmarker comparison execution."""
    _, _ = compile_and_save_prototype(tmp_path)
    
    bench = SparsifyBenchmarker(tmp_path)
    # Verify that traditional and sparsify runs execute without error
    trad_results = bench.run_traditional_benchmark()
    spar_results = bench.run_sparsify_benchmark(cache_policy="lru", prefetch=False)
    
    assert trad_results["peak_memory_mb"] > spar_results["peak_memory_mb"]
    assert spar_results["ratio"] < 1.0
