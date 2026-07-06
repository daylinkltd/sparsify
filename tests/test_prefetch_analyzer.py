"""Unit and integration tests for Sprint 5 prefetch predictability analysis."""
from __future__ import annotations

from pathlib import Path
import pytest

from sparsify.runtime.prefetch_analyzer import PrefetchPredictabilityAnalyzer, PrefetchBenchmarker
from sparsify.prototype.runner import compile_and_save_prototype


def test_prefetch_predictability_and_benchmarking(tmp_path) -> None:
    """Validate predictability metric calculations and prefetching sweeps."""
    
    # 1. Compile mock experts under tmp_path
    _, _ = compile_and_save_prototype(tmp_path)

    # 2. Run predictability analysis
    analyzer = PrefetchPredictabilityAnalyzer(tmp_path)
    analyzer.collect_activation_traces()
    
    # Verify traces were gathered for all 8 layers
    for l in range(8):
        assert len(analyzer.activation_traces[l]) > 0

    top1, top3, matrices = analyzer.calculate_forecasting_accuracy()
    
    # Values should be valid percentages
    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top3 <= 1.0
    assert len(matrices) == 8
    assert matrices[0].shape == (16, 16)

    # 3. Run comparative prefetch benchmarks
    bench = PrefetchBenchmarker(analyzer.registry)
    
    none_res = bench.run_benchmark_run(prefetch_mode="none")
    top1_res = bench.run_benchmark_run(prefetch_mode="top1")
    top3_res = bench.run_benchmark_run(prefetch_mode="top3")

    # Assert basic metric fields are populated
    for res in [none_res, top1_res, top3_res]:
        assert res["throughput_tokens_sec"] > 0.0
        assert 0.0 <= res["cache_hit_ratio"] <= 1.0
        assert res["ssd_read_mb"] >= 0.0
        assert res["cache_evictions"] >= 0
