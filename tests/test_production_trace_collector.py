"""Unit and integration tests for Sprint 8 production trace collection sweeps."""
from __future__ import annotations

import mlx.core as mx
import pytest
from pathlib import Path

from sparsify.runtime.production_trace_collector import ProductionMoeModel, ProductionTraceAnalyzer


def test_production_topologies_and_trace_solvers() -> None:
    """Validate 32-layer Qwen/Mixtral routing sweeps and metric calculators."""
    
    # 1. Instantiate Mixtral 8x7B layout
    model = ProductionMoeModel(num_experts=8)
    assert len(model.layers) == 32
    
    emb = mx.random.normal((1, 4096))
    traces = model.generate_traces(emb, num_tokens=20)
    
    # 32 layers, 20 tokens trace generated
    assert len(traces) == 32
    assert len(traces[0]) == 20
    assert len(traces[0][0]) == 2  # Top-2 experts
    
    # 2. Analyze traces
    analyzer = ProductionTraceAnalyzer(traces, num_experts=8)
    entropy = analyzer.calculate_entropy()
    reuse_dist = analyzer.calculate_reuse_distance()
    w50, w90 = analyzer.calculate_working_set_sizes()
    sparsity = analyzer.calculate_matrix_sparsity()
    top1, top3 = analyzer.calculate_predictability()
    
    assert entropy >= 0.0
    assert reuse_dist >= 0.0
    assert w50 >= 1
    assert w90 >= 1
    assert 0.0 <= sparsity <= 1.0
    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top3 <= 1.0

    # 3. Cache sweeps
    cache_hits = analyzer.sweep_cache_budgets(expert_size_mb=175.0)
    assert "2GB" in cache_hits
    assert "4GB" in cache_hits
    assert "8GB" in cache_hits
    assert "16GB" in cache_hits
    for k, v in cache_hits.items():
        assert 0.0 <= v <= 1.0
