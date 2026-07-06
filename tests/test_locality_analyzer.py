"""Unit and integration tests for Sprint 7 routing locality validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from sparsify.runtime.locality_analyzer import LocalityAnalyzer
from sparsify.prototype.runner import compile_and_save_prototype


def test_routing_locality_validation_and_ram_solving(tmp_path) -> None:
    """Validate routing Shannon Entropy, W90, and minimum RAM matrices calculations."""
    
    # 1. Compile mock experts under tmp_path
    _, _ = compile_and_save_prototype(tmp_path)

    # 2. Run locality analysis
    analyzer = LocalityAnalyzer(tmp_path)
    traces = analyzer.collect_routing_traces()
    
    # Verify traces are collected
    for l in range(8):
        assert len(traces[l]) > 0

    # Compute metrics
    metrics = analyzer.compute_locality_metrics(traces)
    assert metrics["entropy"] >= 0.0
    assert metrics["avg_reuse_distance"] >= 0.0
    assert 1 <= metrics["working_set_size_w90"] <= 16

    # 3. Solve RAM matrix
    matrix = analyzer.solve_minimum_ram_matrix()
    assert "30B" in matrix
    assert "70B" in matrix
    assert "120B" in matrix
    assert "10_tok_sec" in matrix["30B"]
    assert "20_tok_sec" in matrix["30B"]
    assert "30_tok_sec" in matrix["30B"]
