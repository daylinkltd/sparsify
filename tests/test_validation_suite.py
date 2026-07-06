"""Unit tests for the Sparsify Runtime Validation Suite."""
from __future__ import annotations

from pathlib import Path
import pytest

from sparsify.runtime.validation_suite import ValidationRunner
from sparsify.prototype.runner import compile_and_save_prototype


def test_validation_suite_execution_and_plotting(tmp_path) -> None:
    """Validate output parity checking, budget sweeps, and matplotlib plotting."""
    
    # 1. Compile mock experts in tmp_path
    _, _ = compile_and_save_prototype(tmp_path)

    # 2. Initialize runner and verify parity and perplexity calculations
    runner = ValidationRunner(tmp_path, tmp_path)
    parity, perp_delta = runner.check_output_parity_and_perplexity()
    
    assert parity is True
    assert perp_delta == 0.0

    # 3. Run budget sweep stress workloads
    results = runner.run_sweep()
    assert len(results["budgets_gb"]) == 5
    assert len(results["throughput_tok_sec"]) == 5
    assert len(results["hit_ratios"]) == 5

    # 4. Generate Matplotlib visualization plots
    runner.generate_plots(results)
    
    # Verify that the 4 charts are rendered as PNG files on disk
    assert (tmp_path / "ram_vs_throughput.png").exists()
    assert (tmp_path / "hit_ratio_vs_ram.png").exists()
    assert (tmp_path / "bandwidth_vs_tokens.png").exists()
    assert (tmp_path / "latency_distribution.png").exists()
