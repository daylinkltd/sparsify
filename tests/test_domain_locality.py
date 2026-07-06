"""Unit tests for the SP-009 Domain Locality Analysis module."""
from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from sparsify.backends.mlx_backend import MLXBackend
from sparsify.experiments.prompts_bank import PROMPT_BANK
import sparsify.experiments.domain_locality as dl


@pytest.fixture(scope="module")
def loaded_mlx_model():
    """Load the tiny cached Llama 3.2 1B model for module-level tests."""
    backend = MLXBackend()
    if not backend.is_available():
        pytest.skip("MLX backend is not available on this machine (requires macOS Apple Silicon).")
        
    try:
        backend.load_model("mlx-community/Llama-3.2-1B-Instruct-4bit")
        yield backend._model, backend._tokenizer
    finally:
        backend.unload_model()


def test_domain_locality_computation(loaded_mlx_model, tmp_path) -> None:
    """Verify that domain locality analysis processes inputs and outputs correctly."""
    # Slice prompt bank to 2 prompts per domain for fast test execution
    sliced_bank = {k: v[:2] for k, v in PROMPT_BANK.items()}
    
    # Monkey-patch PROMPT_BANK
    original_bank = dl.PROMPT_BANK
    dl.PROMPT_BANK = sliced_bank
    
    try:
        report = dl.run_domain_locality_experiment(
            model_path="mlx-community/Llama-3.2-1B-Instruct-4bit",
            output_dir=tmp_path,
        )
        
        assert report is not None
        assert "timestamp" in report
        assert "metrics" in report
        assert "cosine_overlap_matrix" in report
        assert "jaccard_similarity_matrix" in report
        
        metrics = report["metrics"]
        assert "unrelated_domain_average_overlap" in metrics
        assert "average_silhouette_score" in metrics
        assert "passed_success_criteria" in metrics
        assert "passed_failure_criteria" in metrics
        
        # Verify plots and reports were written to target path
        assert (tmp_path / "layer_activation_heatmap.png").exists()
        assert (tmp_path / "attention_head_heatmap.png").exists()
        assert (tmp_path / "domain_overlap_matrix.png").exists()
        assert (tmp_path / "jaccard_similarity_matrix.png").exists()
        assert (tmp_path / "pca_clusters.png").exists()
        assert (tmp_path / "domain_locality_report.json").exists()
        
        # Check matrix shapes (6 domains)
        overlap = np.array(report["cosine_overlap_matrix"])
        assert overlap.shape == (6, 6)
        
        jaccard = np.array(report["jaccard_similarity_matrix"])
        assert jaccard.shape == (6, 6)
        
    finally:
        # Restore original prompt bank
        dl.PROMPT_BANK = original_bank
