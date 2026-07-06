"""Unit and integration tests for the MLX Experimentation Framework."""
from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from sparsify.backends.mlx_backend import MLXBackend
from sparsify.experiments.hooks import patch_model_for_experimentation
from sparsify.experiments.evaluator import evaluate_perplexity, evaluate_accuracy_probe
from sparsify.experiments.runner import execute_all_experiments


@pytest.fixture(scope="module")
def loaded_mlx_model():
    """Load the tiny cached Llama 3.2 1B model for module-level tests."""
    backend = MLXBackend()
    if not backend.is_available():
        pytest.skip("MLX backend is not available on this machine (requires macOS Apple Silicon).")
        
    try:
        # Load local cached version or pull if missing
        backend.load_model("mlx-community/Llama-3.2-1B-Instruct-4bit")
        yield backend._model, backend._tokenizer
    finally:
        backend.unload_model()


def test_model_patching_and_hooks(loaded_mlx_model) -> None:
    """Test that we can patch model layers and control them via hooks."""
    model, tokenizer = loaded_mlx_model
    
    # 1. Patch the model
    bypass_hooks, head_hooks, sparsity_hooks = patch_model_for_experimentation(model)
    
    assert len(bypass_hooks) > 0
    assert len(head_hooks) == len(bypass_hooks)
    assert len(sparsity_hooks) == len(bypass_hooks)
    
    # Verify hook types
    assert bypass_hooks[0].layer_index == 0
    assert head_hooks[0].layer_index == 0
    assert sparsity_hooks[0].layer_index == 0

    # 2. Test forward pass with bypass enabled
    # We create a dummy input
    inputs = mx.array([[101, 102, 103]])
    
    # Baseline pass
    out_base = model(inputs)
    mx.eval(out_base)
    
    # Enable bypass on layer 0 and 2
    bypass_hooks[0].bypass_active = True
    bypass_hooks[2].bypass_active = True
    
    out_bypassed = model(inputs)
    mx.eval(out_bypassed)
    
    # Outputs should differ but remain valid (no NaNs)
    assert out_bypassed.shape == out_base.shape
    assert not mx.isnan(out_bypassed).any()
    
    # Reset hooks
    bypass_hooks[0].bypass_active = False
    bypass_hooks[2].bypass_active = False


def test_head_masking_hook(loaded_mlx_model) -> None:
    """Test that we can mask attention heads and execute the forward pass."""
    model, tokenizer = loaded_mlx_model
    
    # Patch (idempotent, hooks already installed if run sequentially)
    bypass_hooks, head_hooks, sparsity_hooks = patch_model_for_experimentation(model)
    
    # Mask head 0 and 4 in layer 1
    head_hooks[1].masked_heads.add(0)
    head_hooks[1].masked_heads.add(4)
    
    inputs = mx.array([[101, 102, 103]])
    out = model(inputs)
    mx.eval(out)
    
    assert out.shape == (1, 3, model.args.vocab_size)
    assert not mx.isnan(out).any()
    
    # Reset masks
    head_hooks[1].masked_heads.clear()


def test_activation_sparsity_hook(loaded_mlx_model) -> None:
    """Test that we can monitor and force sparsity in MLP gating."""
    model, tokenizer = loaded_mlx_model
    bypass_hooks, head_hooks, sparsity_hooks = patch_model_for_experimentation(model)
    
    # Enable forced sparsity on layer 0
    sparsity_hooks[0].force_sparsity = True
    sparsity_hooks[0].sparsity_threshold = 1e-3
    
    inputs = mx.array([[101, 102, 103]])
    out = model(inputs)
    mx.eval(out)
    
    # Verify that the sparsity rate is recorded
    assert sparsity_hooks[0].last_sparsity_rate > 0.0
    assert sparsity_hooks[0].last_sparsity_rate <= 1.0
    
    # Reset
    sparsity_hooks[0].force_sparsity = False


def test_evaluator_functions(loaded_mlx_model) -> None:
    """Test perplexity and accuracy functions on the patched model."""
    model, tokenizer = loaded_mlx_model
    
    # Run perplexity on a tiny subset
    ppl = evaluate_perplexity(model, tokenizer, max_tokens=100)
    acc = evaluate_accuracy_probe(model, tokenizer, max_tokens=100)
    
    assert isinstance(ppl, float)
    assert ppl > 1.0
    assert 0.0 <= acc <= 1.0
