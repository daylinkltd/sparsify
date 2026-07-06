"""Unit tests for the static model profiler."""
from __future__ import annotations

from pathlib import Path
from sparsify.profiler.model_profiler import profile_model, format_profile_table
from sparsify.profiler.metrics import ModelProfile

def test_profile_model(tiny_gguf_file: Path) -> None:
    """Test building a ModelProfile statically from a GGUF file."""
    profile = profile_model(tiny_gguf_file)
    
    assert isinstance(profile, ModelProfile)
    assert profile.model_path == str(tiny_gguf_file)
    assert profile.model_name == "TinyTestModel"
    assert profile.architecture == "llama"
    assert profile.quantization == "MOSTLY_Q4_K_M"
    assert profile.layer_count == 2
    assert profile.head_count == 8
    assert profile.kv_head_count == 4
    assert profile.embedding_dim == 256
    assert profile.ffn_dim == 512
    assert profile.context_length == 1024
    
    # Check layer count matches
    assert len(profile.layer_profiles) == 2
    for lp in profile.layer_profiles:
        # Check all expected components exist
        assert "attention_q" in lp.components
        assert "attention_k" in lp.components
        assert "attention_v" in lp.components
        assert "attention_output" in lp.components
        assert "ffn_gate" in lp.components
        assert "ffn_up" in lp.components
        assert "ffn_down" in lp.components
        assert "norm" in lp.components
    
    # Check component summary aggregates correctly
    summary = profile.component_summary
    assert "embedding" in summary
    assert "output" in summary
    assert "attention_q" in summary
    
    # Check KV cache estimates are populated
    assert len(profile.kv_cache_estimates) > 0
    # Context lengths of estimates should not exceed model context length (1024)
    # The default estimate context lengths are [512, 1024, 2048, 4096...]
    # So we expect estimates for 512 and 1024.
    ctx_lengths = [est.context_length for est in profile.kv_cache_estimates]
    assert ctx_lengths == [512, 1024]
    
    # Verify KV size calculation
    # Head dim = 256 // 8 = 32
    # KV per token = 2 * n_layers(2) * n_kv_heads(4) * head_dim(32) * 2 bytes (F16) = 1024 bytes
    # For context 512, size = 512 * 1024 = 524288 bytes (512 KB)
    est_512 = next(est for est in profile.kv_cache_estimates if est.context_length == 512)
    assert est_512.total_size_bytes == 524288
    
    # Check serializability
    d = profile.to_dict()
    assert isinstance(d, dict)
    assert d["architecture"] == "llama"
    assert len(d["layer_profiles"]) == 2
    assert d["component_summary"]["attention_q"]["name"] == "attention_q"

def test_format_profile_table(tiny_gguf_file: Path) -> None:
    """Test that we can generate rich-formatted tables from the profile."""
    profile = profile_model(tiny_gguf_file)
    output = format_profile_table(profile, verbose=True)
    
    assert "Model Summary" in output
    assert "Memory Breakdown by Component" in output
    assert "KV Cache Estimates" in output
    assert "Per-Layer Breakdown" in output
    assert "TinyTestModel" in output
    assert "llama" in output
