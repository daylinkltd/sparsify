"""Unit tests for the GGUF metadata reader."""
from __future__ import annotations

from pathlib import Path
from sparsify.utils.gguf_reader import read_gguf_metadata, list_tensors

def test_read_gguf_metadata(tiny_gguf_file: Path) -> None:
    """Test that we can parse metadata from the synthetic GGUF model."""
    meta = read_gguf_metadata(tiny_gguf_file)
    
    assert meta.file_path == tiny_gguf_file
    assert meta.architecture == "llama"
    assert meta.model_name == "TinyTestModel"
    assert meta.quantization_type == "MOSTLY_Q4_K_M"
    assert meta.block_count == 2
    assert meta.attention_head_count == 8
    assert meta.attention_head_count_kv == 4
    assert meta.embedding_length == 256
    assert meta.feed_forward_length == 512
    assert meta.context_length == 1024
    assert isinstance(meta.raw_metadata, dict)
    assert meta.raw_metadata["general.architecture"] == "llama"
    
    # Check to_dict() serialization works
    d = meta.to_dict()
    assert d["file_path"] == str(tiny_gguf_file)
    assert d["architecture"] == "llama"
    assert d["block_count"] == 2

def test_list_tensors(tiny_gguf_file: Path) -> None:
    """Test that we can parse and classify tensor information from GGUF."""
    tensors = list_tensors(tiny_gguf_file)
    
    # Check that we got all tensors
    # 1 token_embd + 2 layers * 9 tensors + 2 output = 21 tensors
    assert len(tensors) == 21
    
    # Verify classifications and index extraction
    embedding_tensor = next(t for t in tensors if t.name == "token_embd.weight")
    assert embedding_tensor.component == "embedding"
    assert embedding_tensor.layer_index is None
    assert embedding_tensor.dtype == "F32"
    assert embedding_tensor.shape == (256, 100)  # GGUF shapes are stored in reverse
    assert embedding_tensor.n_elements == 100 * 256
    assert embedding_tensor.size_bytes == 100 * 256 * 4
    
    # Layer 0 tensors
    l0_q = next(t for t in tensors if t.name == "blk.0.attn_q.weight")
    assert l0_q.component == "attention_q"
    assert l0_q.layer_index == 0
    assert l0_q.dtype == "F32"
    assert l0_q.size_bytes == 256 * 256 * 4
    
    l0_norm = next(t for t in tensors if t.name == "blk.0.attn_norm.weight")
    assert l0_norm.component == "norm"
    assert l0_norm.layer_index == 0
    
    # Output tensor
    out = next(t for t in tensors if t.name == "output.weight")
    assert out.component == "output"
    assert out.layer_index is None
