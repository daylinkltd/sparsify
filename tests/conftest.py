"""Pytest configuration and shared fixtures."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from gguf import GGUFWriter

@pytest.fixture
def tiny_gguf_file(tmp_path: Path) -> Path:
    """Create a tiny, valid GGUF file with mock layers and weights for testing."""
    path = tmp_path / "tiny_model.gguf"
    writer = GGUFWriter(path, arch="llama")
    
    # Metadata
    writer.add_name("TinyTestModel")
    writer.add_block_count(2)
    writer.add_embedding_length(256)
    writer.add_head_count(8)
    writer.add_head_count_kv(4)
    writer.add_feed_forward_length(512)
    writer.add_context_length(1024)
    writer.add_file_type(15)  # MOSTLY_Q4_K_M
    
    # Add dummy tensors (using float32 to make it easy)
    # Embedding layer
    writer.add_tensor("token_embd.weight", np.zeros((100, 256), dtype=np.float32))
    
    # Layer 0
    writer.add_tensor("blk.0.attn_q.weight", np.zeros((256, 256), dtype=np.float32))
    writer.add_tensor("blk.0.attn_k.weight", np.zeros((128, 256), dtype=np.float32))
    writer.add_tensor("blk.0.attn_v.weight", np.zeros((128, 256), dtype=np.float32))
    writer.add_tensor("blk.0.attn_output.weight", np.zeros((256, 256), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_gate.weight", np.zeros((512, 256), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_up.weight", np.zeros((512, 256), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down.weight", np.zeros((256, 512), dtype=np.float32))
    writer.add_tensor("blk.0.attn_norm.weight", np.zeros((256,), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_norm.weight", np.zeros((256,), dtype=np.float32))
    
    # Layer 1
    writer.add_tensor("blk.1.attn_q.weight", np.zeros((256, 256), dtype=np.float32))
    writer.add_tensor("blk.1.attn_k.weight", np.zeros((128, 256), dtype=np.float32))
    writer.add_tensor("blk.1.attn_v.weight", np.zeros((128, 256), dtype=np.float32))
    writer.add_tensor("blk.1.attn_output.weight", np.zeros((256, 256), dtype=np.float32))
    writer.add_tensor("blk.1.ffn_gate.weight", np.zeros((512, 256), dtype=np.float32))
    writer.add_tensor("blk.1.ffn_up.weight", np.zeros((512, 256), dtype=np.float32))
    writer.add_tensor("blk.1.ffn_down.weight", np.zeros((256, 512), dtype=np.float32))
    writer.add_tensor("blk.1.attn_norm.weight", np.zeros((256,), dtype=np.float32))
    writer.add_tensor("blk.1.ffn_norm.weight", np.zeros((256,), dtype=np.float32))
    
    # Output layer
    writer.add_tensor("output.weight", np.zeros((100, 256), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.zeros((256,), dtype=np.float32))
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    return path
