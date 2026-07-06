"""Unit tests for the SQLite persistence layer."""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from sparsify.storage.database import SparsifyDB, compute_file_hash

@pytest.fixture
def temp_db() -> Path:
    """Fixture to manage a temporary SQLite database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()

def test_database_crud(temp_db: Path) -> None:
    """Test saving, retrieving, and listing profile runs and metadata."""
    # Setup test mock data
    profile = {
        "model_path": "/path/to/mock_model.gguf",
        "model_hash": "abcdef1234567890",
        "architecture": "llama",
        "layer_count": 12,
        "embedding_dim": 1024,
        "some_metric": 42.0,
    }
    
    metadata = {
        "model_path": "/path/to/mock_model.gguf",
        "architecture": "llama",
        "layer_count": 12,
        "head_count": 16,
        "embedding_dim": 1024,
        "quantization": "Q4_K_M",
        "file_size_bytes": 123456789,
    }
    
    with SparsifyDB(temp_db) as db:
        # Save profile run
        run_id = db.save_profile_run(profile)
        assert isinstance(run_id, str)
        assert len(run_id) == 32  # hex UUID
        
        # Get profile run
        fetched_profile = db.get_profile_run(run_id)
        assert fetched_profile is not None
        assert fetched_profile["model_path"] == "/path/to/mock_model.gguf"
        assert fetched_profile["architecture"] == "llama"
        assert fetched_profile["some_metric"] == 42.0
        
        # List runs
        runs = db.list_profile_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id
        assert runs[0]["model_path"] == "/path/to/mock_model.gguf"
        assert runs[0]["architecture"] == "llama"
        
        # Model metadata operations
        db.save_model_metadata("hash123", metadata)
        fetched_meta = db.get_model_metadata("hash123")
        assert fetched_meta is not None
        assert fetched_meta["architecture"] == "llama"
        assert fetched_meta["quantization"] == "Q4_K_M"
        assert fetched_meta["file_size_bytes"] == 123456789
        
        # Get non-existent metadata
        assert db.get_model_metadata("missing_hash") is None

def test_compute_file_hash(tmp_path: Path) -> None:
    """Test fast hashing of files."""
    test_file = tmp_path / "test_hash.txt"
    test_file.write_bytes(b"A" * (2 * 1024 * 1024))  # 2 MB of "A"
    
    hash_val = compute_file_hash(test_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64  # SHA-256 hex string
    
    # Hashing should depend only on first 1MB
    test_file_2 = tmp_path / "test_hash_2.txt"
    test_file_2.write_bytes(b"A" * (1024 * 1024) + b"B" * (1024 * 1024))  # 1MB A, 1MB B
    
    hash_val_2 = compute_file_hash(test_file_2)
    assert hash_val == hash_val_2  # first 1MB is identical
