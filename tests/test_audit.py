"""Unit tests verifying correct scientific audit truth table generation."""
from __future__ import annotations

from pathlib import Path


def test_truth_table_existence_and_format() -> None:
    """Verify that the scientific audit truth table exists and is correctly structured."""
    audit_path = Path("/Users/swaraj/.gemini/antigravity/brain/1a771cdb-36b8-4bcd-89d3-c3f1fe6eeb11/truth_table.md")
    
    # Assert file exists
    assert audit_path.exists()
    
    # Assert format columns are present
    content = audit_path.read_text()
    assert "Metric" in content
    assert "Value" in content
    assert "Source Type" in content
    assert "Confidence" in content
    assert "Requires Real Validation" in content
    
    # Assert key metrics are audited
    assert "Shannon Entropy" in content
    assert "W90 Working Set Size" in content
    assert "Reuse Distance" in content
    assert "Cache Hit Ratios" in content
    assert "RAM (30B Model)" in content
    assert "RAM (70B Model)" in content
    assert "RAM (120B Model)" in content
    assert "Throughput Projections" in content
