"""Unit and integration tests for storage hardware audit and sensitivity maps."""
from __future__ import annotations

import pytest
from pathlib import Path

from sparsify.runtime.storage_audit import StorageAuditor


def test_storage_auditor_detection_and_solvers() -> None:
    """Validate RAM/SSD hardware parameter capture and sensitivity solvers."""
    auditor = StorageAuditor()
    
    # 1. Run detection
    hw = auditor.detect_hardware()
    assert hw["ram_total_gb"] > 0.0
    assert hw["ssd_capacity_gb"] > 0.0
    assert hw["ssd_free_gb"] > 0.0
    assert len(hw["ssd_device"]) > 0
    assert len(hw["protocol"]) > 0
    assert hw["uncached_read_speed"] > 0.0
    
    # 2. Run sensitivity solver
    sensitivity = auditor.solve_sensitivity(hit_ratio=0.95)
    assert "SATA_SSD" in sensitivity
    assert "USB_SSD" in sensitivity
    assert "Thunderbolt_SSD" in sensitivity
    assert "Internal_NVMe_SSD" in sensitivity
    for k, v in sensitivity.items():
        assert v["raw_tok_sec"] > 0.0
        assert v["cached_tok_sec"] > 0.0
        assert v["cached_tok_sec"] > v["raw_tok_sec"]
