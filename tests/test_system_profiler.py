"""Unit tests for the system capabilities profiler."""
from __future__ import annotations

from sparsify.profiler.system_profiler import get_system_info, format_system_info, SystemInfo

def test_system_info_collection() -> None:
    """Test that we can collect system information without errors."""
    info = get_system_info()
    
    assert isinstance(info, SystemInfo)
    assert info.os_name in ("Darwin", "Linux", "Windows", "Unknown")
    assert info.cpu_cores_logical >= 0
    assert info.cpu_cores_physical >= 0
    assert info.ram_total_bytes >= 0
    assert isinstance(info.python_version, str)
    
    # Check serialization
    d = info.to_dict()
    assert isinstance(d, dict)
    assert d["os_name"] == info.os_name
    assert d["cpu_cores_logical"] == info.cpu_cores_logical

def test_format_system_info() -> None:
    """Test rich formatting of system info."""
    info = SystemInfo(
        hostname="test-host",
        os_name="Darwin",
        os_version="15.0",
        arch="arm64",
        cpu_name="Apple M5",
        cpu_cores_physical=8,
        cpu_cores_logical=10,
        ram_total_bytes=16 * 1024**3,
        ram_available_bytes=8 * 1024**3,
        ram_used_bytes=8 * 1024**3,
        gpu_name="Apple M5 GPU",
        gpu_memory_bytes=16 * 1024**3,
        metal_available=True,
        python_version="3.12.0",
    )
    
    output = format_system_info(info)
    assert "System Information" in output
    assert "Darwin" in output
    assert "Apple M5" in output
    assert "16.0 GB" in output
    assert "Metal ✓" in output
