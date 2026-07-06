"""System information collector for Sparsify."""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field

import psutil
from rich.console import Console
from rich.table import Table


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SystemInfo:
    """Snapshot of the host machine's hardware and software environment."""

    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    arch: str = ""
    cpu_name: str = "Unknown"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    ram_used_bytes: int = 0
    gpu_name: str | None = None
    gpu_memory_bytes: int | None = None
    metal_available: bool = False
    cuda_available: bool = False
    python_version: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "hostname": self.hostname,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "arch": self.arch,
            "cpu_name": self.cpu_name,
            "cpu_cores_physical": self.cpu_cores_physical,
            "cpu_cores_logical": self.cpu_cores_logical,
            "ram_total_bytes": self.ram_total_bytes,
            "ram_available_bytes": self.ram_available_bytes,
            "ram_used_bytes": self.ram_used_bytes,
            "gpu_name": self.gpu_name,
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "metal_available": self.metal_available,
            "cuda_available": self.cuda_available,
            "python_version": self.python_version,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_to_gb(b: int) -> str:
    return f"{b / (1024 ** 3):.1f} GB"


def _macos_cpu_name() -> str:
    """Read the CPU brand string on macOS via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "Unknown"


def _linux_nvidia_gpu() -> tuple[str | None, int | None]:
    """Attempt to detect an NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            name = parts[0].strip()
            mem_mib = int(parts[1].strip())
            return name, mem_mib * 1024 * 1024
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_system_info() -> SystemInfo:
    """Collect a :class:`SystemInfo` snapshot of the current host."""
    info = SystemInfo()

    # Basic platform info
    try:
        info.hostname = platform.node()
        info.os_name = platform.system()
        info.os_version = platform.release()
        info.arch = platform.machine()
        info.python_version = platform.python_version()
    except Exception:
        pass

    # CPU cores (psutil)
    try:
        info.cpu_cores_physical = psutil.cpu_count(logical=False) or 0
        info.cpu_cores_logical = psutil.cpu_count(logical=True) or 0
    except Exception:
        pass

    # RAM (psutil)
    try:
        mem = psutil.virtual_memory()
        info.ram_total_bytes = mem.total
        info.ram_available_bytes = mem.available
        info.ram_used_bytes = mem.used
    except Exception:
        pass

    # Platform-specific GPU / CPU detection
    if info.os_name == "Darwin":
        info.cpu_name = _macos_cpu_name()

        # Apple Silicon → Metal + unified memory
        if info.arch == "arm64":
            info.metal_available = True
            info.gpu_name = f"{info.cpu_name} GPU"
            info.gpu_memory_bytes = info.ram_total_bytes  # unified memory

    elif info.os_name == "Linux":
        # Try /proc/cpuinfo for CPU name
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info.cpu_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        # NVIDIA GPU
        gpu_name, gpu_mem = _linux_nvidia_gpu()
        if gpu_name:
            info.gpu_name = gpu_name
            info.gpu_memory_bytes = gpu_mem
            info.cuda_available = True

    return info


def format_system_info(info: SystemInfo) -> str:
    """Render a :class:`SystemInfo` as a Rich table captured to a string."""
    table = Table(
        title="System Information",
        show_header=False,
        title_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("OS", f"{info.os_name} {info.os_version} ({info.arch})")
    table.add_row("Hostname", info.hostname)
    table.add_row("Python", info.python_version)
    table.add_row("CPU", info.cpu_name)
    table.add_row(
        "Cores",
        f"{info.cpu_cores_physical} physical / {info.cpu_cores_logical} logical",
    )
    table.add_row(
        "RAM",
        f"{_bytes_to_gb(info.ram_used_bytes)} used / {_bytes_to_gb(info.ram_total_bytes)} total "
        f"({_bytes_to_gb(info.ram_available_bytes)} available)",
    )

    gpu_text = info.gpu_name or "None detected"
    if info.gpu_memory_bytes:
        gpu_text += f" ({_bytes_to_gb(info.gpu_memory_bytes)})"
    table.add_row("GPU", gpu_text)

    accel_parts: list[str] = []
    if info.metal_available:
        accel_parts.append("Metal ✓")
    if info.cuda_available:
        accel_parts.append("CUDA ✓")
    table.add_row("Accelerators", ", ".join(accel_parts) if accel_parts else "None")

    console = Console(record=True, width=80)
    console.print(table)
    return console.export_text()
