"""Compute-backend detection.

Sparsify's inference core currently runs on MLX (Apple Silicon unified
memory). This module is the single place that decides which backend a
platform gets, so new backends (CUDA/Linux, CPU) slot in here without
touching callers.

Honesty note: platforms without a working backend get a clear, early
error — never a partially-working fallback.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    name: str          # "mlx" | (future: "cuda", "cpu")
    device: str        # human-readable device description


def detect() -> Backend:
    """Return the backend for this machine, or raise with a clear reason."""
    system = platform.system()
    machine = platform.machine()

    # 1. Try Apple Silicon MLX first
    if system == "Darwin" and machine == "arm64":
        try:
            import mlx.core  # noqa: F401
            return Backend(name="mlx", device=f"Apple Silicon ({platform.processor() or machine})")
        except ImportError:
            pass

    # 2. Try PyTorch Backend (CUDA, MPS, or CPU)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "NVIDIA GPU"
            return Backend(name="pytorch", device=f"CUDA: {device_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return Backend(name="pytorch", device="Apple Silicon (MPS via PyTorch)")
        else:
            return Backend(name="pytorch", device=f"CPU ({platform.processor() or machine})")
    except (ImportError, OSError):
        pass

    # 3. Fallback/Error if neither is found
    if system == "Darwin" and machine == "arm64":
        raise RuntimeError(
            "Apple Silicon detected. Please install MLX or PyTorch: "
            "pip install 'sparsify[mlx]' or pip install torch"
        )
    raise RuntimeError(
        f"Sparsify requires PyTorch (Windows/Linux/macOS) or MLX (macOS arm64). "
        f"Please install PyTorch to run on {system}/{machine}."
    )


def require() -> Backend:
    """Detect-or-die with a user-facing message (used by CLI entrypoints)."""
    return detect()
