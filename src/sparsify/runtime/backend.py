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

    if system == "Darwin" and machine == "arm64":
        try:
            import mlx.core  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Apple Silicon detected but MLX is not installed. "
                "Reinstall with: pip install 'sparsify[mlx]'"
            ) from exc
        return Backend(name="mlx", device=f"Apple Silicon ({platform.processor() or machine})")

    if system == "Darwin":
        raise RuntimeError(
            "Sparsify requires Apple Silicon on macOS — MLX does not run on "
            "Intel Macs. CUDA (Linux/Windows) backends are on the roadmap."
        )
    raise RuntimeError(
        f"No Sparsify backend for {system}/{machine} yet. The MLX backend "
        "runs on Apple Silicon today; a CUDA/PyTorch backend for Linux and "
        "Windows is the next backend milestone (the paging architecture — "
        "expert store, cache, module surgery — is backend-agnostic by design). "
        "Track progress: https://github.com/daylinkltd/sparsify"
    )


def require() -> Backend:
    """Detect-or-die with a user-facing message (used by CLI entrypoints)."""
    return detect()
