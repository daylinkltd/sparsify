"""Sparsify configuration and paths."""
from __future__ import annotations

from pathlib import Path

SPARSIFY_DIR = Path.home() / '.sparsify'
DB_PATH = SPARSIFY_DIR / 'sparsify.db'
EXPORT_DIR = SPARSIFY_DIR / 'exports'


def ensure_dirs() -> None:
    """Create Sparsify directories if they don't exist."""
    SPARSIFY_DIR.mkdir(exist_ok=True)
    EXPORT_DIR.mkdir(exist_ok=True)
