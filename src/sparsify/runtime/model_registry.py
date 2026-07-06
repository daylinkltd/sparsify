"""
Sparsify model registry — tracks models pulled onto this machine.

Stored as a JSON file at models/.registry.json on the external SSD.
All entries are real: every model in the registry was actually downloaded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

MODELS_DIR = Path("/Volumes/projects/sparsify/models")
REGISTRY_FILE = MODELS_DIR / ".registry.json"

# ── Well-known MLX model aliases ─────────────────────────────────────────────
# Only 4-bit MLX quantised models are listed — they run on Apple Silicon via the
# Neural Engine and are the most RAM-efficient format available today.
KNOWN_ALIASES: dict[str, str] = {
    # MoE models (Sparsify primary targets)
    "mixtral:8x7b":          "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
    "mixtral:8x7b-instruct": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
    "mixtral:8x7b-4bit":     "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
    # Dense models (supported for completeness, not the primary use-case)
    "llama:1b":         "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama:3b":         "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "llama:8b":         "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "qwen:7b":          "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen:coder-7b":    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "qwen:30b":         "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "qwen:30b-a3b":     "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "mistral:7b":       "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}


def resolve_hf_id(model_tag: str) -> str:
    """Return the HuggingFace repo id for *model_tag*.

    Accepts a Sparsify alias (e.g. ``mixtral:8x7b``) or a raw HF repo id
    (e.g. ``mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit``).
    """
    return KNOWN_ALIASES.get(model_tag.lower(), model_tag)


def _load() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def _save(data: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(data, indent=2))


def register(hf_id: str, local_path: Path, size_bytes: int) -> None:
    """Add or update a model entry after a successful pull."""
    data = _load()
    data[hf_id] = {
        "hf_id": hf_id,
        "local_path": str(local_path),
        "size_gb": round(size_bytes / 1e9, 2),
        "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save(data)


def remove(hf_id: str) -> None:
    data = _load()
    data.pop(hf_id, None)
    _save(data)


def all_models() -> list[dict]:
    """Return all registered models, verifying the local path still exists."""
    data = _load()
    result = []
    for entry in data.values():
        p = Path(entry["local_path"])
        entry["available"] = p.exists()
        result.append(entry)
    return result


def get(hf_id: str) -> Optional[dict]:
    return _load().get(hf_id)
