"""
Sparsify model registry — tracks models pulled onto this machine.

Model directory resolution (first match wins):
  1. ``SPARSIFY_MODELS_DIR`` environment variable
  2. ``models_dir`` in ``~/.sparsify/config.json``
  3. ``~/.sparsify/models``

The registry index (.registry.json) is a cache, not the source of truth:
``all_models()`` reconciles against what is actually on disk, so models
survive a lost or stale index and foreign directories are picked up.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".sparsify"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _resolve_models_dir() -> Path:
    env = os.environ.get("SPARSIFY_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    if CONFIG_FILE.exists():
        try:
            configured = json.loads(CONFIG_FILE.read_text()).get("models_dir")
            if configured:
                return Path(configured).expanduser()
        except (json.JSONDecodeError, OSError):
            pass
    return CONFIG_DIR / "models"


MODELS_DIR = _resolve_models_dir()
REGISTRY_FILE = MODELS_DIR / ".registry.json"

# ── Well-known MLX model aliases ─────────────────────────────────────────────
# 4-bit MLX quantised models (Apple Silicon). "moe" marks storage-backed
# expert paging targets; dense models load fully and pass through unmodified.
# "tested": verified end-to-end on Sparsify's dev hardware. Untested entries
# are real repos (existence checked) using the same mlx-lm MoE structure the
# runtime detects generically — flagged honestly until someone runs them.
KNOWN_ALIASES: dict[str, dict] = {
    # MoE models (Sparsify primary targets) — verified on Sparsify
    "olmoe:1b-7b":           {"hf": "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit", "moe": True, "tested": True, "gb": 3.9},
    "qwen:30b":              {"hf": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "moe": True, "tested": True, "gb": 16.3},
    "qwen:30b-a3b":          {"hf": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "moe": True, "tested": True, "gb": 16.3},
    "mixtral:8x7b":          {"hf": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", "moe": True, "tested": True, "gb": 26.3},
    "mixtral:8x7b-instruct": {"hf": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", "moe": True, "tested": True, "gb": 26.3},
    # MoE models — same architecture family, not yet run on Sparsify
    "qwen1.5:moe-a2.7b":     {"hf": "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit", "moe": True, "tested": False, "gb": 8.5},
    "deepseek:v2-lite":      {"hf": "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx", "moe": True, "tested": False, "gb": 8.8},
    "deepseek:coder-v2-lite": {"hf": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit", "moe": True, "tested": False, "gb": 8.8},
    "phi:3.5-moe":           {"hf": "mlx-community/Phi-3.5-MoE-instruct-4bit", "moe": True, "tested": False, "gb": 23.6},
    "hunyuan:a13b":          {"hf": "mlx-community/Hunyuan-A13B-Instruct-4bit", "moe": True, "tested": False, "gb": 45.2},
    "glm:4.5-air":           {"hf": "mlx-community/GLM-4.5-Air-4bit", "moe": True, "tested": False, "gb": 60.2},
    "mixtral:8x22b":         {"hf": "mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit", "moe": True, "tested": False, "gb": 79.1},
    "qwen:235b-a22b":        {"hf": "mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit", "moe": True, "tested": False, "gb": 132.3},
    # Dense models (supported for completeness, not the primary use-case)
    "llama:1b":         {"hf": "mlx-community/Llama-3.2-1B-Instruct-4bit", "moe": False, "tested": True, "gb": 0.7},
    "llama:3b":         {"hf": "mlx-community/Llama-3.2-3B-Instruct-4bit", "moe": False, "tested": False, "gb": 1.8},
    "llama:8b":         {"hf": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "moe": False, "tested": False, "gb": 4.5},
    "qwen:7b":          {"hf": "mlx-community/Qwen2.5-7B-Instruct-4bit", "moe": False, "tested": False, "gb": 4.3},
    "qwen:coder-7b":    {"hf": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", "moe": False, "tested": False, "gb": 4.3},
    "mistral:7b":       {"hf": "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "moe": False, "tested": False, "gb": 4.1},
}


def resolve_hf_id(model_tag: str) -> str:
    """Return the HuggingFace repo id for *model_tag*.

    Accepts a Sparsify alias (e.g. ``mixtral:8x7b``) or a raw HF repo id
    (e.g. ``mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit``).
    """
    entry = KNOWN_ALIASES.get(model_tag.lower())
    return entry["hf"] if entry else model_tag


def models_dir_status() -> str:
    """State of the configured models directory.

    "ok"        — exists and is readable
    "unmounted" — configured on a volume that is not mounted right now
                  (external SSD unplugged); creating it would silently
                  write to the boot disk, so callers must not mkdir
    "missing"   — simply not created yet (fresh install)

    Symlink-aware (a dangling ~/.sparsify/models -> /Volumes/T7/… reads as
    unmounted, not missing) and stale-mountpoint-aware (macOS can leave an
    empty /Volumes/<name> directory behind after an unclean eject).
    """
    real = Path(os.path.realpath(MODELS_DIR))
    parts = real.parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "Volumes":
        volume = Path(*parts[:3])
        if not volume.exists() or not os.path.ismount(volume):
            return "unmounted"
    return "ok" if MODELS_DIR.exists() and real.exists() else "missing"


def suggest_alias(model_tag: str) -> str | None:
    """Closest known alias for a mistyped tag (e.g. qwen3:30b -> qwen:30b)."""
    import difflib

    tag = model_tag.lower()
    close = difflib.get_close_matches(tag, list(KNOWN_ALIASES), n=1, cutoff=0.6)
    if close:
        return close[0]
    # fall back to matching the part after the colon ("...:30b")
    if ":" in tag:
        suffix = tag.split(":", 1)[1]
        for alias in KNOWN_ALIASES:
            if alias.endswith(":" + suffix):
                return alias
    return None


def alias_for(hf_id: str) -> str | None:
    """Shortest known alias for an HF repo id, if any."""
    matches = [a for a, e in KNOWN_ALIASES.items() if e["hf"] == hf_id]
    return min(matches, key=len) if matches else None


def resolve_local(model_tag: str) -> tuple[str, Path] | None:
    """Resolve *model_tag* against models actually on disk, forgivingly.

    Order: exact alias/hf-id -> unique case-insensitive substring of a
    local model id (so ``Qwen3-30B-A3B-Instruct-2507-4bit`` or ``qwen3``
    find ``mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit``).
    Returns (hf_id, local_path), or None with no unique match.
    """
    hf_id = resolve_hf_id(model_tag)
    exact = MODELS_DIR / hf_id.replace("/", "--")
    if (exact / "config.json").exists():
        return hf_id, exact

    needle = model_tag.lower().replace("/", "--")
    candidates = [
        d for d in sorted(MODELS_DIR.iterdir())
        if d.is_dir() and (d / "config.json").exists()
        and needle in d.name.lower()
    ] if MODELS_DIR.exists() else []
    if len(candidates) == 1:
        d = candidates[0]
        return d.name.replace("--", "/"), d
    return None


def _load() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
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
    """Registered models reconciled with the models directory on disk.

    Unregistered-but-present model directories are adopted (self-healing
    after a lost index); registered-but-absent entries are flagged.
    """
    data = _load()
    adopted = False
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if not d.is_dir() or not (d / "config.json").exists():
                continue
            hf_id = d.name.replace("--", "/")
            if hf_id not in data:
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                data[hf_id] = {
                    "hf_id": hf_id,
                    "local_path": str(d),
                    "size_gb": round(size / 1e9, 2),
                    "pulled_at": "",
                }
                adopted = True
    if adopted:
        _save(data)

    result = []
    for entry in data.values():
        p = Path(entry["local_path"])
        entry["available"] = (p / "config.json").exists()
        result.append(entry)
    return result


def get(hf_id: str) -> Optional[dict]:
    return _load().get(hf_id)
