"""Export utilities for Sparsify profiles."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sparsify.utils.config import EXPORT_DIR


def export_json(data: dict, output_path: Path) -> Path:
    """Write *data* as formatted JSON to *output_path* and return the path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, default=str))
    return output_path


def export_profile_json(
    profile_dict: dict,
    output_dir: Path | None = None,
) -> Path:
    """Export a profile dict to a timestamped JSON file.

    Returns the path to the written file.
    """
    output_dir = output_dir or EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    arch = profile_dict.get("architecture", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"profile_{arch}_{timestamp}.json"

    return export_json(profile_dict, output_dir / filename)
