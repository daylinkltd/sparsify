"""SQLite persistence layer for Sparsify profile runs and model metadata."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

from sparsify.utils.config import DB_PATH

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS profile_runs (
    run_id TEXT PRIMARY KEY,
    model_path TEXT NOT NULL,
    model_hash TEXT,
    architecture TEXT,
    timestamp TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_metadata (
    model_hash TEXT PRIMARY KEY,
    model_path TEXT,
    architecture TEXT,
    layer_count INTEGER,
    head_count INTEGER,
    embedding_dim INTEGER,
    quantization TEXT,
    file_size_bytes INTEGER,
    metadata_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profile_runs_model ON profile_runs(model_path);
CREATE INDEX IF NOT EXISTS idx_profile_runs_timestamp ON profile_runs(timestamp);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MB


def compute_file_hash(path: Path) -> str:
    """Return a SHA-256 hex digest of the first 1 MB of *path* for speed."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        sha.update(fh.read(_HASH_CHUNK_SIZE))
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class SparsifyDB:
    """Thin SQLite wrapper for persisting profile runs and model metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    # -- schema --------------------------------------------------------------

    def _migrate(self) -> None:
        """Apply the current schema (idempotent)."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # -- profile runs --------------------------------------------------------

    def save_profile_run(
        self,
        profile: dict,
        run_id: str | None = None,
    ) -> str:
        """Persist a profile run and return the *run_id*."""
        run_id = run_id or uuid.uuid4().hex
        model_path = profile.get("model_path", "")
        model_hash = profile.get("model_hash")
        architecture = profile.get("architecture")
        timestamp = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO profile_runs
                (run_id, model_path, model_hash, architecture, timestamp, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                str(model_path),
                model_hash,
                architecture,
                timestamp,
                json.dumps(profile, default=str),
            ),
        )
        self._conn.commit()
        return run_id

    def get_profile_run(self, run_id: str) -> dict | None:
        """Fetch a single profile run by *run_id*, or ``None``."""
        row = self._conn.execute(
            "SELECT metrics_json FROM profile_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["metrics_json"])

    def list_profile_runs(self, limit: int = 20) -> list[dict]:
        """Return recent profile run summaries (newest first)."""
        rows = self._conn.execute(
            """
            SELECT run_id, model_path, architecture, timestamp
            FROM profile_runs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- model metadata ------------------------------------------------------

    def save_model_metadata(self, model_hash: str, metadata: dict) -> None:
        """Insert or replace model metadata keyed by *model_hash*."""
        updated_at = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO model_metadata
                (model_hash, model_path, architecture, layer_count,
                 head_count, embedding_dim, quantization, file_size_bytes,
                 metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_hash,
                metadata.get("model_path"),
                metadata.get("architecture"),
                metadata.get("layer_count"),
                metadata.get("head_count"),
                metadata.get("embedding_dim"),
                metadata.get("quantization"),
                metadata.get("file_size_bytes"),
                json.dumps(metadata, default=str),
                updated_at,
            ),
        )
        self._conn.commit()

    def get_model_metadata(self, model_hash: str) -> dict | None:
        """Fetch model metadata by *model_hash*, or ``None``."""
        row = self._conn.execute(
            "SELECT metadata_json FROM model_metadata WHERE model_hash = ?",
            (model_hash,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["metadata_json"])

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> SparsifyDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
