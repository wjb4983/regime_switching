"""SQLite-backed registries for local experiment tracking."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from regime.logging import JsonValue, redact

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_groups (
    group_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT,
    metadata_json TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, group_id TEXT, name TEXT, status TEXT NOT NULL,
    config_hash TEXT, dataset_hash TEXT, feature_hash TEXT, model_hash TEXT,
    metadata_json TEXT NOT NULL, started_at REAL NOT NULL, updated_at REAL NOT NULL,
    completed_at REAL, checkpoint_path TEXT,
    FOREIGN KEY(group_id) REFERENCES experiment_groups(group_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
    path TEXT NOT NULL, hash TEXT, metadata_json TEXT NOT NULL, created_at REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
    model_hash TEXT NOT NULL, path TEXT, metadata_json TEXT NOT NULL, created_at REAL NOT NULL,
    UNIQUE(name, version)
);
CREATE TABLE IF NOT EXISTS results (
    result_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
    value_json TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""

ARTIFACT_KINDS = {
    "config",
    "predictions",
    "probabilities",
    "metrics",
    "model",
    "report",
    "plot",
    "log",
    "provenance",
    "manifest",
    "checkpoint",
}


def _json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(redact(dict(value or {})), sort_keys=True, default=repr)


@dataclass(frozen=True)
class ExperimentStore:
    """Local SQLite registry for runs, groups, artifacts, models, and results."""

    root: Path | str = Path("experiments")
    db_name: str = "experiments.sqlite3"

    def __post_init__(self) -> None:
        root = Path(self.root)
        object.__setattr__(self, "root", root)
        root.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @property
    def db_path(self) -> Path:
        """Return the local registry database path."""
        return Path(self.root) / self.db_name

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row dictionaries enabled."""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def create_group(self, name: str, description: str | None = None, **metadata: Any) -> str:
        """Create or update an experiment group and return its ID."""
        group_id = uuid.uuid4().hex
        with self.connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO experiment_groups VALUES (?, ?, ?, ?, ?)",
                (group_id, name, description, _json(metadata), time.time()),
            )
            row = con.execute(
                "SELECT group_id FROM experiment_groups WHERE name=?", (name,)
            ).fetchone()
        return str(row["group_id"])

    def create_run(
        self,
        *,
        group_id: str | None = None,
        name: str | None = None,
        hashes: Mapping[str, str | None] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register a new resumable run."""
        run_id = uuid.uuid4().hex
        hashes = hashes or {}
        now = time.time()
        with self.connect() as con:
            con.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    group_id,
                    name,
                    "running",
                    hashes.get("config_hash"),
                    hashes.get("dataset_hash"),
                    hashes.get("feature_hash"),
                    hashes.get("model_hash"),
                    _json(metadata),
                    now,
                    now,
                    None,
                    None,
                ),
            )
        return run_id

    def update_run(self, run_id: str, status: str, **fields: Any) -> None:
        """Update run status, completion time, and optional checkpoint path."""
        completed = time.time() if status in {"completed", "failed", "cancelled"} else None
        checkpoint_path = fields.get("checkpoint_path")
        with self.connect() as con:
            con.execute(
                "UPDATE runs SET status=?, updated_at=?, completed_at=COALESCE(?, completed_at), "
                "checkpoint_path=COALESCE(?, checkpoint_path) WHERE run_id=?",
                (status, time.time(), completed, checkpoint_path, run_id),
            )

    def update_hashes(self, run_id: str, **hashes: str | None) -> None:
        """Attach dataset, feature, and model hashes discovered during execution."""
        allowed = ("dataset_hash", "feature_hash", "model_hash")
        unknown = set(hashes) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown run hashes: {', '.join(sorted(unknown))}")
        with self.connect() as con:
            con.execute(
                "UPDATE runs SET dataset_hash=COALESCE(?, dataset_hash), "
                "feature_hash=COALESCE(?, feature_hash), model_hash=COALESCE(?, model_hash), "
                "updated_at=? WHERE run_id=?",
                (*(hashes.get(key) for key in allowed), time.time(), run_id),
            )

    def add_artifact(
        self,
        run_id: str,
        kind: str,
        path: str | Path,
        *,
        artifact_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register a run artifact path."""
        if kind not in ARTIFACT_KINDS:
            raise ValueError(f"Unknown artifact kind: {kind}")
        artifact_id = uuid.uuid4().hex
        with self.connect() as con:
            con.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact_id,
                    run_id,
                    kind,
                    str(path),
                    artifact_hash,
                    _json(metadata),
                    time.time(),
                ),
            )
        return artifact_id

    def register_model(
        self,
        name: str,
        version: str,
        model_hash: str,
        *,
        path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register a model version."""
        model_id = uuid.uuid4().hex
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    name,
                    version,
                    model_hash,
                    str(path) if path else None,
                    _json(metadata),
                    time.time(),
                ),
            )
        return model_id

    def add_result(
        self, run_id: str, kind: str, value: JsonValue, metadata: Mapping[str, Any] | None = None
    ) -> str:
        """Register metrics or other JSON-compatible result values."""
        result_id = uuid.uuid4().hex
        with self.connect() as con:
            con.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)",
                (
                    result_id,
                    run_id,
                    kind,
                    json.dumps(value, sort_keys=True),
                    _json(metadata),
                    time.time(),
                ),
            )
        return result_id

    def latest_run(self, name: str) -> sqlite3.Row | None:
        """Return the most recent run with a given name for resumption."""
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM runs WHERE name=? ORDER BY started_at DESC LIMIT 1", (name,)
            ).fetchone()
        return cast(sqlite3.Row | None, row)


__all__ = ["ARTIFACT_KINDS", "ExperimentStore"]
