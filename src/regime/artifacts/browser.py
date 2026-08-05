"""Read-only queries and loaders for local experiment artifacts.

This module deliberately has no UI dependency.  Reports, notebooks, and optional
applications can all use the same discovery and data-loading rules.
"""

from __future__ import annotations

import json
import mimetypes
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RunSummary:
    """A run available in an existing experiment registry."""

    run_id: str
    name: str
    status: str
    started_at: float
    model_hash: str | None
    metadata: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.name} · {self.run_id[:8]} · {self.status}"


@dataclass(frozen=True)
class ArtifactSummary:
    """Metadata and resolved location for a registered artifact."""

    artifact_id: str
    run_id: str
    kind: str
    path: Path
    created_at: float
    metadata: dict[str, Any]

    @property
    def filename(self) -> str:
        return self.path.name


class ArtifactBrowser:
    """Browse an existing experiment registry without modifying it."""

    def __init__(self, root: str | Path = "experiments", db_name: str = "experiments.sqlite3"):
        self.root = Path(root).expanduser().resolve()
        self.db_path = self.root / db_name
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Experiment registry not found: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _metadata(value: str) -> dict[str, Any]:
        decoded = json.loads(value or "{}")
        return decoded if isinstance(decoded, dict) else {}

    def runs(self) -> tuple[RunSummary, ...]:
        """Return available runs, newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, COALESCE(name, 'unnamed') AS name, status, started_at, "
                "model_hash, metadata_json FROM runs ORDER BY started_at DESC"
            ).fetchall()
        return tuple(
            RunSummary(
                run_id=str(row["run_id"]),
                name=str(row["name"]),
                status=str(row["status"]),
                started_at=float(row["started_at"]),
                model_hash=row["model_hash"],
                metadata=self._metadata(str(row["metadata_json"])),
            )
            for row in rows
        )

    def artifacts(self, run_ids: list[str] | tuple[str, ...]) -> tuple[ArtifactSummary, ...]:
        """Return registered artifacts for the selected runs."""
        if not run_ids:
            return ()
        placeholders = ",".join("?" for _ in run_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT artifact_id, run_id, kind, path, created_at, metadata_json "
                f"FROM artifacts WHERE run_id IN ({placeholders}) ORDER BY created_at DESC",
                tuple(run_ids),
            ).fetchall()
        return tuple(
            ArtifactSummary(
                artifact_id=str(row["artifact_id"]),
                run_id=str(row["run_id"]),
                kind=str(row["kind"]),
                path=self._resolve_path(str(row["path"])),
                created_at=float(row["created_at"]),
                metadata=self._metadata(str(row["metadata_json"])),
            )
            for row in rows
        )

    def results(self, run_ids: list[str] | tuple[str, ...]) -> pd.DataFrame:
        """Flatten stored JSON results into a comparison-ready table."""
        if not run_ids:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in run_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT run_id, kind, value_json FROM results "
                f"WHERE run_id IN ({placeholders}) ORDER BY created_at",
                tuple(run_ids),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            value = json.loads(str(row["value_json"]))
            record = {"run_id": str(row["run_id"]), "result": str(row["kind"])}
            if isinstance(value, dict):
                record.update(value)
            else:
                record["value"] = value
            records.append(record)
        return pd.DataFrame.from_records(records)

    def load_table(self, artifact: ArtifactSummary) -> pd.DataFrame:
        """Load a registered JSON/CSV/Parquet artifact as tabular data."""
        suffix = artifact.path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(artifact.path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(artifact.path)
        if suffix == ".json":
            payload = json.loads(artifact.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("records", payload)
            return pd.DataFrame(payload if isinstance(payload, list) else [payload])
        raise ValueError(f"Unsupported tabular artifact: {artifact.filename}")

    def download(self, artifact: ArtifactSummary) -> tuple[bytes, str]:
        """Return verified artifact bytes and a best-effort media type."""
        path = artifact.path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.read_bytes(), media_type

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()


def matching_artifacts(
    artifacts: tuple[ArtifactSummary, ...], *terms: str
) -> tuple[ArtifactSummary, ...]:
    """Find artifacts by kind, filename, or their optional ``view`` metadata."""
    def normalize(value: str) -> str:
        return value.casefold().replace("probabilities", "probability")

    needles = tuple(normalize(term) for term in terms)
    return tuple(
        artifact
        for artifact in artifacts
        if any(
            needle
            in normalize(
                " ".join(
                    (artifact.kind, artifact.filename, str(artifact.metadata.get("view", "")))
                )
            )
            for needle in needles
        )
    )
