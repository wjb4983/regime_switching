"""SQLite-backed local catalog for immutable dataset partitions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from regime.data.storage.manifest import DatasetManifest


@dataclass(frozen=True)
class CatalogEntry:
    """Catalog row for one materialized partition."""

    dataset: str
    source: str
    asset_class: str
    date: str
    version: str
    path: str
    manifest_path: str
    content_hash: str
    schema_name: str
    row_count: int
    created_at: str


class LocalCatalog:
    """Small SQLite catalog that tracks local immutable Parquet partitions."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS partitions (
                    dataset TEXT NOT NULL,
                    source TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    date TEXT NOT NULL,
                    version TEXT NOT NULL,
                    path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    schema_name TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (dataset, source, asset_class, date, version)
                )
                """
            )

    def upsert_manifest(self, root: Path, manifest: DatasetManifest) -> None:
        """Insert or update a partition row from its manifest."""
        record = manifest.catalog_record(root)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO partitions VALUES (
                    :dataset, :source, :asset_class, :date, :version, :path, :manifest_path,
                    :content_hash, :schema_name, :row_count, :created_at
                )
                ON CONFLICT(dataset, source, asset_class, date, version) DO UPDATE SET
                    path=excluded.path,
                    manifest_path=excluded.manifest_path,
                    content_hash=excluded.content_hash,
                    schema_name=excluded.schema_name,
                    row_count=excluded.row_count,
                    created_at=excluded.created_at
                """,
                record,
            )

    def list_entries(self, *, dataset: str | None = None) -> tuple[CatalogEntry, ...]:
        """Return catalog entries, optionally filtered by dataset name."""
        sql = "SELECT * FROM partitions"
        params: tuple[str, ...] = ()
        if dataset is not None:
            sql += " WHERE dataset = ?"
            params = (dataset,)
        sql += " ORDER BY dataset, source, asset_class, date, version"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return tuple(CatalogEntry(*row) for row in rows)
