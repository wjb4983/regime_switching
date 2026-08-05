"""Manifest models for immutable Parquet dataset partitions."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from regime.data.schemas import SchemaName, TabularSchema


class DatasetPartition(BaseModel):
    """Logical partition coordinates for a local dataset slice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: SchemaName
    source: str = Field(min_length=1)
    asset_class: str = Field(min_length=1)
    date: date
    version: str = Field(min_length=1)

    @field_validator("source", "asset_class", "version")
    @classmethod
    def _reject_path_separators(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("partition values must be plain path segments")
        return value

    def directory(self, root: Path) -> Path:
        """Return the Hive-style directory for this partition under ``root``."""
        return root.joinpath(
            f"dataset={self.dataset.value}",
            f"source={self.source}",
            f"asset_class={self.asset_class}",
            f"date={self.date.isoformat()}",
            f"version={self.version}",
        )


class DatasetManifest(BaseModel):
    """Sidecar metadata proving partition contents, schema, and immutability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition: DatasetPartition
    schema_name: SchemaName
    schema_fingerprint: str
    parquet_file: str
    content_hash: str
    row_count: int
    columns: tuple[str, ...]
    created_at: datetime

    @classmethod
    def build(
        cls,
        *,
        partition: DatasetPartition,
        schema: TabularSchema,
        parquet_file: str,
        content_hash: str,
        row_count: int,
        columns: tuple[str, ...],
    ) -> DatasetManifest:
        """Create a manifest from freshly written dataset metadata."""
        return cls(
            partition=partition,
            schema_name=schema.name,
            schema_fingerprint=schema.model_dump_json(),
            parquet_file=parquet_file,
            content_hash=content_hash,
            row_count=row_count,
            columns=columns,
            created_at=datetime.now(tz=UTC),
        )

    def write_json(self, path: Path) -> None:
        """Atomically write this manifest to ``path``."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def read_json(cls, path: Path) -> DatasetManifest:
        """Load a manifest sidecar from disk."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def catalog_record(self, root: Path) -> dict[str, Any]:
        """Return a serializable record suitable for the local catalog."""
        dataset_dir = self.partition.directory(root)
        return {
            "dataset": self.partition.dataset.value,
            "source": self.partition.source,
            "asset_class": self.partition.asset_class,
            "date": self.partition.date.isoformat(),
            "version": self.partition.version,
            "path": str(dataset_dir / self.parquet_file),
            "manifest_path": str(dataset_dir / "manifest.json"),
            "content_hash": self.content_hash,
            "schema_name": self.schema_name.value,
            "row_count": self.row_count,
            "created_at": self.created_at.isoformat(),
        }
