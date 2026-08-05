"""DuckDB query adapter for local Parquet datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

from regime.data.schemas import TabularSchema
from regime.data.storage.manifest import DatasetManifest, DatasetPartition
from regime.data.storage.writer import _sha256_file


class DuckDBQueryAdapter:
    """Read immutable Parquet partitions lazily with projection and validation."""

    def __init__(self, root: Path | str, *, database: Path | str = ":memory:") -> None:
        self.root = Path(root)
        self.connection = duckdb.connect(str(database))

    def relation(
        self,
        *,
        schema: TabularSchema,
        partition: DatasetPartition | None = None,
        columns: Sequence[str] | None = None,
    ) -> duckdb.DuckDBPyRelation:
        """Return a lazy DuckDB relation over one partition or all dataset partitions."""
        paths = self._paths(schema=schema, partition=partition)
        projection = _projection(columns)
        glob = paths[0] if len(paths) == 1 else paths
        rel = self.connection.sql(f"SELECT {projection} FROM read_parquet(?)", params=[glob])
        _validate_projected_columns(schema, rel.columns, columns)
        validation_schema = _project_schema(schema, rel.columns)
        validation_schema.validate_frame(rel, strict=False).raise_for_errors()
        return rel

    def query(self, sql: str, *, view_name: str, relation: duckdb.DuckDBPyRelation) -> Any:
        """Register ``relation`` as ``view_name`` and execute ``sql`` lazily in DuckDB."""
        relation.create_view(view_name, replace=True)
        return self.connection.sql(sql)

    def _paths(self, *, schema: TabularSchema, partition: DatasetPartition | None) -> list[str]:
        if partition is not None:
            manifest_path = partition.directory(self.root) / "manifest.json"
            manifest = DatasetManifest.read_json(manifest_path)
            if manifest.schema_name != schema.name:
                raise ValueError("manifest schema does not match requested schema")
            parquet_path = partition.directory(self.root) / manifest.parquet_file
            if _sha256_file(parquet_path) != manifest.content_hash:
                raise ValueError("parquet content hash does not match manifest")
            return [str(parquet_path)]
        pattern = f"dataset={schema.name.value}/**/data.parquet"
        return [str(path) for path in self.root.glob(pattern)]


def _projection(columns: Sequence[str] | None) -> str:
    if columns is None:
        return "*"
    return ", ".join(f'"{column}"' for column in columns)


def _validate_projected_columns(
    schema: TabularSchema, relation_columns: Sequence[str], requested: Sequence[str] | None
) -> None:
    declared = set(schema.column_names)
    unknown = tuple(column for column in relation_columns if column not in declared)
    if unknown:
        raise ValueError(f"read returned columns outside schema: {unknown}")
    if requested is not None:
        missing = tuple(column for column in requested if column not in relation_columns)
        if missing:
            raise ValueError(f"requested columns are missing: {missing}")


def _project_schema(schema: TabularSchema, columns: Sequence[str]) -> TabularSchema:
    fields = tuple(field for field in schema.fields if field.name in set(columns))
    primary_key = tuple(column for column in schema.primary_key if column in set(columns))
    return TabularSchema(
        name=schema.name,
        fields=fields,
        primary_key=primary_key,
        description=schema.description,
    )
