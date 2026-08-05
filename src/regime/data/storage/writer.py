"""Immutable Parquet dataset writer with schema validation and content hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq

from regime.data.schemas import TabularSchema
from regime.data.storage.catalog import LocalCatalog
from regime.data.storage.manifest import DatasetManifest, DatasetPartition


@dataclass(frozen=True)
class DatasetWriteResult:
    """Result returned after writing or reusing a dataset partition."""

    path: Path
    manifest_path: Path
    manifest: DatasetManifest
    written: bool


class ParquetDatasetWriter:
    """Write schema-validated immutable Parquet partitions to local storage."""

    def __init__(self, root: Path | str, *, catalog_path: Path | str | None = None) -> None:
        self.root = Path(root)
        self.catalog = LocalCatalog(catalog_path or self.root / "catalog.sqlite")

    def write(
        self,
        data: object,
        *,
        schema: TabularSchema,
        partition: DatasetPartition,
    ) -> DatasetWriteResult:
        """Validate and write ``data`` idempotently for the given partition."""
        if partition.dataset != schema.name:
            raise ValueError("partition dataset must match schema name")
        schema.validate_strict(data)
        table = _to_arrow_table(data, schema)
        schema.validate_strict(table)
        partition_dir = partition.directory(self.root)
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / "data.parquet"
        manifest_path = partition_dir / "manifest.json"

        tmp_path = _write_temp_parquet(table, partition_dir, schema)
        content_hash = _sha256_file(tmp_path)
        columns = tuple(table.schema.names)
        manifest = DatasetManifest.build(
            partition=partition,
            schema=schema,
            parquet_file=parquet_path.name,
            content_hash=content_hash,
            row_count=table.num_rows,
            columns=columns,
        )

        if manifest_path.exists() and parquet_path.exists():
            existing = DatasetManifest.read_json(manifest_path)
            same_content = existing.content_hash == content_hash
            same_schema = existing.schema_fingerprint == manifest.schema_fingerprint
            if same_content and same_schema:
                tmp_path.unlink(missing_ok=True)
                self.catalog.upsert_manifest(self.root, existing)
                return DatasetWriteResult(parquet_path, manifest_path, existing, written=False)
            tmp_path.unlink(missing_ok=True)
            raise FileExistsError(
                f"immutable partition already exists with different content: {partition_dir}"
            )

        tmp_path.replace(parquet_path)
        manifest.write_json(manifest_path)
        self.catalog.upsert_manifest(self.root, manifest)
        return DatasetWriteResult(parquet_path, manifest_path, manifest, written=True)


def _to_arrow_table(data: object, schema: TabularSchema) -> pa.Table:
    if isinstance(data, pa.Table):
        table = data
    else:
        module = type(data).__module__
        if module.startswith("pandas"):
            table = pa.Table.from_pandas(data, preserve_index=False)
        elif module.startswith("polars"):
            table = data.to_arrow()  # type: ignore[attr-defined]
        else:
            try:
                table = data.arrow()  # type: ignore[attr-defined]
            except AttributeError as exc:
                message = "data must be a pandas, Polars, Arrow, or DuckDB table/relation"
                raise TypeError(message) from exc
    return table.cast(schema.to_arrow_schema(), safe=False)


def _write_temp_parquet(table: pa.Table, directory: Path, schema: TabularSchema) -> Path:
    with NamedTemporaryFile(
        prefix=".data-", suffix=".parquet.tmp", dir=directory, delete=False
    ) as fh:
        tmp_path = Path(fh.name)
    try:
        pq.write_table(table, tmp_path, version="2.6", compression="zstd", store_schema=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
