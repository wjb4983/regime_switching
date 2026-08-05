"""Immutable local Parquet storage with manifests, schema checks, and DuckDB reads."""

from regime.data.storage.catalog import CatalogEntry, LocalCatalog
from regime.data.storage.duckdb import DuckDBQueryAdapter
from regime.data.storage.manifest import DatasetManifest, DatasetPartition
from regime.data.storage.writer import DatasetWriteResult, ParquetDatasetWriter

__all__ = [
    "CatalogEntry",
    "DatasetManifest",
    "DatasetPartition",
    "DatasetWriteResult",
    "DuckDBQueryAdapter",
    "LocalCatalog",
    "ParquetDatasetWriter",
]
