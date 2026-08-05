"""Tests for immutable local Parquet dataset storage."""

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from regime.data.schemas import SchemaField, SchemaName, TabularSchema
from regime.data.storage import DatasetPartition, DuckDBQueryAdapter, ParquetDatasetWriter


def _schema() -> TabularSchema:
    return TabularSchema(
        name=SchemaName.EQUITY_ETF_OHLCV,
        fields=(
            SchemaField(name="symbol", kind="string"),
            SchemaField(name="event_ts", kind="timestamp"),
            SchemaField(name="source_id", kind="string"),
            SchemaField(name="close", kind="float"),
        ),
        primary_key=("symbol", "event_ts", "source_id"),
        description="test schema",
    )


def _table() -> pa.Table:
    return pa.table(
        {
            "symbol": ["SPY", "QQQ"],
            "event_ts": pa.array(
                [
                    datetime(2026, 8, 5, 14, 30, tzinfo=UTC),
                    datetime(2026, 8, 5, 14, 30, tzinfo=UTC),
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "source_id": ["unit", "unit"],
            "close": [500.0, 400.0],
        }
    )


def _partition() -> DatasetPartition:
    return DatasetPartition(
        dataset=SchemaName.EQUITY_ETF_OHLCV,
        source="unit",
        asset_class="equity",
        date=date(2026, 8, 5),
        version="v1",
    )


def test_writer_creates_partition_manifest_hash_and_catalog(tmp_path) -> None:
    writer = ParquetDatasetWriter(tmp_path)

    result = writer.write(_table(), schema=_schema(), partition=_partition())

    assert result.written is True
    assert result.path.exists()
    assert result.manifest_path.exists()
    assert result.manifest.content_hash
    assert result.manifest.row_count == 2
    entries = writer.catalog.list_entries(dataset=SchemaName.EQUITY_ETF_OHLCV.value)
    assert len(entries) == 1
    assert entries[0].content_hash == result.manifest.content_hash


def test_writer_is_idempotent_for_identical_content(tmp_path) -> None:
    writer = ParquetDatasetWriter(tmp_path)
    first = writer.write(_table(), schema=_schema(), partition=_partition())

    second = writer.write(_table(), schema=_schema(), partition=_partition())

    assert first.written is True
    assert second.written is False
    assert second.manifest.content_hash == first.manifest.content_hash


def test_writer_rejects_different_content_for_existing_partition(tmp_path) -> None:
    writer = ParquetDatasetWriter(tmp_path)
    writer.write(_table(), schema=_schema(), partition=_partition())
    changed = _table().set_column(3, "close", pa.array([501.0, 401.0]))

    with pytest.raises(FileExistsError):
        writer.write(changed, schema=_schema(), partition=_partition())


def test_duckdb_adapter_reads_lazily_with_column_projection(tmp_path) -> None:
    writer = ParquetDatasetWriter(tmp_path)
    writer.write(_table(), schema=_schema(), partition=_partition())
    adapter = DuckDBQueryAdapter(tmp_path)

    relation = adapter.relation(
        schema=_schema(), partition=_partition(), columns=["symbol", "close"]
    )

    assert relation.columns == ["symbol", "close"]
    assert relation.order("symbol").fetchall() == [("QQQ", 400.0), ("SPY", 500.0)]
