"""Tests for built-in market-data schema metadata and validation helpers."""

from datetime import UTC, datetime

import pytest

from regime.data.schemas import (
    EQUITY_ETF_OHLCV_SCHEMA,
    SCHEMAS,
    SchemaName,
    SchemaValidationError,
)


class _FakeSeries:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def dropna(self) -> "_FakeSeries":
        return _FakeSeries([value for value in self._values if value is not None])

    def unique(self) -> "_FakeSeries":
        return _FakeSeries(list(dict.fromkeys(self._values)))

    def tolist(self) -> list[object]:
        return self._values


class _FakePandasFrame:
    __module__ = "pandas.core.frame"

    def __init__(self, data: dict[str, list[object]], dtypes: dict[str, str]) -> None:
        self._data = data
        self.columns = list(data)
        self.dtypes = dtypes

    def __len__(self) -> int:
        return len(next(iter(self._data.values()))) if self._data else 0

    def __getitem__(self, column: str) -> _FakeSeries:
        return _FakeSeries(self._data[column])


def test_all_requested_schemas_are_registered() -> None:
    """Every canonical dataset family should have a strict schema definition."""
    assert set(SCHEMAS) == set(SchemaName)
    assert len(SCHEMAS) == 18


def test_common_lineage_columns_are_present() -> None:
    """Schemas should carry point-in-time metadata and quality lineage columns."""
    required_metadata = {
        "event_ts",
        "publication_ts",
        "vendor_received_ts",
        "effective_ts",
        "market_session",
        "source_id",
        "ingested_at",
        "revision_id",
        "data_quality_flags",
    }

    for schema in SCHEMAS.values():
        assert required_metadata.issubset(schema.column_names)


def test_equity_schema_validates_pandas_compatible_frame() -> None:
    """Pandas-like frames should validate against columns, types, and enums."""
    now = datetime(2026, 8, 5, tzinfo=UTC)
    data = {
        "symbol": ["SPY"],
        "security_id": ["US78462F1030"],
        "event_ts": [now],
        "publication_ts": [now],
        "vendor_received_ts": [now],
        "effective_ts": [now],
        "market_session": ["regular"],
        "source_id": ["test_vendor"],
        "ingested_at": [now],
        "revision_id": ["1"],
        "data_quality_flags": [{"flags": ["ok"]}],
        "adjustment_status": ["split_dividend_adjusted"],
        "open": [1.0],
        "high": [2.0],
        "low": [0.5],
        "close": [1.5],
        "volume": [100],
        "vwap": [1.4],
    }
    dtypes = {
        column: (
            "datetime64[ns, UTC]" if column.endswith("_ts") or column == "ingested_at" else "object"
        )
        for column in data
    }
    for column in ["open", "high", "low", "close", "vwap"]:
        dtypes[column] = "float64"
    dtypes["volume"] = "int64"

    report = EQUITY_ETF_OHLCV_SCHEMA.validate_strict(_FakePandasFrame(data, dtypes))

    assert report.is_valid
    assert report.row_count == 1


def test_validation_reports_missing_columns_and_enum_errors() -> None:
    """Invalid frames should report all detected problems before raising."""
    frame = _FakePandasFrame(
        {"symbol": ["SPY"], "market_session": ["invalid"]},
        {"symbol": "object", "market_session": "object"},
    )
    report = EQUITY_ETF_OHLCV_SCHEMA.validate_frame(frame)

    assert "event_ts" in report.missing_columns
    assert report.enum_violations
    with pytest.raises(SchemaValidationError, match="event_ts"):
        report.raise_for_errors()
