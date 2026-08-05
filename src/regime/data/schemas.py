"""Strict tabular market-data schemas and dataframe validation helpers.

The module intentionally represents schemas as typed metadata rather than binding to one
frame engine.  The same :class:`TabularSchema` can validate pandas, Polars, Arrow tables,
and DuckDB relations, while also exposing Arrow and DuckDB type declarations for storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, Self, TypeAlias, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DataQualityFlag: TypeAlias = Literal[
    "ok",
    "missing_value",
    "stale",
    "outlier",
    "corrected",
    "estimated",
    "partial",
    "late",
    "vendor_warning",
]

MARKET_SESSION_VALUES = (
    "pre_market",
    "regular",
    "post_market",
    "overnight",
    "closed",
    "auction",
    "continuous",
)
MarketSession: TypeAlias = Literal[
    "pre_market",
    "regular",
    "post_market",
    "overnight",
    "closed",
    "auction",
    "continuous",
]

ADJUSTMENT_STATUS_VALUES = (
    "raw",
    "split_adjusted",
    "dividend_adjusted",
    "split_dividend_adjusted",
    "back_adjusted",
    "not_applicable",
)
AdjustmentStatus: TypeAlias = Literal[
    "raw",
    "split_adjusted",
    "dividend_adjusted",
    "split_dividend_adjusted",
    "back_adjusted",
    "not_applicable",
]

FieldKind: TypeAlias = Literal[
    "string",
    "integer",
    "float",
    "boolean",
    "timestamp",
    "date",
    "json",
    "list_float",
]

DataFrameBackend: TypeAlias = Literal["pandas", "polars", "arrow", "duckdb"]


class SchemaValidationError(ValueError):
    """Raised when a tabular object does not satisfy a declared schema."""


class SchemaName(StrEnum):
    """Canonical names for built-in datasets."""

    EQUITY_ETF_OHLCV = "equity_etf_ohlcv"
    CORPORATE_ACTIONS = "corporate_actions"
    INDEX_DATA = "index_data"
    FUNDAMENTALS_FACTORS = "fundamentals_factors"
    OPTION_QUOTES = "option_quotes"
    OPTION_TRADES = "option_trades"
    OPTION_GREEKS = "option_greeks"
    IMPLIED_VOL_SURFACES = "implied_volatility_surfaces"
    VOLATILITY_INDICES = "volatility_indices"
    FUTURES = "futures"
    RATES_YIELD_CURVES = "rates_yield_curves"
    CREDIT_SPREADS = "credit_spreads"
    FX = "fx"
    BREADTH = "breadth"
    SHORT_INTEREST_BORROW = "short_interest_borrow"
    LIQUIDITY_MICROSTRUCTURE = "liquidity_microstructure"
    MACRO_RELEASES = "macro_releases"
    NEWS_TEXT_EMBEDDINGS = "news_text_embeddings"


class SchemaField(BaseModel):
    """A strict column definition that can be translated across dataframe engines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: FieldKind
    required: bool = True
    nullable: bool = False
    description: str = Field(default="", min_length=0)
    allowed_values: tuple[str, ...] | None = None

    @field_validator("allowed_values", mode="before")
    @classmethod
    def _coerce_allowed_values(cls, value: object) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return (value,)
        return tuple(cast(Iterable[str], value))


class ValidationReport(BaseModel):
    """Summary returned by schema validation helpers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: SchemaName
    backend: DataFrameBackend
    row_count: int | None
    missing_columns: tuple[str, ...] = ()
    extra_columns: tuple[str, ...] = ()
    type_mismatches: tuple[str, ...] = ()
    enum_violations: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether no validation problems were found."""
        return not (
            self.missing_columns
            or self.extra_columns
            or self.type_mismatches
            or self.enum_violations
        )

    def raise_for_errors(self) -> Self:
        """Raise :class:`SchemaValidationError` when the report contains problems."""
        if not self.is_valid:
            raise SchemaValidationError(self.model_dump_json())
        return self


@runtime_checkable
class _DuckDBRelation(Protocol):
    columns: list[str]

    types: Any

    def count(self, column: str) -> Any: ...


class TabularSchema(BaseModel):
    """Engine-neutral schema with pandas, Polars, Arrow, and DuckDB helpers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SchemaName
    fields: tuple[SchemaField, ...]
    primary_key: tuple[str, ...]
    description: str

    @field_validator("fields", mode="before")
    @classmethod
    def _coerce_fields(cls, value: object) -> tuple[SchemaField, ...]:
        return cast(tuple[SchemaField, ...], tuple(cast(Iterable[object], value)))

    @field_validator("primary_key", mode="before")
    @classmethod
    def _coerce_primary_key(cls, value: object) -> tuple[str, ...]:
        return tuple(cast(Iterable[str], value))

    @model_validator(mode="after")
    def _validate_unique_fields_and_keys(self) -> Self:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"schema {self.name} contains duplicate fields")
        missing_key_fields = tuple(name for name in self.primary_key if name not in names)
        if missing_key_fields:
            raise ValueError(f"primary key references missing fields: {missing_key_fields}")
        return self

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Names of columns that must be present."""
        return tuple(field.name for field in self.fields if field.required)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Names of all declared columns in stable order."""
        return tuple(field.name for field in self.fields)

    def field_map(self) -> dict[str, SchemaField]:
        """Return field definitions keyed by column name."""
        return {field.name: field for field in self.fields}

    def validate_frame(self, data: object, *, strict: bool = True) -> ValidationReport:
        """Validate a pandas/Polars/Arrow/DuckDB tabular object against this schema."""
        backend, columns, type_by_column, row_count = _inspect_tabular(data)
        missing = tuple(name for name in self.required_columns if name not in columns)
        declared = set(self.column_names)
        extra = tuple(name for name in columns if strict and name not in declared)
        type_mismatches = tuple(
            f"{field.name}: expected {field.kind}, got {type_by_column[field.name]}"
            for field in self.fields
            if field.name in type_by_column
            and not _type_matches(field.kind, type_by_column[field.name])
        )
        enum_violations = _enum_violations(data, backend, self.fields)
        return ValidationReport(
            schema_name=self.name,
            backend=backend,
            row_count=row_count,
            missing_columns=missing,
            extra_columns=extra,
            type_mismatches=type_mismatches,
            enum_violations=enum_violations,
        )

    def validate_strict(self, data: object) -> ValidationReport:
        """Validate and raise if required columns, types, enums, or extras are invalid."""
        return self.validate_frame(data, strict=True).raise_for_errors()

    def to_arrow_schema(self) -> Any:
        """Return a ``pyarrow.Schema`` with metadata preserving the canonical schema name."""
        import pyarrow as pa

        return pa.schema(
            [
                pa.field(field.name, _arrow_type(field.kind), nullable=field.nullable)
                for field in self.fields
            ],
            metadata={b"regime_schema": self.name.value.encode()},
        )

    def duckdb_columns_sql(self) -> str:
        """Return a DuckDB ``CREATE TABLE`` column fragment for this schema."""
        chunks = []
        for field in self.fields:
            nullability = "" if field.nullable else " NOT NULL"
            chunks.append(f'"{field.name}" {_duckdb_type(field.kind)}{nullability}')
        return ",\n".join(chunks)


def _common_fields(*, instrument: bool = True, adjustment: bool = False) -> list[SchemaField]:
    fields = [
        SchemaField(
            name="event_ts",
            kind="timestamp",
            description="Timestamp when the market/economic event occurred.",
        ),
        SchemaField(
            name="publication_ts",
            kind="timestamp",
            nullable=True,
            description="Timestamp when the value became public.",
        ),
        SchemaField(
            name="vendor_received_ts",
            kind="timestamp",
            description="Timestamp when the vendor observed the value.",
        ),
        SchemaField(
            name="effective_ts",
            kind="timestamp",
            nullable=True,
            description="Timestamp when the value becomes effective for downstream use.",
        ),
        SchemaField(
            name="market_session",
            kind="string",
            allowed_values=MARKET_SESSION_VALUES,
            nullable=True,
        ),
        SchemaField(name="source_id", kind="string"),
        SchemaField(name="ingested_at", kind="timestamp", description="UTC ingestion timestamp."),
        SchemaField(name="revision_id", kind="string", nullable=True),
        SchemaField(
            name="data_quality_flags",
            kind="json",
            description="Array/object of quality flags from validation and vendors.",
        ),
    ]
    if instrument:
        fields[0:0] = [
            SchemaField(name="symbol", kind="string"),
            SchemaField(name="security_id", kind="string", nullable=True),
        ]
    if adjustment:
        fields.append(
            SchemaField(
                name="adjustment_status", kind="string", allowed_values=ADJUSTMENT_STATUS_VALUES
            )
        )
    return fields


def _schema(
    name: SchemaName, fields: Sequence[SchemaField], primary_key: Sequence[str], description: str
) -> TabularSchema:
    return TabularSchema(
        name=name, fields=tuple(fields), primary_key=tuple(primary_key), description=description
    )


def _price_fields() -> list[SchemaField]:
    return [SchemaField(name=name, kind="float") for name in ("open", "high", "low", "close")]


EQUITY_ETF_OHLCV_SCHEMA = _schema(
    SchemaName.EQUITY_ETF_OHLCV,
    [
        *_common_fields(adjustment=True),
        *_price_fields(),
        SchemaField(name="volume", kind="integer"),
        SchemaField(name="vwap", kind="float", nullable=True),
    ],
    ["symbol", "event_ts", "source_id", "revision_id"],
    "Equity and ETF open/high/low/close/volume bars.",
)
CORPORATE_ACTIONS_SCHEMA = _schema(
    SchemaName.CORPORATE_ACTIONS,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="action_type", kind="string"),
        SchemaField(name="cash_amount", kind="float", nullable=True),
        SchemaField(name="split_ratio", kind="float", nullable=True),
    ],
    ["symbol", "effective_ts", "action_type", "source_id", "revision_id"],
    "Splits, dividends, mergers, and other issuer actions.",
)
INDEX_DATA_SCHEMA = _schema(
    SchemaName.INDEX_DATA,
    [
        *_common_fields(adjustment=True),
        SchemaField(name="index_level", kind="float"),
        SchemaField(name="return_value", kind="float", nullable=True),
        SchemaField(name="constituent_count", kind="integer", nullable=True),
    ],
    ["symbol", "event_ts", "source_id", "revision_id"],
    "Index levels, returns, and composition counts.",
)
FUNDAMENTALS_FACTORS_SCHEMA = _schema(
    SchemaName.FUNDAMENTALS_FACTORS,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="period_end_date", kind="date"),
        SchemaField(name="factor_name", kind="string"),
        SchemaField(name="factor_value", kind="float"),
        SchemaField(name="reporting_currency", kind="string", nullable=True),
    ],
    ["symbol", "period_end_date", "factor_name", "source_id", "revision_id"],
    "Point-in-time fundamentals and factor values.",
)
OPTION_QUOTES_SCHEMA = _schema(
    SchemaName.OPTION_QUOTES,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="underlying_symbol", kind="string"),
        SchemaField(name="expiration_date", kind="date"),
        SchemaField(name="strike", kind="float"),
        SchemaField(name="option_type", kind="string", allowed_values=("call", "put")),
        SchemaField(name="bid", kind="float"),
        SchemaField(name="ask", kind="float"),
        SchemaField(name="bid_size", kind="integer", nullable=True),
        SchemaField(name="ask_size", kind="integer", nullable=True),
    ],
    ["symbol", "event_ts", "source_id", "revision_id"],
    "Option NBBO or vendor quote snapshots.",
)
OPTION_TRADES_SCHEMA = _schema(
    SchemaName.OPTION_TRADES,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="underlying_symbol", kind="string"),
        SchemaField(name="expiration_date", kind="date"),
        SchemaField(name="strike", kind="float"),
        SchemaField(name="option_type", kind="string", allowed_values=("call", "put")),
        SchemaField(name="trade_price", kind="float"),
        SchemaField(name="trade_size", kind="integer"),
        SchemaField(name="exchange", kind="string", nullable=True),
    ],
    ["symbol", "event_ts", "source_id", "revision_id"],
    "Option executions and prints.",
)
OPTION_GREEKS_SCHEMA = _schema(
    SchemaName.OPTION_GREEKS,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="underlying_symbol", kind="string"),
        SchemaField(name="expiration_date", kind="date"),
        SchemaField(name="strike", kind="float"),
        SchemaField(name="option_type", kind="string", allowed_values=("call", "put")),
        *[
            SchemaField(name=n, kind="float", nullable=True)
            for n in ("delta", "gamma", "theta", "vega", "rho", "implied_volatility")
        ],
    ],
    ["symbol", "event_ts", "source_id", "revision_id"],
    "Point-in-time option Greeks and implied volatility.",
)
IMPLIED_VOLATILITY_SURFACES_SCHEMA = _schema(
    SchemaName.IMPLIED_VOL_SURFACES,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="underlying_symbol", kind="string"),
        SchemaField(name="tenor_days", kind="integer"),
        SchemaField(name="moneyness", kind="float"),
        SchemaField(name="delta", kind="float", nullable=True),
        SchemaField(name="implied_volatility", kind="float"),
        SchemaField(name="surface_model", kind="string", nullable=True),
    ],
    ["underlying_symbol", "event_ts", "tenor_days", "moneyness", "source_id", "revision_id"],
    "Implied-volatility surface nodes.",
)
VOLATILITY_INDICES_SCHEMA = _schema(
    SchemaName.VOLATILITY_INDICES,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="index_level", kind="float"),
        SchemaField(name="term", kind="string", nullable=True),
    ],
    ["symbol", "event_ts", "source_id", "revision_id"],
    "Volatility index levels such as VIX-family series.",
)
FUTURES_SCHEMA = _schema(
    SchemaName.FUTURES,
    [
        *_common_fields(adjustment=True),
        SchemaField(name="contract_code", kind="string"),
        SchemaField(name="expiration_date", kind="date"),
        *_price_fields(),
        SchemaField(name="volume", kind="integer", nullable=True),
        SchemaField(name="open_interest", kind="integer", nullable=True),
    ],
    ["symbol", "contract_code", "event_ts", "source_id", "revision_id"],
    "Futures contract OHLCV and open interest.",
)
RATES_YIELD_CURVES_SCHEMA = _schema(
    SchemaName.RATES_YIELD_CURVES,
    [
        *_common_fields(instrument=False),
        SchemaField(name="curve_id", kind="string"),
        SchemaField(name="tenor", kind="string"),
        SchemaField(name="tenor_days", kind="integer"),
        SchemaField(name="rate", kind="float"),
        SchemaField(name="currency", kind="string"),
    ],
    ["curve_id", "tenor", "event_ts", "source_id", "revision_id"],
    "Rates and yield-curve nodes.",
)
CREDIT_SPREADS_SCHEMA = _schema(
    SchemaName.CREDIT_SPREADS,
    [
        *_common_fields(instrument=False),
        SchemaField(name="spread_id", kind="string"),
        SchemaField(name="sector", kind="string", nullable=True),
        SchemaField(name="rating", kind="string", nullable=True),
        SchemaField(name="tenor", kind="string", nullable=True),
        SchemaField(name="spread_bps", kind="float"),
    ],
    ["spread_id", "event_ts", "source_id", "revision_id"],
    "Credit spread series and curves.",
)
FX_SCHEMA = _schema(
    SchemaName.FX,
    [
        *_common_fields(instrument=False),
        SchemaField(name="base_currency", kind="string"),
        SchemaField(name="quote_currency", kind="string"),
        SchemaField(name="bid", kind="float", nullable=True),
        SchemaField(name="ask", kind="float", nullable=True),
        SchemaField(name="mid", kind="float"),
    ],
    ["base_currency", "quote_currency", "event_ts", "source_id", "revision_id"],
    "Foreign-exchange spot, forwards, or fixings.",
)
BREADTH_SCHEMA = _schema(
    SchemaName.BREADTH,
    [
        *_common_fields(instrument=False),
        SchemaField(name="universe_id", kind="string"),
        SchemaField(name="advancers", kind="integer", nullable=True),
        SchemaField(name="decliners", kind="integer", nullable=True),
        SchemaField(name="new_highs", kind="integer", nullable=True),
        SchemaField(name="new_lows", kind="integer", nullable=True),
        SchemaField(name="percent_above_ma", kind="float", nullable=True),
    ],
    ["universe_id", "event_ts", "source_id", "revision_id"],
    "Market breadth measures by universe.",
)
SHORT_INTEREST_BORROW_SCHEMA = _schema(
    SchemaName.SHORT_INTEREST_BORROW,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="settlement_date", kind="date", nullable=True),
        SchemaField(name="short_interest", kind="integer", nullable=True),
        SchemaField(name="days_to_cover", kind="float", nullable=True),
        SchemaField(name="borrow_rate", kind="float", nullable=True),
        SchemaField(name="available_to_borrow", kind="integer", nullable=True),
    ],
    ["symbol", "event_ts", "source_id", "revision_id"],
    "Short-interest and securities-lending availability/rate data.",
)
LIQUIDITY_MICROSTRUCTURE_SCHEMA = _schema(
    SchemaName.LIQUIDITY_MICROSTRUCTURE,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="bid", kind="float", nullable=True),
        SchemaField(name="ask", kind="float", nullable=True),
        SchemaField(name="spread_bps", kind="float", nullable=True),
        SchemaField(name="depth", kind="float", nullable=True),
        SchemaField(name="turnover", kind="float", nullable=True),
        SchemaField(name="trade_count", kind="integer", nullable=True),
    ],
    ["symbol", "event_ts", "source_id", "revision_id"],
    "Liquidity, quote depth, turnover, and microstructure metrics.",
)
MACRO_RELEASES_SCHEMA = _schema(
    SchemaName.MACRO_RELEASES,
    [
        *_common_fields(instrument=False),
        SchemaField(name="release_id", kind="string"),
        SchemaField(name="region", kind="string"),
        SchemaField(name="indicator", kind="string"),
        SchemaField(name="period", kind="string"),
        SchemaField(name="actual_value", kind="float"),
        SchemaField(name="survey_median", kind="float", nullable=True),
        SchemaField(name="prior_value", kind="float", nullable=True),
        SchemaField(name="unit", kind="string", nullable=True),
    ],
    ["release_id", "period", "publication_ts", "source_id", "revision_id"],
    "Point-in-time macroeconomic releases, estimates, and revisions.",
)
NEWS_TEXT_EMBEDDINGS_SCHEMA = _schema(
    SchemaName.NEWS_TEXT_EMBEDDINGS,
    [
        *_common_fields(adjustment=False),
        SchemaField(name="document_id", kind="string"),
        SchemaField(name="headline", kind="string", nullable=True),
        SchemaField(name="language", kind="string", nullable=True),
        SchemaField(name="embedding_model", kind="string"),
        SchemaField(name="embedding", kind="list_float"),
        SchemaField(name="sentiment_score", kind="float", nullable=True),
    ],
    ["document_id", "symbol", "publication_ts", "source_id", "revision_id"],
    "News metadata and text embeddings for point-in-time research.",
)

SCHEMAS: dict[SchemaName, TabularSchema] = {
    schema.name: schema
    for schema in (
        EQUITY_ETF_OHLCV_SCHEMA,
        CORPORATE_ACTIONS_SCHEMA,
        INDEX_DATA_SCHEMA,
        FUNDAMENTALS_FACTORS_SCHEMA,
        OPTION_QUOTES_SCHEMA,
        OPTION_TRADES_SCHEMA,
        OPTION_GREEKS_SCHEMA,
        IMPLIED_VOLATILITY_SURFACES_SCHEMA,
        VOLATILITY_INDICES_SCHEMA,
        FUTURES_SCHEMA,
        RATES_YIELD_CURVES_SCHEMA,
        CREDIT_SPREADS_SCHEMA,
        FX_SCHEMA,
        BREADTH_SCHEMA,
        SHORT_INTEREST_BORROW_SCHEMA,
        LIQUIDITY_MICROSTRUCTURE_SCHEMA,
        MACRO_RELEASES_SCHEMA,
        NEWS_TEXT_EMBEDDINGS_SCHEMA,
    )
}


def get_schema(name: SchemaName | str) -> TabularSchema:
    """Return a built-in schema by enum or string name."""
    return SCHEMAS[SchemaName(name)]


def utc_ingestion_timestamp() -> datetime:
    """Return a timezone-aware UTC timestamp suitable for ``ingested_at``."""
    return datetime.now(tz=UTC)


def _inspect_tabular(
    data: object,
) -> tuple[DataFrameBackend, list[str], dict[str, str], int | None]:
    module = type(data).__module__
    if module.startswith("pandas"):
        frame = cast(Any, data)
        columns = [str(column) for column in frame.columns]
        return (
            "pandas",
            columns,
            {column: str(frame.dtypes[column]) for column in columns},
            len(frame),
        )
    if module.startswith("polars"):
        frame = cast(Any, data)
        return (
            "polars",
            list(frame.columns),
            {name: str(dtype) for name, dtype in frame.schema.items()},
            frame.height,
        )
    if module.startswith("pyarrow"):
        frame = cast(Any, data)
        return (
            "arrow",
            list(frame.schema.names),
            {field.name: str(field.type) for field in frame.schema},
            frame.num_rows,
        )
    if hasattr(data, "columns") and hasattr(data, "types"):
        relation = cast(_DuckDBRelation, data)
        return (
            "duckdb",
            list(relation.columns),
            dict(zip(relation.columns, map(str, _duckdb_relation_types(relation)), strict=True)),
            None,
        )
    raise TypeError(
        "expected a pandas DataFrame, Polars DataFrame, Arrow Table, or DuckDB relation"
    )


def _type_matches(kind: FieldKind, dtype: str) -> bool:
    normalized = dtype.lower()
    expected = {
        "string": ("string", "str", "object", "utf8", "varchar", "large_string"),
        "integer": ("int", "integer", "bigint", "int64", "uint"),
        "float": ("float", "double", "decimal", "real"),
        "boolean": ("bool", "boolean"),
        "timestamp": ("datetime", "timestamp"),
        "date": ("date",),
        "json": ("object", "struct", "list", "json", "map", "large_string", "string", "varchar"),
        "list_float": ("list", "array", "object"),
    }[kind]
    return any(token in normalized for token in expected)


def _duckdb_relation_types(relation: _DuckDBRelation) -> list[str]:
    types = relation.types
    raw_types = types() if callable(types) else types
    return [str(value) for value in raw_types]


def _enum_violations(
    data: object, backend: DataFrameBackend, fields: Sequence[SchemaField]
) -> tuple[str, ...]:
    violations: list[str] = []
    for field in fields:
        if not field.allowed_values:
            continue
        values = _unique_non_null_values(data, backend, field.name)
        if values is None:
            continue
        invalid = sorted(value for value in values if value not in field.allowed_values)
        if invalid:
            violations.append(
                f"{field.name}: invalid values {invalid}; allowed {list(field.allowed_values)}"
            )
    return tuple(violations)


def _unique_non_null_values(
    data: object, backend: DataFrameBackend, column: str
) -> set[str] | None:
    try:
        if backend == "pandas":
            frame = cast(Any, data)
            if column not in frame.columns:
                return None
            return {str(value) for value in frame[column].dropna().unique().tolist()}
        if backend == "polars":
            frame = cast(Any, data)
            if column not in frame.columns:
                return None
            return {
                str(value) for value in frame.get_column(column).drop_nulls().unique().to_list()
            }
        if backend == "arrow":
            import pyarrow.compute as pc

            frame = cast(Any, data)
            if column not in frame.schema.names:
                return None
            return {
                str(value.as_py()) for value in pc.unique(frame[column].drop_null()).to_pylist()
            }
    except (AttributeError, KeyError, TypeError):
        return None
    return None


def _arrow_type(kind: FieldKind) -> Any:
    import pyarrow as pa

    return {
        "string": pa.string(),
        "integer": pa.int64(),
        "float": pa.float64(),
        "boolean": pa.bool_(),
        "timestamp": pa.timestamp("us", tz="UTC"),
        "date": pa.date32(),
        "json": pa.large_string(),
        "list_float": pa.list_(pa.float64()),
    }[kind]


def _duckdb_type(kind: FieldKind) -> str:
    return {
        "string": "VARCHAR",
        "integer": "BIGINT",
        "float": "DOUBLE",
        "boolean": "BOOLEAN",
        "timestamp": "TIMESTAMPTZ",
        "date": "DATE",
        "json": "JSON",
        "list_float": "DOUBLE[]",
    }[kind]


__all__ = [
    "BREADTH_SCHEMA",
    "CORPORATE_ACTIONS_SCHEMA",
    "CREDIT_SPREADS_SCHEMA",
    "EQUITY_ETF_OHLCV_SCHEMA",
    "FUNDAMENTALS_FACTORS_SCHEMA",
    "FUTURES_SCHEMA",
    "FX_SCHEMA",
    "IMPLIED_VOLATILITY_SURFACES_SCHEMA",
    "INDEX_DATA_SCHEMA",
    "LIQUIDITY_MICROSTRUCTURE_SCHEMA",
    "MACRO_RELEASES_SCHEMA",
    "NEWS_TEXT_EMBEDDINGS_SCHEMA",
    "OPTION_GREEKS_SCHEMA",
    "OPTION_QUOTES_SCHEMA",
    "OPTION_TRADES_SCHEMA",
    "RATES_YIELD_CURVES_SCHEMA",
    "SCHEMAS",
    "SHORT_INTEREST_BORROW_SCHEMA",
    "VOLATILITY_INDICES_SCHEMA",
    "AdjustmentStatus",
    "DataQualityFlag",
    "MarketSession",
    "SchemaField",
    "SchemaName",
    "SchemaValidationError",
    "TabularSchema",
    "ValidationReport",
    "get_schema",
    "utc_ingestion_timestamp",
]
