"""Provider interfaces for vendor-neutral market and alternative data ingestion.

The classes in this module define small, typed contracts that concrete adapters can
implement for live vendors, local files, or deterministic tests.  The base provider
captures cross-cutting ingestion requirements (capabilities, date ranges, identifier
mapping, pagination, throttling, retries, caching, partial recovery, idempotency, and
secret hygiene), while domain-specific abstract classes name the canonical fetch
operations expected by the rest of the data layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar

Record: TypeAlias = Mapping[str, object]
TRecord = TypeVar("TRecord", bound=Record)


class DataDomain(StrEnum):
    """Canonical provider capability domains."""

    EQUITY_OHLCV = "equity_ohlcv"
    CORPORATE_ACTIONS = "corporate_actions"
    INDEX_DATA = "index_data"
    OPTIONS_CHAINS = "options_chains"
    OPTIONS_QUOTES = "options_quotes"
    OPTIONS_TRADES = "options_trades"
    GREEKS = "greeks"
    VOLATILITY_SURFACES = "volatility_surfaces"
    RATES = "rates"
    CREDIT = "credit"
    FX = "fx"
    MACRO_RELEASES = "macro_releases"
    LIQUIDITY = "liquidity"
    BREADTH = "breadth"
    TEXT_EMBEDDINGS = "text_embeddings"


class IdentifierType(StrEnum):
    """Supported security identifier namespaces."""

    SYMBOL = "symbol"
    TICKER = "ticker"
    FIGI = "figi"
    CUSIP = "cusip"
    ISIN = "isin"
    SEDOL = "sedol"
    PERMID = "permid"
    VENDOR = "vendor"


@dataclass(frozen=True)
class DateRange:
    """Inclusive date or timestamp range for provider requests."""

    start: date | datetime
    end: date | datetime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("date range end must be greater than or equal to start")


@dataclass(frozen=True)
class SecurityIdentifier:
    """A security identifier in a known namespace."""

    value: str
    identifier_type: IdentifierType = IdentifierType.SYMBOL
    exchange: str | None = None
    asset_class: str | None = None


@dataclass(frozen=True)
class ProviderCapability:
    """Describes one fetchable domain and its operational limits."""

    domain: DataDomain
    description: str = ""
    max_page_size: int | None = None
    min_granularity: str | None = None
    supports_intraday: bool = False
    supports_adjusted: bool = False
    supported_identifier_types: tuple[IdentifierType, ...] = (IdentifierType.SYMBOL,)


@dataclass(frozen=True)
class PageCursor:
    """Opaque pagination cursor returned by provider adapters."""

    token: str | None = None
    page_number: int = 1


@dataclass(frozen=True)
class Page(Generic[TRecord]):
    """One page of provider records."""

    records: tuple[TRecord, ...]
    next_cursor: PageCursor | None = None
    provider_request_id: str | None = None
    is_partial: bool = False


@dataclass(frozen=True)
class RateLimitPolicy:
    """Simple client-side throttling policy."""

    requests_per_second: float | None = None
    min_interval: timedelta = timedelta(0)

    def delay_seconds(self, *, elapsed_since_last_request: float) -> float:
        interval = self.min_interval.total_seconds()
        if self.requests_per_second and self.requests_per_second > 0:
            interval = max(interval, 1.0 / self.requests_per_second)
        return max(0.0, interval - elapsed_since_last_request)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry policy for transient vendor failures."""

    max_attempts: int = 3
    initial_backoff: timedelta = timedelta(seconds=0.25)
    multiplier: float = 2.0
    max_backoff: timedelta = timedelta(seconds=10)
    retryable_status_codes: tuple[int, ...] = (408, 429, 500, 502, 503, 504)

    def backoff(self, attempt: int) -> float:
        if attempt < 1:
            raise ValueError("attempt must be one-indexed")
        seconds = self.initial_backoff.total_seconds() * (self.multiplier ** (attempt - 1))
        return min(seconds, self.max_backoff.total_seconds())


@dataclass(frozen=True)
class CachePolicy:
    """Request-cache settings used by adapters or orchestration code."""

    enabled: bool = True
    namespace: str = "provider"
    ttl: timedelta | None = None


@dataclass(frozen=True)
class ProviderRequest:
    """Common request envelope for all provider domains."""

    domain: DataDomain
    identifiers: tuple[SecurityIdentifier, ...]
    date_range: DateRange
    fields: tuple[str, ...] = ()
    cursor: PageCursor | None = None
    page_size: int | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)

    def cache_key(self) -> str:
        """Return a stable hash suitable for request caching and recovery manifests."""
        payload = {
            "cursor": None if self.cursor is None else self.cursor.__dict__,
            "date_range": {
                "start": self.date_range.start.isoformat(),
                "end": self.date_range.end.isoformat(),
            },
            "domain": self.domain.value,
            "fields": self.fields,
            "identifiers": [identifier.__dict__ for identifier in self.identifiers],
            "page_size": self.page_size,
            "parameters": dict(sorted(self.parameters.items())),
        }
        raw = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IngestionCheckpoint:
    """Checkpoint for resuming partial downloads idempotently."""

    request_cache_key: str
    completed_pages: tuple[str, ...] = ()
    next_cursor: PageCursor | None = None
    content_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderCredentials:
    """Credential bag with environment/local-file loading and redaction helpers."""

    values: Mapping[str, str] = field(default_factory=dict)
    secret_keys: tuple[str, ...] = ()

    @classmethod
    def load(
        cls,
        *,
        env_prefix: str,
        secret_file: Path | str | None = None,
        required: Iterable[str] = (),
    ) -> ProviderCredentials:
        """Load credentials from ``ENV_PREFIX_NAME`` variables and an ignored JSON file."""
        loaded: dict[str, str] = {}
        required_names = tuple(required)
        for name in required_names:
            value = os.getenv(f"{env_prefix}_{name}")
            if value:
                loaded[name] = value
        if secret_file is not None:
            path = Path(secret_file).expanduser()
            if path.exists():
                file_values = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(file_values, dict):
                    raise ValueError("secret file must contain a JSON object")
                loaded.update({str(key): str(value) for key, value in file_values.items()})
        missing = tuple(name for name in required_names if name not in loaded)
        if missing:
            raise ValueError(f"missing required provider credentials: {missing}")
        return cls(values=loaded, secret_keys=tuple(loaded))

    def redacted(self) -> Mapping[str, str]:
        """Return credentials safe for diagnostic logs."""
        return {key: SecretRedactor.redact(value) for key, value in self.values.items()}


class SecretRedactor:
    """Utility for masking credentials before they reach logs."""

    @staticmethod
    def redact(value: str) -> str:
        if len(value) <= 4:
            return "****"
        return f"{value[:2]}…{value[-2:]}"

    @classmethod
    def redact_mapping(
        cls, values: Mapping[str, object], secrets: Iterable[str]
    ) -> dict[str, object]:
        secret_set = set(secrets)
        return {
            key: cls.redact(str(value)) if key in secret_set else value
            for key, value in values.items()
        }


class BaseDataProvider(ABC):
    """Base contract shared by every concrete data adapter."""

    def __init__(
        self,
        *,
        credentials: ProviderCredentials | None = None,
        rate_limit: RateLimitPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
        cache_policy: CachePolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.credentials = credentials or ProviderCredentials()
        self.rate_limit = rate_limit or RateLimitPolicy()
        self.retry_policy = retry_policy or RetryPolicy()
        self.cache_policy = cache_policy or CachePolicy()
        self._sleep = sleep
        self._last_request_at = 0.0
        self._request_cache: dict[str, Page[Record]] = {}

    @abstractmethod
    def capabilities(self) -> tuple[ProviderCapability, ...]:
        """Return capabilities supported by this adapter."""

    @abstractmethod
    def map_identifier(
        self, identifier: SecurityIdentifier, target_type: IdentifierType
    ) -> SecurityIdentifier:
        """Map a symbol or security id into another namespace."""

    @abstractmethod
    def fetch(self, request: ProviderRequest) -> Page[Record]:
        """Fetch one page for a fully specified request."""

    def fetch_page(self, request: ProviderRequest) -> Page[Record]:
        """Fetch one page with cache lookup, throttling, and retry handling."""
        cache_key = request.cache_key()
        if self.cache_policy.enabled and cache_key in self._request_cache:
            return self._request_cache[cache_key]

        def fetch_current() -> Page[Record]:
            return self.fetch(request)

        page = self.request_with_retries(fetch_current)
        if self.cache_policy.enabled:
            self._request_cache[cache_key] = page
        return page

    def clear_request_cache(self) -> None:
        """Clear the local request cache."""
        self._request_cache.clear()

    def fetch_all_pages(self, request: ProviderRequest) -> tuple[Record, ...]:
        """Fetch all pages for a request, following cursors until completion."""
        records: list[Record] = []
        current = request
        while True:
            page = self.fetch_page(current)
            records.extend(page.records)
            if page.next_cursor is None:
                return tuple(records)
            current = replace(current, cursor=page.next_cursor)

    def wait_for_rate_limit(self) -> None:
        """Apply client-side rate limiting before issuing an upstream request."""
        now = time.monotonic()
        delay = self.rate_limit.delay_seconds(
            elapsed_since_last_request=now - self._last_request_at
        )
        if delay > 0:
            self._sleep(delay)
        self._last_request_at = time.monotonic()

    def request_with_retries(self, operation: Callable[[], Page[TRecord]]) -> Page[TRecord]:
        """Run an operation with exponential backoff for retryable exceptions."""
        last_error: Exception | None = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                self.wait_for_rate_limit()
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt == self.retry_policy.max_attempts:
                    break
                self._sleep(self.retry_policy.backoff(attempt))
        if last_error is None:
            raise RuntimeError("provider operation failed without an exception")
        raise last_error

    def recover_partial(
        self, request: ProviderRequest, checkpoint: IngestionCheckpoint
    ) -> ProviderRequest:
        """Return a request positioned at the checkpoint's next cursor."""
        if checkpoint.request_cache_key != request.cache_key():
            raise ValueError("checkpoint does not match request")
        return replace(request, cursor=checkpoint.next_cursor)

    def ingestion_id(self, request: ProviderRequest, page: Page[Record]) -> str:
        """Stable id used by writers to make ingestion idempotent."""
        payload = {"request": request.cache_key(), "records": list(page.records)}
        raw = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def safe_log_context(self) -> Mapping[str, object]:
        """Return provider settings with secrets redacted for logs."""
        return {"credentials": self.credentials.redacted(), "cache": self.cache_policy}


class EquityOHLCVProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_equity_ohlcv(self, request: ProviderRequest) -> Page[Record]: ...


class CorporateActionsProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_corporate_actions(self, request: ProviderRequest) -> Page[Record]: ...


class IndexDataProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_index_data(self, request: ProviderRequest) -> Page[Record]: ...


class OptionsChainsProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_options_chains(self, request: ProviderRequest) -> Page[Record]: ...


class OptionsQuotesTradesProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_options_quotes(self, request: ProviderRequest) -> Page[Record]: ...

    @abstractmethod
    def get_options_trades(self, request: ProviderRequest) -> Page[Record]: ...


class GreeksProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_greeks(self, request: ProviderRequest) -> Page[Record]: ...


class VolatilitySurfaceProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_volatility_surfaces(self, request: ProviderRequest) -> Page[Record]: ...


class RatesProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_rates(self, request: ProviderRequest) -> Page[Record]: ...


class CreditProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_credit(self, request: ProviderRequest) -> Page[Record]: ...


class FXProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_fx(self, request: ProviderRequest) -> Page[Record]: ...


class MacroReleasesProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_macro_releases(self, request: ProviderRequest) -> Page[Record]: ...


class LiquidityProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_liquidity(self, request: ProviderRequest) -> Page[Record]: ...


class BreadthProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_breadth(self, request: ProviderRequest) -> Page[Record]: ...


class TextEmbeddingsProvider(BaseDataProvider, ABC):
    @abstractmethod
    def get_text_embeddings(self, request: ProviderRequest) -> Page[Record]: ...


class MockProvider(
    EquityOHLCVProvider,
    CorporateActionsProvider,
    IndexDataProvider,
    OptionsChainsProvider,
    OptionsQuotesTradesProvider,
    GreeksProvider,
    VolatilitySurfaceProvider,
    RatesProvider,
    CreditProvider,
    FXProvider,
    MacroReleasesProvider,
    LiquidityProvider,
    BreadthProvider,
    TextEmbeddingsProvider,
):
    """Deterministic in-memory provider for tests and examples."""

    def __init__(
        self, records_by_domain: Mapping[DataDomain, Sequence[Record]] | None = None
    ) -> None:
        super().__init__(sleep=lambda _seconds: None)
        self._records_by_domain = {
            domain: tuple(records)
            for domain, records in (records_by_domain or _default_mock_records()).items()
        }

    def capabilities(self) -> tuple[ProviderCapability, ...]:
        return tuple(
            ProviderCapability(
                domain=domain,
                description=f"Mock {domain.value} records",
                max_page_size=10_000,
                supports_intraday=domain
                in {DataDomain.EQUITY_OHLCV, DataDomain.OPTIONS_QUOTES, DataDomain.OPTIONS_TRADES},
                supports_adjusted=domain is DataDomain.EQUITY_OHLCV,
                supported_identifier_types=(IdentifierType.SYMBOL, IdentifierType.FIGI),
            )
            for domain in DataDomain
        )

    def map_identifier(
        self, identifier: SecurityIdentifier, target_type: IdentifierType
    ) -> SecurityIdentifier:
        if identifier.identifier_type is target_type:
            return identifier
        return SecurityIdentifier(
            value=f"{target_type.value}:{identifier.value}",
            identifier_type=target_type,
            exchange=identifier.exchange,
            asset_class=identifier.asset_class,
        )

    def fetch(self, request: ProviderRequest) -> Page[Record]:
        records = self._records_by_domain.get(request.domain, ())
        page_size = request.page_size or len(records) or 1
        start = request.cursor.page_number - 1 if request.cursor else 0
        offset = start * page_size
        page_records = tuple(records[offset : offset + page_size])
        next_offset = offset + page_size
        next_cursor = None
        if next_offset < len(records):
            next_cursor = PageCursor(page_number=start + 2)
        return Page(
            records=page_records, next_cursor=next_cursor, provider_request_id=request.cache_key()
        )

    def get_equity_ohlcv(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.EQUITY_OHLCV))

    def get_corporate_actions(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.CORPORATE_ACTIONS))

    def get_index_data(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.INDEX_DATA))

    def get_options_chains(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.OPTIONS_CHAINS))

    def get_options_quotes(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.OPTIONS_QUOTES))

    def get_options_trades(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.OPTIONS_TRADES))

    def get_greeks(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.GREEKS))

    def get_volatility_surfaces(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.VOLATILITY_SURFACES))

    def get_rates(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.RATES))

    def get_credit(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.CREDIT))

    def get_fx(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.FX))

    def get_macro_releases(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.MACRO_RELEASES))

    def get_liquidity(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.LIQUIDITY))

    def get_breadth(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.BREADTH))

    def get_text_embeddings(self, request: ProviderRequest) -> Page[Record]:
        return self.fetch(replace(request, domain=DataDomain.TEXT_EMBEDDINGS))


def _default_mock_records() -> Mapping[DataDomain, tuple[Record, ...]]:
    as_of = date(2024, 1, 2).isoformat()
    return {
        domain: (
            {
                "domain": domain.value,
                "symbol": "MOCK",
                "as_of": as_of,
                "value": float(index + 1),
            },
        )
        for index, domain in enumerate(DataDomain)
    }
