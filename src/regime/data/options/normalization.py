"""Normalization helpers for heterogeneous option vendor records."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time
from typing import Any, SupportsFloat, cast

from regime.data.options.types import CorporateActionAdjustment, OptionQuote, OptionType

_OSI_RE = re.compile(r"^([A-Z0-9.]{1,6})(\d{6})([CP])(\d{8})$")
_TYPE_ALIASES = {"C": "call", "CALL": "call", "1": "call", "P": "put", "PUT": "put", "0": "put"}


def normalize_option_symbol(value: str) -> str:
    """Normalize vendor symbols to compact uppercase OSI-like text when possible."""
    symbol = re.sub(r"[\s_:-]+", "", value.strip().upper())
    match = _OSI_RE.match(symbol)
    if match:
        root, expiry, cp, strike = match.groups()
        return f"{root}{expiry}{cp}{strike}"
    return symbol


def normalize_expiration(value: str | date | datetime) -> date:
    """Normalize expiration values to a calendar date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = value.strip()
    formats = (
        ("%y%m%d",) if len(text) == 6 and text.isdigit() else ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y")
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unsupported expiration format: {value!r}")


def normalize_strike(value: str | int | float, *, osi_encoded: bool = False) -> float:
    """Normalize strike prices, including OSI 1/1000 encoded strikes."""
    number = float(str(value).strip().replace(",", ""))
    if osi_encoded:
        number /= 1000.0
    if number <= 0:
        raise ValueError("strike must be positive")
    return number


def normalize_option_type(value: str) -> OptionType:
    """Normalize call/put flags to ``call`` or ``put``."""
    try:
        return _TYPE_ALIASES[value.strip().upper()]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"unsupported option type: {value!r}") from exc


def normalize_option_quote(record: dict[str, Any]) -> OptionQuote:
    """Build an :class:`OptionQuote` from common vendor field names."""
    symbol = normalize_option_symbol(str(record.get("symbol") or record.get("option_symbol")))
    expiration = normalize_expiration(
        cast(str | date | datetime, record.get("expiration") or record.get("expiry"))
    )
    quote_time = record.get("quote_time") or record.get("timestamp")
    if not isinstance(quote_time, datetime):
        quote_time = datetime.fromisoformat(str(quote_time).replace("Z", "+00:00"))
    if quote_time.tzinfo is None:
        quote_time = datetime.combine(quote_time.date(), time(), UTC).replace(
            hour=quote_time.hour, minute=quote_time.minute, second=quote_time.second
        )
    return OptionQuote(
        underlying=str(record.get("underlying") or record.get("root") or "").strip().upper(),
        symbol=symbol,
        expiration=expiration,
        strike=normalize_strike(cast(str | int | float, record.get("strike"))),
        option_type=normalize_option_type(str(record.get("option_type") or record.get("right"))),
        bid=float(record.get("bid", 0.0)),
        ask=float(record.get("ask", 0.0)),
        quote_time=quote_time,
        underlying_price=float(
            cast(SupportsFloat, record.get("underlying_price") or record.get("spot"))
        ),
        multiplier=float(record.get("multiplier", 100.0)),
        bid_size=_optional_float(record.get("bid_size")),
        ask_size=_optional_float(record.get("ask_size")),
        volume=_optional_float(record.get("volume")),
        open_interest=_optional_float(record.get("open_interest")),
        adjustment=CorporateActionAdjustment(
            adjusted=bool(record.get("adjusted", False)),
            adjustment_factor=float(record.get("adjustment_factor", 1.0)),
            deliverable=str(record.get("deliverable", "100 shares")),
            source=record.get("adjustment_source"),
            note=record.get("adjustment_note"),
        ),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None or value == "" else float(cast(SupportsFloat, value))
