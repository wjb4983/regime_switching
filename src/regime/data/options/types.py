"""Typed option contract and quote records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

OptionType = Literal["call", "put"]


@dataclass(frozen=True, slots=True)
class CorporateActionAdjustment:
    """Metadata describing how an option contract was adjusted for corporate actions."""

    adjusted: bool = False
    adjustment_factor: float = 1.0
    deliverable: str = "100 shares"
    effective_date: date | None = None
    source: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.adjustment_factor <= 0:
            raise ValueError("adjustment_factor must be positive")


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """Normalized option quote with contract multiplier and adjustment metadata."""

    underlying: str
    symbol: str
    expiration: date
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    quote_time: datetime
    underlying_price: float
    multiplier: float = 100.0
    bid_size: float | None = None
    ask_size: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    adjustment: CorporateActionAdjustment = field(default_factory=CorporateActionAdjustment)

    @property
    def mid(self) -> float:
        """Mid-market option premium."""
        return (self.bid + self.ask) / 2.0

    @property
    def notional_mid(self) -> float:
        """Premium multiplied by the contract multiplier."""
        return self.mid * self.multiplier
