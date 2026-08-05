"""Option quote quality filters and no-arbitrage diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import exp, log

from regime.data.options.types import OptionQuote


@dataclass(frozen=True, slots=True)
class LiquidityThresholds:
    min_bid_size: float = 1.0
    min_ask_size: float = 1.0
    min_volume: float = 0.0
    min_open_interest: float = 0.0
    max_spread_pct: float = 1.0


_DEFAULT_LIQUIDITY = LiquidityThresholds()


@dataclass(frozen=True, slots=True)
class QuoteQualityResult:
    is_valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NoArbitrageResult:
    is_valid: bool
    lower_bound: float
    upper_bound: float
    reasons: tuple[str, ...]


def validate_option_quote(
    quote: OptionQuote,
    *,
    as_of: datetime,
    max_age_seconds: float = 900.0,
    liquidity: LiquidityThresholds = _DEFAULT_LIQUIDITY,
) -> QuoteQualityResult:
    """Validate bid/ask, crossed markets, stale quotes, and minimum liquidity."""
    reasons: list[str] = []
    if quote.bid < 0 or quote.ask < 0:
        reasons.append("negative_bid_or_ask")
    if quote.bid > quote.ask:
        reasons.append("crossed_market")
    if (as_of - quote.quote_time).total_seconds() > max_age_seconds:
        reasons.append("stale_quote")
    if quote.mid > 0 and (quote.ask - quote.bid) / quote.mid > liquidity.max_spread_pct:
        reasons.append("wide_spread")
    if quote.bid_size is not None and quote.bid_size < liquidity.min_bid_size:
        reasons.append("low_bid_size")
    if quote.ask_size is not None and quote.ask_size < liquidity.min_ask_size:
        reasons.append("low_ask_size")
    if quote.volume is not None and quote.volume < liquidity.min_volume:
        reasons.append("low_volume")
    if quote.open_interest is not None and quote.open_interest < liquidity.min_open_interest:
        reasons.append("low_open_interest")
    return QuoteQualityResult(is_valid=not reasons, reasons=tuple(reasons))


def no_arbitrage_check(
    quote: OptionQuote,
    *,
    tenor: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> NoArbitrageResult:
    """Check European option premium bounds under continuous rates/dividends."""
    discounted_spot = quote.underlying_price * exp(-dividend_yield * tenor)
    discounted_strike = quote.strike * exp(-rate * tenor)
    if quote.option_type == "call":
        lower = max(discounted_spot - discounted_strike, 0.0)
        upper = discounted_spot
    else:
        lower = max(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike
    reasons = []
    if quote.mid < lower:
        reasons.append("below_lower_bound")
    if quote.mid > upper:
        reasons.append("above_upper_bound")
    return NoArbitrageResult(not reasons, lower, upper, tuple(reasons))


def filter_crossed_markets(quotes: Sequence[OptionQuote]) -> list[OptionQuote]:
    """Drop quotes with bid greater than ask."""
    return [quote for quote in quotes if quote.bid <= quote.ask]


def estimate_forward(
    call: OptionQuote, put: OptionQuote, *, tenor: float, rate: float = 0.0
) -> float:
    """Estimate forward with put-call parity for matched strike/expiry call and put quotes."""
    if call.strike != put.strike or call.expiration != put.expiration:
        raise ValueError("call and put must share strike and expiration")
    return call.strike + exp(rate * tenor) * (call.mid - put.mid)


def implied_dividend_yield(
    *, spot: float, forward: float, tenor: float, rate: float = 0.0
) -> float:
    """Infer continuous dividend yield from spot, forward, tenor, and rate."""
    if tenor <= 0:
        return 0.0
    return rate - log(forward / spot) / tenor
