"""Tests for option-data normalization, quality, and pricing helpers."""

from datetime import UTC, date, datetime, timedelta

import pytest

from regime.data.options import (
    BlackScholesInputs,
    LiquidityThresholds,
    VolSurface,
    build_delta_moneyness_tenor_grid,
    compute_option_factors,
    delta_hedged_return_inputs,
    estimate_forward,
    implied_correlation,
    implied_volatility,
    interpolate_surface,
    normalize_expiration,
    normalize_option_quote,
    normalize_option_symbol,
    normalize_option_type,
    normalize_strike,
    option_greeks,
    validate_option_quote,
    variance_risk_premium_inputs,
)
from regime.data.options.pricing import black_scholes_price
from regime.data.options.quality import no_arbitrage_check


def test_normalization_and_multiplier_adjustment_metadata() -> None:
    quote = normalize_option_quote(
        {
            "underlying": "spy",
            "symbol": "SPY 260116 C 00450000",
            "expiration": "2026-01-16",
            "strike": "450",
            "right": "C",
            "bid": 10,
            "ask": 10.4,
            "quote_time": "2026-01-02T14:30:00+00:00",
            "spot": 455,
            "multiplier": 100,
            "adjusted": True,
            "adjustment_factor": 0.5,
            "deliverable": "50 shares + cash",
        }
    )

    assert normalize_option_symbol("SPY 260116 C 00450000") == "SPY260116C00450000"
    assert normalize_expiration("260116") == date(2026, 1, 16)
    assert normalize_strike("00450000", osi_encoded=True) == 450.0
    assert normalize_option_type("put") == "put"
    assert quote.underlying == "SPY"
    assert quote.option_type == "call"
    assert quote.notional_mid == pytest.approx(1020.0)
    assert quote.adjustment.adjusted is True
    assert quote.adjustment.deliverable == "50 shares + cash"


def test_quote_quality_filters_crossed_stale_and_liquidity() -> None:
    quote = normalize_option_quote(
        {
            "underlying": "SPY",
            "symbol": "SPY260116P00450000",
            "expiration": "2026-01-16",
            "strike": 450,
            "right": "P",
            "bid": 4.0,
            "ask": 3.5,
            "quote_time": "2026-01-02T14:00:00+00:00",
            "spot": 455,
            "bid_size": 0,
            "ask_size": 0,
            "volume": 0,
            "open_interest": 0,
        }
    )

    result = validate_option_quote(
        quote,
        as_of=datetime(2026, 1, 2, 15, 0, tzinfo=UTC),
        liquidity=LiquidityThresholds(min_volume=1, min_open_interest=1),
    )

    assert result.is_valid is False
    assert "crossed_market" in result.reasons
    assert "stale_quote" in result.reasons
    assert "low_volume" in result.reasons


def test_no_arbitrage_forward_iv_and_greeks() -> None:
    call = normalize_option_quote(
        {
            "underlying": "XYZ",
            "symbol": "XYZ260116C00100000",
            "expiration": "2026-01-16",
            "strike": 100,
            "right": "C",
            "bid": 5.0,
            "ask": 5.2,
            "quote_time": datetime.now(UTC) - timedelta(seconds=5),
            "spot": 100,
        }
    )
    put = normalize_option_quote(
        {
            "underlying": "XYZ",
            "symbol": "XYZ260116P00100000",
            "expiration": "2026-01-16",
            "strike": 100,
            "right": "P",
            "bid": 4.0,
            "ask": 4.2,
            "quote_time": datetime.now(UTC) - timedelta(seconds=5),
            "spot": 100,
        }
    )
    inputs = BlackScholesInputs(spot=100, strike=100, tenor=0.5, rate=0.03, dividend_yield=0.01)
    price = black_scholes_price(inputs, 0.2, "call")

    assert no_arbitrage_check(call, tenor=0.5, rate=0.03).is_valid is True
    assert estimate_forward(call, put, tenor=0.5, rate=0.03) > 100
    assert implied_volatility(price, inputs, "call") == pytest.approx(0.2, rel=1e-6)
    greeks = option_greeks(inputs, 0.2, "call")
    assert 0 < greeks.delta < 1
    assert greeks.vega > 0


def test_surface_and_factor_inputs() -> None:
    surface = VolSurface.from_rows(((0.1, 0.9, 0.22), (0.5, 1.0, 0.20), (1.0, 1.1, 0.24)))
    grid = build_delta_moneyness_tenor_grid(
        deltas=(-0.25, 0.25), moneyness=(0.9, 1.0), tenors=(0.1, 1.0)
    )
    factors = compute_option_factors(((0.1, -0.25, 0.30), (0.1, 0.25, 0.20), (1.0, 0.5, 0.25)))

    assert len(grid) == 8
    assert interpolate_surface(surface, tenor=0.5, moneyness=1.0) == pytest.approx(0.20)
    assert factors.skew == pytest.approx(0.10)
    assert implied_correlation(0.04, [0.05, 0.03], [0.6, 0.4]) == pytest.approx(0.925, rel=1e-2)
    assert variance_risk_premium_inputs(implied_volatility=0.2, realized_volatility=0.1)[
        "vrp"
    ] == pytest.approx(0.03)
    hedged = delta_hedged_return_inputs(
        option_return=0.03,
        underlying_return=0.01,
        greeks=option_greeks(BlackScholesInputs(100, 100, 0.5), 0.2, "call"),
    )
    assert "delta_hedged_return" in hedged
