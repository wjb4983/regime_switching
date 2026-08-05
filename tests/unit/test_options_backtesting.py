from __future__ import annotations

import pandas as pd

from regime.backtesting.options import OptionBacktestConfig, run_options_backtest


def test_options_backtest_filters_selects_hedges_and_reports_buckets() -> None:
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    prices = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0], index=dates)
    rows = []
    for timestamp, spot in prices.items():
        rows.append(
            {
                "timestamp": timestamp,
                "expiration": timestamp + pd.Timedelta(days=30),
                "strike": 95.0,
                "option_type": "put",
                "bid": 1.00,
                "ask": 1.10,
                "quote_time": timestamp - pd.Timedelta(minutes=1),
                "underlying_price": spot,
                "delta": -0.25,
                "implied_volatility": 0.25,
                "volume": 50.0,
                "open_interest": 500.0,
                "bid_size": 10.0,
                "ask_size": 10.0,
            }
        )
        rows.append(
            {
                "timestamp": timestamp,
                "expiration": timestamp + pd.Timedelta(days=30),
                "strike": 105.0,
                "option_type": "call",
                "bid": 1.20,
                "ask": 1.10,
                "quote_time": timestamp - pd.Timedelta(minutes=1),
                "underlying_price": spot,
                "delta": 0.25,
                "implied_volatility": 0.25,
                "volume": 50.0,
                "open_interest": 500.0,
                "bid_size": 10.0,
                "ask_size": 10.0,
            }
        )
    chain = pd.DataFrame(rows)

    result = run_options_backtest(
        chain,
        prices,
        regime=pd.Series(["calm", "calm", "stress", "stress", "calm"], index=dates),
        confidence=pd.Series([0.1, 0.4, 0.8, 0.7, 0.2], index=dates),
        config=OptionBacktestConfig(strategy="tail_hedge", target_contracts=1),
    )

    assert len(result.trades) == len(dates)
    assert len(result.rejected_chain) == len(dates)
    assert result.hedge_trades["trade_shares"].abs().sum() > 0
    assert result.capital_usage.max() > 0
    assert {
        "regime",
        "confidence_bucket",
        "tenor",
        "moneyness",
        "delta",
        "liquidity_bucket",
        "volatility_bucket",
        "holding_period",
    }.issubset(result.performance_by)
