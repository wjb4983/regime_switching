"""Liquidity, trading activity, and financing proxy feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def bid_ask_spreads(
    quotes: pd.DataFrame, bid_column: str = "bid", ask_column: str = "ask"
) -> pd.DataFrame:
    """Compute quoted and proportional bid/ask spreads."""
    bid = quotes[bid_column].astype(float)
    ask = quotes[ask_column].astype(float)
    midpoint = (bid + ask) / 2.0
    return _clean(
        pd.DataFrame(
            {"quoted_spread": ask - bid, "proportional_spread": (ask - bid) / midpoint},
            index=quotes.index,
        )
    )


def volume(trades: pd.DataFrame, volume_column: str = "volume", window: int = 21) -> pd.DataFrame:
    """Compute raw volume, rolling average volume, and volume shock."""
    vol = trades[volume_column].astype(float)
    average = vol.rolling(window).mean()
    return _clean(
        pd.DataFrame(
            {
                "volume": vol,
                f"average_volume_{window}": average,
                f"volume_shock_{window}": vol / average - 1.0,
            },
            index=trades.index,
        )
    )


def turnover(
    trades: pd.DataFrame, shares_outstanding: float | pd.Series, volume_column: str = "volume"
) -> pd.DataFrame:
    """Compute turnover as volume divided by shares outstanding."""
    shares = (
        shares_outstanding
        if isinstance(shares_outstanding, pd.Series)
        else pd.Series(shares_outstanding, index=trades.index)
    )
    return _clean(
        pd.DataFrame({"turnover": trades[volume_column].astype(float) / shares}, index=trades.index)
    )


def amihud_style_measures(
    returns: pd.Series, dollar_volume: pd.Series, window: int = 21
) -> pd.DataFrame:
    """Compute Amihud illiquidity and its rolling average."""
    daily = returns.abs() / dollar_volume.replace(0.0, np.nan)
    return _clean(
        pd.DataFrame(
            {
                "amihud_illiquidity": daily,
                f"amihud_illiquidity_{window}": daily.rolling(window).mean(),
            },
            index=returns.index,
        )
    )


def order_imbalance(
    order_flow: pd.DataFrame, buy_column: str = "buy_volume", sell_column: str = "sell_volume"
) -> pd.DataFrame:
    """Compute signed and normalized order imbalance when buy/sell flow is available."""
    buy = order_flow[buy_column].astype(float)
    sell = order_flow[sell_column].astype(float)
    return _clean(
        pd.DataFrame(
            {
                "signed_order_imbalance": buy - sell,
                "normalized_order_imbalance": (buy - sell) / (buy + sell),
            },
            index=order_flow.index,
        )
    )


def market_depth(
    order_book: pd.DataFrame, bid_size_column: str = "bid_size", ask_size_column: str = "ask_size"
) -> pd.DataFrame:
    """Compute displayed top-of-book depth."""
    bid_size = order_book[bid_size_column].astype(float)
    ask_size = order_book[ask_size_column].astype(float)
    return _clean(
        pd.DataFrame(
            {
                "market_depth": bid_size + ask_size,
                "depth_imbalance": (bid_size - ask_size) / (bid_size + ask_size),
            },
            index=order_book.index,
        )
    )


def quote_dispersion(
    quotes: pd.DataFrame,
    bid_column: str = "bid",
    ask_column: str = "ask",
    venue_column: str = "venue",
) -> pd.DataFrame:
    """Compute cross-venue midpoint dispersion for timestamped quotes."""
    midpoint = (quotes[bid_column].astype(float) + quotes[ask_column].astype(float)) / 2.0
    data = quotes.assign(_midpoint=midpoint)
    dispersion = (
        data.groupby(level=0 if venue_column in quotes.columns else None)["_midpoint"].std()
        if venue_column in quotes.columns
        else midpoint.rolling(5).std()
    )
    return _clean(pd.DataFrame({"quote_dispersion": dispersion}))


def funding_and_borrow_proxies(
    data: pd.DataFrame,
    rebate_column: str = "borrow_rebate",
    utilization_column: str = "borrow_utilization",
    funding_column: str = "funding_rate",
) -> pd.DataFrame:
    """Return funding, borrow utilization, and specialness proxies when present."""
    values: dict[str, pd.Series] = {}
    if rebate_column in data:
        values["borrow_specialness"] = -data[rebate_column].astype(float)
    if utilization_column in data:
        values["borrow_utilization"] = data[utilization_column].astype(float)
    if funding_column in data:
        values["funding_rate"] = data[funding_column].astype(float)
    return _clean(pd.DataFrame(values, index=data.index))
