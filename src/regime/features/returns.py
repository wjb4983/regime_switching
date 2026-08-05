"""Return, momentum, trend, gap, and drawdown feature builders."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _column_frame(index: pd.Index, values: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(values, index=index).replace([np.inf, -np.inf], np.nan)


def multi_horizon_returns(
    prices: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 21, 63),
    price_column: str = "close",
) -> pd.DataFrame:
    """Compute percentage returns over several lookback horizons."""
    close = prices[price_column].astype(float)
    return _column_frame(
        prices.index,
        {f"return_{horizon}": close.pct_change(horizon) for horizon in horizons},
    )


def momentum(
    prices: pd.DataFrame,
    horizons: Sequence[int] = (21, 63, 126, 252),
    skip: int = 1,
    price_column: str = "close",
) -> pd.DataFrame:
    """Compute skip-period total-return momentum signals."""
    close = prices[price_column].astype(float)
    anchor = close.shift(skip)
    return _column_frame(
        prices.index,
        {
            f"momentum_{horizon}_skip_{skip}": anchor / close.shift(horizon + skip) - 1.0
            for horizon in horizons
        },
    )


def reversal(
    prices: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 21),
    price_column: str = "close",
) -> pd.DataFrame:
    """Compute short-horizon reversal as the negative trailing return."""
    close = prices[price_column].astype(float)
    return _column_frame(
        prices.index,
        {f"reversal_{horizon}": -close.pct_change(horizon) for horizon in horizons},
    )


def moving_average_relationships(
    prices: pd.DataFrame,
    windows: Sequence[int] = (10, 20, 50, 200),
    price_column: str = "close",
) -> pd.DataFrame:
    """Measure distance from moving averages and selected fast/slow ratios."""
    close = prices[price_column].astype(float)
    moving_averages = {window: close.rolling(window).mean() for window in windows}
    values: dict[str, pd.Series] = {
        f"ma_distance_{window}": close / average - 1.0
        for window, average in moving_averages.items()
    }
    ordered_windows = tuple(windows)
    for fast, slow in pairwise(ordered_windows):
        values[f"ma_ratio_{fast}_{slow}"] = moving_averages[fast] / moving_averages[slow] - 1.0
    return _column_frame(prices.index, values)


def trend_strength(
    prices: pd.DataFrame,
    windows: Sequence[int] = (20, 60),
    price_column: str = "close",
) -> pd.DataFrame:
    """Estimate normalized linear trend slope and directional efficiency."""
    close = prices[price_column].astype(float)
    values: dict[str, pd.Series] = {}
    for window in windows:
        x = np.arange(window, dtype=float)
        x_centered = x - x.mean()
        denominator = float(np.square(x_centered).sum())

        def slope(
            sample: pd.Series,
            centered: np.ndarray = x_centered,
            scale: float = denominator,
        ) -> float:
            y = sample.to_numpy(dtype=float)
            return float(np.dot(centered, y - y.mean()) / scale)

        values[f"trend_slope_{window}"] = close.rolling(window).apply(slope, raw=False) / close
        path = close.diff().abs().rolling(window).sum()
        values[f"trend_efficiency_{window}"] = close.diff(window).abs() / path
    return _column_frame(prices.index, values)


def gap_and_overnight_returns(
    prices: pd.DataFrame,
    open_column: str = "open",
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
) -> pd.DataFrame:
    """Compute overnight gaps, intraday returns, and normalized gap location."""
    open_ = prices[open_column].astype(float)
    high = prices[high_column].astype(float)
    low = prices[low_column].astype(float)
    close = prices[close_column].astype(float)
    previous_close = close.shift(1)
    return _column_frame(
        prices.index,
        {
            "overnight_return": open_ / previous_close - 1.0,
            "intraday_return": close / open_ - 1.0,
            "close_to_close_return": close.pct_change(),
            "gap_range_position": (open_ - previous_close) / (high - low),
        },
    )


def drawdown(prices: pd.DataFrame, window: int = 252, price_column: str = "close") -> pd.DataFrame:
    """Compute trailing drawdown and maximum drawdown over a lookback window."""
    close = prices[price_column].astype(float)
    running_peak = close.rolling(window, min_periods=1).max()
    drawdown_series = close / running_peak - 1.0
    return _column_frame(
        prices.index,
        {
            f"drawdown_{window}": drawdown_series,
            f"max_drawdown_{window}": drawdown_series.rolling(window, min_periods=1).min(),
        },
    )
