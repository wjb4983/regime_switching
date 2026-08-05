"""Realized risk feature builders for return panels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def realized_volatility(
    returns: pd.Series, windows: Sequence[int] = (21, 63, 252), annualization: float = 252.0
) -> pd.DataFrame:
    """Compute annualized realized volatility from simple returns."""
    values = {
        f"realized_volatility_{window}": returns.rolling(window).std() * np.sqrt(annualization)
        for window in windows
    }
    return _clean(pd.DataFrame(values, index=returns.index))


def semivariance(
    returns: pd.Series, windows: Sequence[int] = (21, 63), annualization: float = 252.0
) -> pd.DataFrame:
    """Compute downside and upside realized semivariance."""
    downside = returns.clip(upper=0.0)
    upside = returns.clip(lower=0.0)
    values: dict[str, pd.Series] = {}
    for window in windows:
        values[f"downside_semivariance_{window}"] = (
            downside.pow(2).rolling(window).mean() * annualization
        )
        values[f"upside_semivariance_{window}"] = (
            upside.pow(2).rolling(window).mean() * annualization
        )
    return _clean(pd.DataFrame(values, index=returns.index))


def bipower_variation(
    returns: pd.Series, windows: Sequence[int] = (21, 63), annualization: float = 252.0
) -> pd.DataFrame:
    """Estimate bipower variation using adjacent absolute returns."""
    adjacent_product = returns.abs() * returns.shift(1).abs()
    constant = np.pi / 2.0
    values = {
        f"bipower_variation_{window}": constant
        * adjacent_product.rolling(window).sum()
        * annualization
        / window
        for window in windows
    }
    return _clean(pd.DataFrame(values, index=returns.index))


def jump_variation(
    returns: pd.Series, windows: Sequence[int] = (21, 63), annualization: float = 252.0
) -> pd.DataFrame:
    """Estimate jump variation as positive realized variance minus bipower variation."""
    realized_variance = returns.pow(2)
    bipower = bipower_variation(returns, windows, annualization=annualization)
    values: dict[str, pd.Series] = {}
    for window in windows:
        rv = realized_variance.rolling(window).sum() * annualization / window
        values[f"jump_variation_{window}"] = (rv - bipower[f"bipower_variation_{window}"]).clip(
            lower=0.0
        )
    return _clean(pd.DataFrame(values, index=returns.index))


def volatility_of_volatility(
    returns: pd.Series, vol_window: int = 21, windows: Sequence[int] = (21, 63)
) -> pd.DataFrame:
    """Compute rolling volatility of rolling realized volatility."""
    rolling_vol = returns.rolling(vol_window).std()
    values = {
        f"volatility_of_volatility_{vol_window}_{window}": rolling_vol.rolling(window).std()
        for window in windows
    }
    return _clean(pd.DataFrame(values, index=returns.index))


def tail_estimates(
    returns: pd.Series,
    windows: Sequence[int] = (63, 252),
    quantiles: Sequence[float] = (0.01, 0.05),
) -> pd.DataFrame:
    """Compute rolling lower-tail quantiles and expected shortfall estimates."""
    values: dict[str, pd.Series] = {}
    for window in windows:
        rolling = returns.rolling(window)
        for quantile in quantiles:
            value_at_risk = rolling.quantile(quantile)
            values[f"var_{int(quantile * 100):02d}_{window}"] = value_at_risk
            values[f"expected_shortfall_{int(quantile * 100):02d}_{window}"] = returns.rolling(
                window
            ).apply(lambda x, q=quantile: float(np.mean(x[x <= np.quantile(x, q)])), raw=True)
    return _clean(pd.DataFrame(values, index=returns.index))


def realized_beta(
    asset_returns: pd.Series, market_returns: pd.Series, windows: Sequence[int] = (63, 252)
) -> pd.DataFrame:
    """Compute rolling beta to a market return series."""
    values = {}
    for window in windows:
        covariance = asset_returns.rolling(window).cov(market_returns)
        variance = market_returns.rolling(window).var()
        values[f"realized_beta_{window}"] = covariance / variance
    return _clean(pd.DataFrame(values, index=asset_returns.index))
