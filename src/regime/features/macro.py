"""Macro, cross-asset, rates, credit, inflation, and policy feature builders."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def _rolling_dependence(left: pd.Series, right: pd.Series, name: str, window: int) -> pd.DataFrame:
    return _clean(
        pd.DataFrame(
            {
                f"{name}_correlation_{window}": left.rolling(window).corr(right),
                f"{name}_beta_{window}": left.rolling(window).cov(right)
                / right.rolling(window).var(),
            },
            index=left.index,
        )
    )


def equity_rates_dependence(
    equity_returns: pd.Series, rate_changes: pd.Series, window: int = 63
) -> pd.DataFrame:
    """Compute rolling equity/rates correlation and beta."""
    return _rolling_dependence(equity_returns, rate_changes, "equity_rates", window)


def equity_credit_dependence(
    equity_returns: pd.Series, credit_spread_changes: pd.Series, window: int = 63
) -> pd.DataFrame:
    """Compute rolling equity/credit-spread correlation and beta."""
    return _rolling_dependence(equity_returns, credit_spread_changes, "equity_credit", window)


def dollar_and_commodity_relationships(
    equity_returns: pd.Series,
    dollar_returns: pd.Series,
    commodity_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """Compute rolling equity dependence on dollar and commodity returns."""
    dollar = _rolling_dependence(equity_returns, dollar_returns, "equity_dollar", window)
    commodity = _rolling_dependence(equity_returns, commodity_returns, "equity_commodity", window)
    return _clean(pd.concat([dollar, commodity], axis=1))


def yield_curve_factors(
    curves: pd.DataFrame, short_tenor: str = "2y", belly_tenor: str = "5y", long_tenor: str = "10y"
) -> pd.DataFrame:
    """Compute yield-curve level, slope, and curvature factors."""
    short = curves[short_tenor].astype(float)
    belly = curves[belly_tenor].astype(float)
    long = curves[long_tenor].astype(float)
    return _clean(
        pd.DataFrame(
            {
                "yield_curve_level": curves.astype(float).mean(axis=1),
                "yield_curve_slope": long - short,
                "yield_curve_curvature": 2.0 * belly - short - long,
            },
            index=curves.index,
        )
    )


def credit_spreads(spreads: pd.DataFrame, windows: Sequence[int] = (21, 63)) -> pd.DataFrame:
    """Compute spread levels, changes, and z-scores for credit series."""
    values: dict[str, pd.Series] = {}
    for column in spreads.columns:
        series = spreads[column].astype(float)
        values[f"credit_spread_{column}"] = series
        values[f"credit_spread_change_{column}"] = series.diff()
        for window in windows:
            values[f"credit_spread_zscore_{column}_{window}"] = (
                series - series.rolling(window).mean()
            ) / series.rolling(window).std()
    return _clean(pd.DataFrame(values, index=spreads.index))


def inflation_and_growth_surprises(
    actual: pd.DataFrame, consensus: pd.DataFrame, columns: Sequence[str] = ("inflation", "growth")
) -> pd.DataFrame:
    """Compute macro surprise features as actual releases minus consensus expectations."""
    values: dict[str, pd.Series] = {}
    for column in columns:
        values[f"{column}_surprise"] = actual[column].astype(float) - consensus[column].astype(
            float
        )
        values[f"{column}_surprise_zscore"] = (
            values[f"{column}_surprise"] - values[f"{column}_surprise"].expanding().mean()
        ) / values[f"{column}_surprise"].expanding().std()
    return _clean(pd.DataFrame(values, index=actual.index))


def monetary_policy_features(
    policy: pd.DataFrame,
    target_column: str = "policy_rate",
    expected_column: str = "expected_policy_rate",
    balance_sheet_column: str = "central_bank_assets",
) -> pd.DataFrame:
    """Compute policy-rate surprises, stance, and balance-sheet impulse features."""
    target = policy[target_column].astype(float)
    expected = policy[expected_column].astype(float)
    assets = policy[balance_sheet_column].astype(float)
    return _clean(
        pd.DataFrame(
            {
                "policy_rate": target,
                "policy_rate_change": target.diff(),
                "policy_rate_surprise": target - expected,
                "policy_stance_zscore": (target - target.expanding().mean())
                / target.expanding().std(),
                "balance_sheet_growth": assets.pct_change(),
            },
            index=policy.index,
        )
    )
