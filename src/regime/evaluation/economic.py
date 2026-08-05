"""Economic performance metrics for regime-aware trading evaluations.

The functions in this module accept NumPy-like arrays or pandas objects and return
plain Python floats/dictionaries so they are easy to serialize in experiment
artifacts.  Metrics are computed after aligning inputs on their pandas indexes
when possible and dropping rows with missing required observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

ArrayLike = Sequence[float] | np.ndarray | pd.Series
FrameLike = Sequence[Sequence[float]] | np.ndarray | pd.DataFrame
SliceName = Literal[
    "inferred_regime",
    "model_confidence",
    "market_period",
    "liquidity_bucket",
    "volatility_bucket",
    "asset",
    "sector",
    "option_tenor",
    "moneyness",
    "holding_period",
]

DEFAULT_SLICES: tuple[SliceName, ...] = (
    "inferred_regime",
    "model_confidence",
    "market_period",
    "liquidity_bucket",
    "volatility_bucket",
    "asset",
    "sector",
    "option_tenor",
    "moneyness",
    "holding_period",
)


@dataclass(frozen=True)
class EconomicMetrics:
    """Aggregate portfolio-quality, cost, exposure, capacity, and utility metrics."""

    n_obs: int
    annualized_return: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    expected_shortfall: float
    var: float
    hit_rate: float
    profit_factor: float
    turnover: float
    gross_exposure: float
    net_exposure: float
    beta: float
    tail_performance: float
    transaction_costs: float
    slippage: float
    borrow_costs: float
    option_bid_ask_costs: float
    total_costs: float
    margin_or_capital_usage: float
    utility: float
    certainty_equivalent_return: float
    capacity_proxies: Mapping[str, float]
    factor_exposures: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""
        return {
            "n_obs": self.n_obs,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "expected_shortfall": self.expected_shortfall,
            "var": self.var,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "turnover": self.turnover,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "beta": self.beta,
            "tail_performance": self.tail_performance,
            "transaction_costs": self.transaction_costs,
            "slippage": self.slippage,
            "borrow_costs": self.borrow_costs,
            "option_bid_ask_costs": self.option_bid_ask_costs,
            "total_costs": self.total_costs,
            "margin_or_capital_usage": self.margin_or_capital_usage,
            "utility": self.utility,
            "certainty_equivalent_return": self.certainty_equivalent_return,
            "capacity_proxies": dict(self.capacity_proxies),
            "factor_exposures": dict(self.factor_exposures),
        }


def _series(values: ArrayLike | None, name: str, index: pd.Index | None = None) -> pd.Series:
    if values is None:
        if index is None:
            raise ValueError(f"{name} requires an index when values are omitted")
        return pd.Series(0.0, index=index, name=name, dtype=float)
    if isinstance(values, pd.Series):
        return values.rename(name).astype(float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return pd.Series(arr, index=index, name=name, dtype=float)


def _frame(values: FrameLike | None, name: str, index: pd.Index | None = None) -> pd.DataFrame:
    if values is None:
        return pd.DataFrame(index=index)
    if isinstance(values, pd.DataFrame):
        return values.astype(float).copy()
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    return pd.DataFrame(arr, index=index, columns=[f"{name}_{i}" for i in range(arr.shape[1])])


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _drawdown(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def economic_metrics(
    returns: ArrayLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    alpha: float = 0.05,
    positions: FrameLike | None = None,
    benchmark_returns: ArrayLike | None = None,
    factor_returns: FrameLike | None = None,
    transaction_costs: ArrayLike | None = None,
    slippage: ArrayLike | None = None,
    borrow_costs: ArrayLike | None = None,
    option_bid_ask_costs: ArrayLike | None = None,
    margin_or_capital_usage: ArrayLike | None = None,
    volume: ArrayLike | None = None,
    market_impact: ArrayLike | None = None,
    risk_aversion: float = 1.0,
) -> EconomicMetrics:
    """Compute aggregate economic metrics for an aligned strategy return series.

    Returns are interpreted as periodic simple returns.  Cost inputs should be in
    return units for the same periods; if omitted they default to zero.  Exposures
    and turnover are inferred from ``positions`` when supplied.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    ret = _series(returns, "returns").replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        raise ValueError("returns must contain at least one finite observation")
    index = ret.index
    costs = {
        "transaction_costs": _series(transaction_costs, "transaction_costs", index),
        "slippage": _series(slippage, "slippage", index),
        "borrow_costs": _series(borrow_costs, "borrow_costs", index),
        "option_bid_ask_costs": _series(option_bid_ask_costs, "option_bid_ask_costs", index),
    }
    data = (
        pd.concat([ret, *costs.values()], axis=1, join="inner")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    ret = data["returns"]
    excess = ret - risk_free_rate / periods_per_year
    mean_return = float(ret.mean())
    ann_return = float((1.0 + mean_return) ** periods_per_year - 1.0)
    vol = float(ret.std(ddof=1) * np.sqrt(periods_per_year)) if ret.size > 1 else 0.0
    downside = excess[excess < 0.0]
    downside_dev = (
        float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year))
        if not downside.empty
        else 0.0
    )
    sharpe = _safe_divide(float(excess.mean() * periods_per_year), vol)
    sortino = _safe_divide(float(excess.mean() * periods_per_year), downside_dev)
    mdd = float(_drawdown(ret).min())
    calmar = _safe_divide(ann_return, abs(mdd))
    var_value = float(ret.quantile(alpha))
    tail = ret[ret <= var_value]
    es = float(tail.mean()) if not tail.empty else var_value
    gains = ret[ret > 0.0].sum()
    losses = ret[ret < 0.0].sum()

    pos = _frame(positions, "position", index).reindex(ret.index)
    turnover = float(pos.diff().abs().sum(axis=1).mean()) if not pos.empty else 0.0
    gross = float(pos.abs().sum(axis=1).mean()) if not pos.empty else 0.0
    net = float(pos.sum(axis=1).mean()) if not pos.empty else 0.0

    bench = (
        _series(benchmark_returns, "benchmark", index) if benchmark_returns is not None else None
    )
    if bench is not None:
        aligned = (
            pd.concat([ret, bench], axis=1, join="inner")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        beta = (
            _safe_divide(float(aligned.cov().iloc[0, 1]), float(aligned["benchmark"].var(ddof=1)))
            if len(aligned) > 1
            else float("nan")
        )
        threshold = float(aligned["benchmark"].quantile(alpha))
        stress = aligned.loc[aligned["benchmark"] <= threshold, "returns"]
        tail_perf = float(stress.mean()) if not stress.empty else float("nan")
    else:
        beta = float("nan")
        tail_perf = es

    factors = _frame(factor_returns, "factor", index)
    exposures: dict[str, float] = {}
    for column in factors.columns:
        aligned = (
            pd.concat([ret, factors[column]], axis=1, join="inner")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        exposures[str(column)] = (
            _safe_divide(float(aligned.cov().iloc[0, 1]), float(aligned[column].var(ddof=1)))
            if len(aligned) > 1
            else float("nan")
        )

    margin = (
        _series(margin_or_capital_usage, "margin_or_capital_usage", index)
        if margin_or_capital_usage is not None
        else None
    )
    volume_s = _series(volume, "volume", index) if volume is not None else None
    impact_s = _series(market_impact, "market_impact", index) if market_impact is not None else None
    capacity = {
        "average_turnover": turnover,
        "average_gross_exposure": gross,
        "average_margin_or_capital_usage": float(margin.reindex(ret.index).mean())
        if margin is not None
        else gross,
        "average_volume": float(volume_s.reindex(ret.index).mean())
        if volume_s is not None
        else float("nan"),
        "average_market_impact": float(impact_s.reindex(ret.index).mean())
        if impact_s is not None
        else float("nan"),
    }
    total_cost = sum(float(data[name].mean()) for name in costs)
    variance = float(ret.var(ddof=1)) if ret.size > 1 else 0.0
    utility = float(mean_return - 0.5 * risk_aversion * variance)
    cer = float(utility * periods_per_year)

    return EconomicMetrics(
        n_obs=int(ret.size),
        annualized_return=ann_return,
        volatility=vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        expected_shortfall=es,
        var=var_value,
        hit_rate=float((ret > 0.0).mean()),
        profit_factor=_safe_divide(float(gains), abs(float(losses))),
        turnover=turnover,
        gross_exposure=gross,
        net_exposure=net,
        beta=beta,
        tail_performance=tail_perf,
        transaction_costs=float(data["transaction_costs"].mean()),
        slippage=float(data["slippage"].mean()),
        borrow_costs=float(data["borrow_costs"].mean()),
        option_bid_ask_costs=float(data["option_bid_ask_costs"].mean()),
        total_costs=total_cost,
        margin_or_capital_usage=capacity["average_margin_or_capital_usage"],
        utility=utility,
        certainty_equivalent_return=cer,
        capacity_proxies=capacity,
        factor_exposures=exposures,
    )


def conditional_economic_metrics(
    returns: ArrayLike,
    slices: Mapping[str, Sequence[Any] | pd.Series],
    **metric_kwargs: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute economic metrics by conditional slice and slice value."""
    ret = _series(returns, "returns")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for slice_name, labels in slices.items():
        label_series = pd.Series(
            labels,
            index=ret.index if not isinstance(labels, pd.Series) else labels.index,
            name=slice_name,
        )
        aligned = pd.concat([ret, label_series], axis=1, join="inner").dropna()
        result[slice_name] = {}
        for value, group in aligned.groupby(slice_name, sort=True):
            result[slice_name][str(value)] = economic_metrics(
                group["returns"], **metric_kwargs
            ).as_dict()
    return result


def regime_baselines(
    returns: ArrayLike,
    *,
    regimes: Sequence[Any] | pd.Series | None = None,
    signals: Sequence[float] | pd.Series | None = None,
    oracle_returns: ArrayLike | None = None,
    periods_per_year: int = 252,
) -> dict[str, dict[str, Any]]:
    """Return required no-regime, simple-rule, and oracle/synthetic upper-bound baselines."""
    ret = _series(returns, "returns")
    out = {"no_regime": economic_metrics(ret, periods_per_year=periods_per_year).as_dict()}
    if signals is None:
        rule_position = np.sign(ret.rolling(20, min_periods=1).mean()).replace(0.0, 1.0)
    else:
        rule_position = np.sign(_series(signals, "signals", ret.index)).replace(0.0, 1.0)
    simple = ret * rule_position.reindex(ret.index).fillna(0.0)
    out["simple_rule"] = economic_metrics(
        simple, positions=rule_position, periods_per_year=periods_per_year
    ).as_dict()
    if oracle_returns is not None:
        oracle = _series(oracle_returns, "oracle_returns", ret.index)
    elif regimes is not None:
        labels = pd.Series(
            regimes, index=ret.index if not isinstance(regimes, pd.Series) else regimes.index
        )
        aligned = pd.concat([ret, labels.rename("regime")], axis=1, join="inner").dropna()
        signs = aligned.groupby("regime")["returns"].transform(
            lambda x: 1.0 if x.mean() >= 0.0 else -1.0
        )
        oracle = aligned["returns"] * signs
    else:
        oracle = ret.abs()
    out["oracle_or_synthetic_upper_bound"] = economic_metrics(
        oracle, periods_per_year=periods_per_year
    ).as_dict()
    return out


__all__ = [
    "DEFAULT_SLICES",
    "EconomicMetrics",
    "conditional_economic_metrics",
    "economic_metrics",
    "regime_baselines",
]
