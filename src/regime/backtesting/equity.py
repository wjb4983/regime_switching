"""Equity strategy backtesting with exposure, timing, and cost controls.

The module is intentionally dataframe-first: callers provide realized asset returns and
optional target weights, timing signals, factor returns, constraints, and capacity data.
The engine applies an execution delay, strategy/factor gates, portfolio constraints,
volatility targeting, gross/net exposure controls, transaction-cost-aware turnover
control, financing/borrow costs, slippage, and cash handling before computing a broad
set of portfolio diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import pandas as pd

FloatFrame: TypeAlias = pd.DataFrame
FloatSeries: TypeAlias = pd.Series


@dataclass(frozen=True)
class PositionConstraints:
    """Per-asset and portfolio-level position limits."""

    min_weight: float | pd.Series = -1.0
    max_weight: float | pd.Series = 1.0
    max_position_abs: float | None = None
    long_only: bool = False


@dataclass(frozen=True)
class EquityBacktestConfig:
    """Configuration for an equity portfolio backtest."""

    periods_per_year: int = 252
    execution_delay: int = 1
    initial_capital: float = 1.0
    risk_free_rate: float = 0.0
    volatility_target: float | None = None
    volatility_lookback: int = 63
    volatility_floor: float = 1.0e-6
    max_leverage_multiplier: float = 3.0
    gross_exposure_limit: float | None = 1.0
    net_exposure_limit: float | None = 1.0
    max_turnover: float | None = None
    turnover_penalty: float = 0.0
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    borrow_cost_bps: float = 0.0
    cash_return: float | None = None
    var_level: float = 0.95
    expected_shortfall_level: float = 0.95
    tail_quantile: float = 0.05
    utility_risk_aversion: float = 3.0
    constraints: PositionConstraints = field(default_factory=PositionConstraints)


@dataclass(frozen=True)
class EquityBacktestResult:
    """Full equity backtest output."""

    metrics: dict[str, float]
    returns: FloatSeries
    equity_curve: FloatSeries
    weights: FloatFrame
    target_weights: FloatFrame
    turnover: FloatSeries
    gross_exposure: FloatSeries
    net_exposure: FloatSeries
    factor_exposures: FloatFrame
    beta_estimates: FloatFrame
    capacity: FloatFrame
    costs: FloatFrame


def run_equity_backtest(
    returns: FloatFrame,
    target_weights: FloatFrame | FloatSeries | None = None,
    *,
    allocation: FloatFrame | FloatSeries | None = None,
    strategy_gate: FloatSeries | None = None,
    factor_timing: FloatFrame | FloatSeries | None = None,
    factor_returns: FloatFrame | FloatSeries | None = None,
    benchmark_returns: FloatSeries | None = None,
    capacity: FloatFrame | FloatSeries | None = None,
    config: EquityBacktestConfig | None = None,
) -> EquityBacktestResult:
    """Run an equity backtest with realistic implementation controls.

    Args:
        returns: Periodic simple returns by asset.
        target_weights: Desired asset weights before execution delay and controls. If
            omitted, an equal-weight portfolio is used.
        allocation: Optional portfolio allocation/scaling signal. A series scales all
            assets; a frame scales matching asset columns.
        strategy_gate: Optional 0/1 or fractional gate that disables/reduces exposure.
        factor_timing: Optional timing signal. A series scales all assets; a frame with
            asset columns scales positions directly. If columns match factor returns,
            the average factor timing score is used as a portfolio-level gate.
        factor_returns: Optional factor return series/frame for factor exposure and beta
            estimates.
        benchmark_returns: Optional market/benchmark return series for beta estimates.
        capacity: Optional capacity proxy such as ADV dollars by asset. Reported as
            absolute traded notional divided by capacity.
        config: Backtest settings.
    """

    cfg = config or EquityBacktestConfig()
    clean_returns = _as_float_frame(returns, "returns").fillna(0.0)
    assets = clean_returns.columns
    desired = _initial_targets(clean_returns, target_weights)
    desired = _multiply_signal(
        desired, _aligned_multiplier(allocation, clean_returns, "allocation")
    )
    desired = _multiply_signal(
        desired, _aligned_multiplier(strategy_gate, clean_returns, "strategy_gate")
    )
    desired = _multiply_signal(
        desired, _factor_timing_multiplier(factor_timing, factor_returns, clean_returns)
    )
    delayed = desired.shift(cfg.execution_delay).fillna(0.0)

    constrained = _apply_position_constraints(delayed, cfg.constraints, assets)
    controlled = _apply_exposure_limits(
        constrained, cfg.gross_exposure_limit, cfg.net_exposure_limit
    )
    controlled = _apply_volatility_target(controlled, clean_returns, cfg)
    weights = _apply_turnover_control(controlled, cfg)

    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    cash_weight = 1.0 - net

    gross_asset_returns = (weights * clean_returns).sum(axis=1)
    cash_rate = (
        cfg.cash_return
        if cfg.cash_return is not None
        else cfg.risk_free_rate / cfg.periods_per_year
    )
    cost_frame = _implementation_costs(weights, turnover, capacity, cfg)
    portfolio_returns = gross_asset_returns + cash_weight * cash_rate - cost_frame.sum(axis=1)
    equity_curve = (1.0 + portfolio_returns).cumprod() * cfg.initial_capital

    factors = _build_factor_matrix(factor_returns, benchmark_returns, clean_returns.index)
    beta_estimates = _rolling_betas(portfolio_returns, factors, cfg.volatility_lookback)
    factor_exposures = _factor_exposures(weights, clean_returns, factors)
    capacity_frame = _capacity_usage(weights, capacity)
    metrics = _metrics(
        portfolio_returns,
        equity_curve,
        turnover,
        gross,
        net,
        beta_estimates,
        factor_exposures,
        cfg,
    )
    return EquityBacktestResult(
        metrics=metrics,
        returns=portfolio_returns.rename("portfolio_return"),
        equity_curve=equity_curve.rename("equity"),
        weights=weights,
        target_weights=desired.reindex(columns=assets),
        turnover=turnover.rename("turnover"),
        gross_exposure=gross.rename("gross_exposure"),
        net_exposure=net.rename("net_exposure"),
        factor_exposures=factor_exposures,
        beta_estimates=beta_estimates,
        capacity=capacity_frame,
        costs=cost_frame,
    )


def _multiply_signal(weights: FloatFrame, signal: FloatFrame | FloatSeries) -> FloatFrame:
    if isinstance(signal, pd.Series):
        return weights.mul(signal, axis=0)
    return weights.mul(signal, axis=0)


def _as_float_frame(frame: FloatFrame | FloatSeries, name: str) -> FloatFrame:
    if isinstance(frame, pd.Series):
        return frame.astype(float).to_frame(frame.name or name)
    return frame.astype(float).copy()


def _initial_targets(
    returns: FloatFrame, target_weights: FloatFrame | FloatSeries | None
) -> FloatFrame:
    if target_weights is None:
        return pd.DataFrame(
            1.0 / len(returns.columns), index=returns.index, columns=returns.columns
        )
    target = _as_float_frame(target_weights, "target_weight").reindex(returns.index)
    if target.shape[1] == 1 and len(returns.columns) > 1:
        return pd.DataFrame(
            np.repeat(target.to_numpy(), len(returns.columns), axis=1),
            index=returns.index,
            columns=returns.columns,
        )
    return target.reindex(columns=returns.columns).fillna(0.0)


def _aligned_multiplier(
    signal: FloatFrame | FloatSeries | None, returns: FloatFrame, name: str
) -> FloatFrame | FloatSeries:
    if signal is None:
        return pd.Series(1.0, index=returns.index, name=name)
    aligned = _as_float_frame(signal, name).reindex(returns.index).ffill().fillna(0.0)
    if aligned.shape[1] == 1:
        return aligned.iloc[:, 0]
    return aligned.reindex(columns=returns.columns).fillna(1.0)


def _factor_timing_multiplier(
    timing: FloatFrame | FloatSeries | None,
    factors: FloatFrame | FloatSeries | None,
    returns: FloatFrame,
) -> FloatFrame | FloatSeries:
    if timing is None:
        return pd.Series(1.0, index=returns.index, name="factor_timing")
    aligned = _as_float_frame(timing, "factor_timing").reindex(returns.index).ffill().fillna(0.0)
    if set(returns.columns).issubset(aligned.columns):
        return aligned.reindex(columns=returns.columns).fillna(0.0)
    if factors is not None:
        factor_cols = set(_as_float_frame(factors, "factor").columns)
        matching = [column for column in aligned.columns if column in factor_cols]
        if matching:
            return aligned[matching].mean(axis=1)
    if aligned.shape[1] == 1:
        return aligned.iloc[:, 0]
    return aligned.mean(axis=1)


def _apply_position_constraints(
    weights: FloatFrame, constraints: PositionConstraints, assets: pd.Index
) -> FloatFrame:
    lower = 0.0 if constraints.long_only else constraints.min_weight
    clipped = weights.clip(lower=lower, upper=constraints.max_weight, axis=1)
    if constraints.max_position_abs is not None:
        clipped = clipped.clip(
            lower=-constraints.max_position_abs,
            upper=constraints.max_position_abs,
            axis=1,
        )
    return clipped.reindex(columns=assets).fillna(0.0)


def _apply_exposure_limits(
    weights: FloatFrame, gross_limit: float | None, net_limit: float | None
) -> FloatFrame:
    adjusted = weights.copy()
    if gross_limit is not None:
        gross = adjusted.abs().sum(axis=1)
        scale = (gross_limit / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        adjusted = adjusted.mul(scale, axis=0)
    if net_limit is not None:
        net = adjusted.sum(axis=1)
        excess = (net.abs() - net_limit).clip(lower=0.0)
        counts = adjusted.ne(0.0).sum(axis=1).replace(0, len(adjusted.columns))
        correction = np.sign(net) * excess / counts
        adjusted = adjusted.sub(correction, axis=0)
        if gross_limit is not None:
            adjusted = _apply_exposure_limits(adjusted, gross_limit, None)
    return adjusted


def _apply_volatility_target(
    weights: FloatFrame, returns: FloatFrame, cfg: EquityBacktestConfig
) -> FloatFrame:
    if cfg.volatility_target is None:
        return weights
    raw = (weights * returns).sum(axis=1)
    realized = raw.rolling(cfg.volatility_lookback, min_periods=2).std() * np.sqrt(
        cfg.periods_per_year
    )
    scale = (cfg.volatility_target / realized.clip(lower=cfg.volatility_floor)).clip(
        upper=cfg.max_leverage_multiplier
    )
    return weights.mul(scale.shift(1).fillna(1.0), axis=0)


def _apply_turnover_control(weights: FloatFrame, cfg: EquityBacktestConfig) -> FloatFrame:
    rows: list[pd.Series] = []
    previous = pd.Series(0.0, index=weights.columns)
    for _, row in weights.iterrows():
        desired_trade = row - previous
        turnover = float(desired_trade.abs().sum())
        scale = 1.0
        if cfg.max_turnover is not None and turnover > cfg.max_turnover > 0.0:
            scale = min(scale, cfg.max_turnover / turnover)
        if cfg.turnover_penalty > 0.0:
            scale = min(scale, 1.0 / (1.0 + cfg.turnover_penalty * turnover))
        current = previous + desired_trade * scale
        rows.append(current)
        previous = current
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns)


def _implementation_costs(
    weights: FloatFrame,
    turnover: FloatSeries,
    capacity: FloatFrame | FloatSeries | None,
    cfg: EquityBacktestConfig,
) -> FloatFrame:
    trade = weights.diff().abs().fillna(weights.abs())
    transaction = turnover * cfg.transaction_cost_bps / 10_000.0
    slippage = turnover * cfg.slippage_bps / 10_000.0
    borrow = (
        weights.clip(upper=0.0).abs().sum(axis=1)
        * cfg.borrow_cost_bps
        / 10_000.0
        / cfg.periods_per_year
    )
    cap_usage = _capacity_usage(weights, capacity).mean(axis=1).reindex(weights.index).fillna(0.0)
    capacity_cost = cap_usage.pow(2) * turnover
    _ = trade  # retained as the asset-level turnover source for future extensions
    return pd.DataFrame(
        {
            "transaction_cost": transaction,
            "slippage": slippage,
            "borrow_cost": borrow,
            "capacity_cost": capacity_cost,
        },
        index=weights.index,
    )


def _capacity_usage(weights: FloatFrame, capacity: FloatFrame | FloatSeries | None) -> FloatFrame:
    traded = weights.diff().abs().fillna(weights.abs())
    if capacity is None:
        return pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    cap = _as_float_frame(capacity, "capacity").reindex(weights.index).ffill()
    if cap.shape[1] == 1:
        values = np.repeat(cap.to_numpy(), len(weights.columns), axis=1)
        cap = pd.DataFrame(values, index=weights.index, columns=weights.columns)
    cap = cap.reindex(columns=weights.columns).replace(0.0, np.nan)
    return (traded / cap).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_factor_matrix(
    factor_returns: FloatFrame | FloatSeries | None,
    benchmark_returns: FloatSeries | None,
    index: pd.Index,
) -> FloatFrame:
    frames: list[FloatFrame] = []
    if factor_returns is not None:
        frames.append(_as_float_frame(factor_returns, "factor").reindex(index).fillna(0.0))
    if benchmark_returns is not None:
        frames.append(benchmark_returns.astype(float).reindex(index).fillna(0.0).to_frame("benchmark_beta"))
    if not frames:
        return pd.DataFrame(index=index)
    return pd.concat(frames, axis=1)


def _rolling_betas(returns: FloatSeries, factors: FloatFrame, lookback: int) -> FloatFrame:
    if factors.empty:
        return pd.DataFrame(index=returns.index)
    output = pd.DataFrame(index=returns.index)
    for column in factors.columns:
        cov = returns.rolling(lookback, min_periods=3).cov(factors[column])
        var = factors[column].rolling(lookback, min_periods=3).var()
        output[column] = cov / var.replace(0.0, np.nan)
    return output.fillna(0.0)


def _factor_exposures(weights: FloatFrame, returns: FloatFrame, factors: FloatFrame) -> FloatFrame:
    if factors.empty:
        return pd.DataFrame(index=weights.index)
    asset_betas = pd.DataFrame(index=returns.columns)
    for column in factors.columns:
        factor = factors[column]
        cov = returns.apply(lambda asset, factor=factor: asset.cov(factor))
        var = float(factor.var())
        asset_betas[column] = 0.0 if var == 0.0 else cov / var
    exposures = weights @ asset_betas.fillna(0.0)
    return exposures.reindex(weights.index).fillna(0.0)


def _metrics(
    returns: FloatSeries,
    equity_curve: FloatSeries,
    turnover: FloatSeries,
    gross: FloatSeries,
    net: FloatSeries,
    betas: FloatFrame,
    factor_exposures: FloatFrame,
    cfg: EquityBacktestConfig,
) -> dict[str, float]:
    mean = float(returns.mean())
    annualized_return = mean * cfg.periods_per_year
    annualized_vol = float(returns.std(ddof=1) * np.sqrt(cfg.periods_per_year))
    downside = returns[returns < 0.0].std(ddof=1) * np.sqrt(cfg.periods_per_year)
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    sharpe = _safe_div(annualized_return - cfg.risk_free_rate, annualized_vol)
    sortino = _safe_div(annualized_return - cfg.risk_free_rate, float(downside))
    calmar = _safe_div(annualized_return, abs(max_drawdown))
    losses = -returns
    var = float(losses.quantile(cfg.var_level))
    tail_losses = losses[losses >= var]
    expected_shortfall = float(tail_losses.mean()) if not tail_losses.empty else var
    positive = returns[returns > 0.0]
    negative = returns[returns < 0.0]
    profit_factor = _safe_div(float(positive.sum()), abs(float(negative.sum())))
    tail_cutoff = returns.quantile(cfg.tail_quantile)
    tail_performance = float(returns[returns <= tail_cutoff].mean())
    variance = float(returns.var(ddof=1))
    certainty_equivalent = (
        annualized_return
        - 0.5 * cfg.utility_risk_aversion * variance * cfg.periods_per_year
    )
    metrics = {
        "annualized_return": annualized_return,
        "volatility": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "maximum_drawdown": max_drawdown,
        "expected_shortfall": expected_shortfall,
        "var": var,
        "hit_rate": float((returns > 0.0).mean()),
        "profit_factor": profit_factor,
        "turnover": float(turnover.mean()),
        "gross_exposure": float(gross.mean()),
        "net_exposure": float(net.mean()),
        "beta": float(betas.mean().mean()) if not betas.empty else 0.0,
        "factor_exposure": (
            float(factor_exposures.abs().mean().mean())
            if not factor_exposures.empty
            else 0.0
        ),
        "tail_performance": tail_performance,
        "certainty_equivalent_return": float(certainty_equivalent),
        "utility": float(certainty_equivalent),
    }
    return {key: 0.0 if not np.isfinite(value) else float(value) for key, value in metrics.items()}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


__all__ = [
    "EquityBacktestConfig",
    "EquityBacktestResult",
    "PositionConstraints",
    "run_equity_backtest",
]
