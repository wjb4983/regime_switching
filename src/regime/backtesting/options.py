"""Options strategy backtesting with regime-aware diagnostics.

The engine is intentionally dataframe-first and conservative about market realism. Each
row in ``option_chain`` is treated as information available at its timestamp only; rows
that are stale, crossed, too wide, too small, or insufficiently liquid are excluded
before strategy selection. Trades are executed at bid/ask-aware prices, include contract
multipliers, reserve capital for premium/margin, and optionally delta hedge the
underlying with configurable frequency and slippage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

FloatFrame: TypeAlias = pd.DataFrame
FloatSeries: TypeAlias = pd.Series
StrategyName = Literal[
    "delta_hedged",
    "volatility_carry",
    "skew_trading",
    "term_structure",
    "tail_hedge",
    "realized_vs_implied",
]


@dataclass(frozen=True)
class OptionBacktestConfig:
    """Configuration for options strategy backtests."""

    strategy: StrategyName = "delta_hedged"
    initial_capital: float = 1_000_000.0
    periods_per_year: int = 252
    execution_delay: int = 1
    max_quote_age_minutes: float = 15.0
    max_bid_ask_spread_pct: float = 0.25
    min_bid: float = 0.05
    min_open_interest: float = 100.0
    min_volume: float = 0.0
    min_quote_size: float = 1.0
    target_contracts: int = 1
    max_contracts: int = 100
    target_delta_abs: float = 0.25
    delta_tolerance: float = 0.10
    target_tenor_days: int = 30
    tenor_tolerance_days: int = 21
    tail_budget_fraction: float = 0.01
    carry_entry_z: float = 0.0
    skew_entry_z: float = 0.0
    term_structure_entry_z: float = 0.0
    realized_vs_implied_entry_z: float = 0.0
    delta_hedging_frequency: int = 1
    hedge_slippage_bps: float = 1.0
    option_fee_per_contract: float = 0.0
    underlying_fee_bps: float = 0.0
    risk_free_rate: float | FloatSeries = 0.0
    dividend_yield: float | FloatSeries = 0.0
    margin_rate_short_option: float = 0.20
    assignment_buffer_days: int = 2
    exercise_intrinsic_threshold: float = 0.0
    confidence_bins: tuple[float, ...] = (0.0, 0.33, 0.67, 1.0)
    volatility_bins: tuple[float, ...] = (0.0, 0.2, 0.4, np.inf)
    liquidity_bins: tuple[float, ...] = (0.0, 100.0, 1_000.0, 10_000.0, np.inf)
    holding_period_bins: tuple[int, ...] = (0, 5, 21, 63, 252)
    allow_american_assignment: bool = True


@dataclass(frozen=True)
class OptionBacktestResult:
    """Full options backtest output."""

    metrics: dict[str, float]
    returns: FloatSeries
    equity_curve: FloatSeries
    trades: FloatFrame
    positions: FloatFrame
    hedge_trades: FloatFrame
    costs: FloatFrame
    capital_usage: FloatSeries
    performance_by: dict[str, FloatFrame]
    filtered_chain: FloatFrame
    rejected_chain: FloatFrame


def run_options_backtest(
    option_chain: FloatFrame,
    underlying_prices: FloatSeries,
    *,
    regime: FloatSeries | None = None,
    confidence: FloatSeries | None = None,
    realized_volatility: FloatSeries | None = None,
    implied_volatility_reference: FloatSeries | None = None,
    config: OptionBacktestConfig | None = None,
) -> OptionBacktestResult:
    """Run a realistic options backtest with strategy selection and dynamic hedging.

    Required chain columns are ``timestamp``, ``expiration``, ``strike``, ``option_type``,
    ``bid``, ``ask``, ``quote_time``, ``underlying_price``, ``delta``, and
    ``implied_volatility``. Optional columns such as ``volume``, ``open_interest``,
    ``bid_size``, ``ask_size``, ``multiplier``, ``skew_score``, ``term_structure_score``,
    and ``realized_implied_spread`` improve filtering and strategy selection.
    """

    cfg = config or OptionBacktestConfig()
    prices = underlying_prices.astype(float).sort_index()
    chain = _prepare_chain(option_chain, prices, cfg)
    tradable = chain[chain["is_tradable"]].copy()
    rejected = chain[~chain["is_tradable"]].copy()
    selections = _select_contracts(tradable, prices.index, cfg, regime, confidence)
    trades = _size_and_price_trades(selections, prices, cfg)
    positions = _build_positions(trades, prices, cfg)
    hedge_trades = _build_delta_hedges(positions, prices, cfg)
    costs = _costs(trades, hedge_trades, cfg)
    pnl = _option_pnl(positions, prices, cfg) + _hedge_pnl(hedge_trades, prices) - costs.sum(axis=1)
    capital_usage = _capital_usage(positions, prices, cfg).clip(lower=1.0)
    returns = (pnl / capital_usage.shift(1).fillna(cfg.initial_capital)).rename("option_return")
    equity = (cfg.initial_capital + pnl.cumsum()).rename("equity")
    metrics = _metrics(returns, pnl, equity, capital_usage, cfg)
    perf = _performance_by(trades, pnl, returns)
    return OptionBacktestResult(
        metrics=metrics,
        returns=returns,
        equity_curve=equity,
        trades=trades,
        positions=positions,
        hedge_trades=hedge_trades,
        costs=costs,
        capital_usage=capital_usage.rename("capital_usage"),
        performance_by=perf,
        filtered_chain=tradable,
        rejected_chain=rejected,
    )


def _prepare_chain(chain: FloatFrame, prices: FloatSeries, cfg: OptionBacktestConfig) -> FloatFrame:
    required = {
        "timestamp",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
        "quote_time",
        "underlying_price",
        "delta",
        "implied_volatility",
    }
    missing = required.difference(chain.columns)
    if missing:
        raise ValueError(f"option_chain missing required columns: {sorted(missing)}")
    out = chain.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["quote_time"] = pd.to_datetime(out["quote_time"])
    out["expiration"] = pd.to_datetime(out["expiration"])
    out = out.sort_values(["timestamp", "expiration", "strike", "option_type"])
    out["mid"] = (out["bid"].astype(float) + out["ask"].astype(float)) / 2.0
    out["spread_pct"] = (out["ask"] - out["bid"]) / out["mid"].replace(0.0, np.nan)
    out["quote_age_minutes"] = (out["timestamp"] - out["quote_time"]).dt.total_seconds() / 60.0
    out["tenor_days"] = (out["expiration"] - out["timestamp"]).dt.days.clip(lower=0)
    out["moneyness"] = out["strike"].astype(float) / out["underlying_price"].astype(float)
    out["multiplier"] = out.get("multiplier", 100.0)
    out["volume"] = out.get("volume", 0.0)
    out["open_interest"] = out.get("open_interest", 0.0)
    out["bid_size"] = out.get("bid_size", np.inf)
    out["ask_size"] = out.get("ask_size", np.inf)
    out["decision_price_available"] = out["timestamp"].isin(prices.index)
    out["crossed_market"] = out["bid"] > out["ask"]
    out["is_stale"] = out["quote_age_minutes"] > cfg.max_quote_age_minutes
    out["is_available"] = out["quote_time"] <= out["timestamp"]
    out["is_liquid"] = (
        (out["bid"] >= cfg.min_bid)
        & (out["spread_pct"] <= cfg.max_bid_ask_spread_pct)
        & (out["open_interest"] >= cfg.min_open_interest)
        & (out["volume"] >= cfg.min_volume)
        & (out[["bid_size", "ask_size"]].min(axis=1) >= cfg.min_quote_size)
    )
    out["is_tradable"] = (
        out["decision_price_available"]
        & out["is_available"]
        & ~out["crossed_market"]
        & ~out["is_stale"]
        & out["is_liquid"]
        & (out["tenor_days"] > cfg.assignment_buffer_days)
    )
    return out


def _select_contracts(
    chain: FloatFrame,
    dates: pd.Index,
    cfg: OptionBacktestConfig,
    regime: FloatSeries | None,
    confidence: FloatSeries | None,
) -> FloatFrame:
    rows: list[pd.Series] = []
    for date in dates:
        today = chain[chain["timestamp"] == date]
        if today.empty:
            continue
        scored = today.copy()
        scored["strategy_signal"] = _strategy_signal(scored, cfg)
        scored = scored[scored["strategy_signal"] != 0.0]
        if scored.empty:
            continue
        scored["selection_error"] = (scored["delta"].abs() - cfg.target_delta_abs).abs() + (
            scored["tenor_days"] - cfg.target_tenor_days
        ).abs() / 365.0
        selected = scored.sort_values(["selection_error", "spread_pct"]).iloc[0].copy()
        selected["regime"] = _lookup(regime, date, "unknown")
        selected["confidence"] = float(_lookup(confidence, date, 0.0))
        rows.append(selected)
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))
    return pd.DataFrame(rows).set_index("timestamp", drop=False)


def _strategy_signal(chain: FloatFrame, cfg: OptionBacktestConfig) -> FloatSeries:
    if cfg.strategy == "tail_hedge":
        return pd.Series(np.where(chain["option_type"] == "put", 1.0, 0.0), index=chain.index)
    if cfg.strategy == "volatility_carry":
        score = chain.get(
            "vol_carry_score", chain["implied_volatility"] - chain.get("realized_volatility", 0.0)
        )
        return -np.sign(score.where(score.abs() >= cfg.carry_entry_z, 0.0))
    if cfg.strategy == "skew_trading":
        score = chain.get("skew_score", 1.0 - chain["moneyness"])
        return np.sign(score.where(score.abs() >= cfg.skew_entry_z, 0.0))
    if cfg.strategy == "term_structure":
        score = chain.get("term_structure_score", chain["tenor_days"] - cfg.target_tenor_days)
        return np.sign(score.where(score.abs() >= cfg.term_structure_entry_z, 0.0))
    if cfg.strategy == "realized_vs_implied":
        score = chain.get(
            "realized_implied_spread",
            chain.get("realized_volatility", 0.0) - chain["implied_volatility"],
        )
        return np.sign(score.where(score.abs() >= cfg.realized_vs_implied_entry_z, 0.0))
    return pd.Series(1.0, index=chain.index)


def _size_and_price_trades(
    selections: FloatFrame, prices: FloatSeries, cfg: OptionBacktestConfig
) -> FloatFrame:
    if selections.empty:
        return _empty_trades(prices.index)
    trades = selections.copy()
    side = np.sign(trades["strategy_signal"].astype(float)).replace(0.0, 1.0)
    if cfg.strategy == "tail_hedge":
        budget = cfg.initial_capital * cfg.tail_budget_fraction
        contracts = np.floor(
            budget / (trades["ask"] * trades["multiplier"].replace(0.0, np.nan))
        ).fillna(0)
    else:
        contracts = pd.Series(cfg.target_contracts, index=trades.index)
    trades["contracts"] = (contracts.clip(lower=0, upper=cfg.max_contracts) * side).astype(float)
    trades["execution_price"] = np.where(trades["contracts"] >= 0, trades["ask"], trades["bid"])
    trades["premium_cashflow"] = (
        -trades["contracts"] * trades["execution_price"] * trades["multiplier"]
    )
    trades["tenor_bucket"] = pd.cut(trades["tenor_days"], [0, 30, 90, 180, 365, np.inf])
    trades["moneyness_bucket"] = pd.cut(trades["moneyness"], [0, 0.9, 0.97, 1.03, 1.1, np.inf])
    trades["delta_bucket"] = pd.cut(trades["delta"], [-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
    trades["liquidity_bucket"] = pd.cut(trades["open_interest"], cfg.liquidity_bins)
    trades["volatility_bucket"] = pd.cut(trades["implied_volatility"], cfg.volatility_bins)
    trades["confidence_bucket"] = pd.cut(
        trades["confidence"], cfg.confidence_bins, include_lowest=True
    )
    trades["holding_period"] = trades["tenor_days"].clip(upper=cfg.target_tenor_days)
    trades["holding_period_bucket"] = pd.cut(trades["holding_period"], cfg.holding_period_bins)
    return trades


def _build_positions(
    trades: FloatFrame, prices: FloatSeries, cfg: OptionBacktestConfig
) -> FloatFrame:
    idx = prices.index
    if trades.empty:
        return pd.DataFrame(index=idx)
    frames = []
    for _, trade in trades.iterrows():
        active = idx[(idx >= trade.name) & (idx <= trade["expiration"])]
        if len(active) == 0:
            continue
        frame = pd.DataFrame(index=active)
        for col in ["contracts", "strike", "delta", "implied_volatility", "multiplier"]:
            frame[col] = float(trade[col])
        frame["underlying_price"] = prices.reindex(active)
        frame["option_type"] = trade["option_type"]
        frame["trade_timestamp"] = trade.name
        frame["expiration"] = trade["expiration"]
        frame["days_to_expiration"] = np.maximum(
            (pd.Timestamp(trade["expiration"]) - active).days, 0
        )
        frame["mark"] = _option_mark(
            frame["option_type"],
            prices.reindex(active),
            frame["strike"],
            frame["days_to_expiration"],
            frame["implied_volatility"],
            _aligned_assumption(cfg.risk_free_rate, active),
            _aligned_assumption(cfg.dividend_yield, active),
            cfg,
        )
        frame["assignment_or_exercise"] = _assignment_or_exercise_flag(frame, cfg)
        frame["regime"] = trade.get("regime", "unknown")
        frames.append(frame)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame(index=idx)


def _build_delta_hedges(
    positions: FloatFrame, prices: FloatSeries, cfg: OptionBacktestConfig
) -> FloatFrame:
    hedge = pd.DataFrame(
        index=prices.index, columns=["shares", "trade_shares", "slippage_cost"]
    ).fillna(0.0)
    if positions.empty:
        return hedge
    exposure = (
        (positions["contracts"] * positions["delta"] * positions["multiplier"])
        .groupby(level=0)
        .sum()
    )
    target = -exposure.reindex(prices.index).fillna(0.0)
    rebalance = pd.Series(
        np.arange(len(target)) % max(cfg.delta_hedging_frequency, 1) == 0, index=target.index
    )
    hedge.loc[rebalance, "shares"] = target[rebalance]
    hedge["shares"] = hedge["shares"].replace(0.0, np.nan).ffill().fillna(0.0)
    hedge["trade_shares"] = hedge["shares"].diff().fillna(hedge["shares"])
    hedge["slippage_cost"] = (
        hedge["trade_shares"].abs() * prices * cfg.hedge_slippage_bps / 10_000.0
    )
    return hedge


def _option_pnl(
    positions: FloatFrame, prices: FloatSeries, cfg: OptionBacktestConfig
) -> FloatSeries:
    if positions.empty:
        return pd.Series(0.0, index=prices.index)
    value = (
        (positions["mark"] * positions["contracts"] * positions["multiplier"])
        .groupby(level=0)
        .sum()
    )
    return value.reindex(prices.index).fillna(0.0).diff().fillna(0.0)


def _hedge_pnl(hedges: FloatFrame, prices: FloatSeries) -> FloatSeries:
    return (hedges["shares"].shift(1).fillna(0.0) * prices.diff().fillna(0.0)).rename("hedge_pnl")


def _costs(trades: FloatFrame, hedges: FloatFrame, cfg: OptionBacktestConfig) -> FloatFrame:
    costs = pd.DataFrame(index=hedges.index)
    costs["option_bid_ask"] = 0.0
    costs["option_fees"] = 0.0
    if not trades.empty:
        spread = (
            (trades["ask"] - trades["bid"]) / 2.0 * trades["contracts"].abs() * trades["multiplier"]
        )
        fees = trades["contracts"].abs() * cfg.option_fee_per_contract
        costs.loc[trades.index, "option_bid_ask"] = spread.groupby(level=0).sum()
        costs.loc[trades.index, "option_fees"] = fees.groupby(level=0).sum()
    costs["hedge_slippage"] = hedges["slippage_cost"]
    costs["underlying_fees"] = hedges["trade_shares"].abs() * cfg.underlying_fee_bps / 10_000.0
    return costs.fillna(0.0)


def _capital_usage(
    positions: FloatFrame, prices: FloatSeries, cfg: OptionBacktestConfig
) -> FloatSeries:
    if positions.empty:
        return pd.Series(cfg.initial_capital, index=prices.index)
    premium = (
        (positions["mark"].abs() * positions["contracts"].abs() * positions["multiplier"])
        .groupby(level=0)
        .sum()
    )
    underlying = prices.reindex(premium.index)
    short_margin = (positions["contracts"].clip(upper=0).abs() * positions["multiplier"]).mul(
        underlying.reindex(positions.index), axis=0
    ).groupby(level=0).sum() * cfg.margin_rate_short_option
    return (premium + short_margin).reindex(prices.index).ffill().fillna(cfg.initial_capital)


def _option_mark(
    option_type: pd.Series | str,
    spot: FloatSeries,
    strike: FloatSeries,
    days_to_expiration: FloatSeries,
    implied_volatility: FloatSeries,
    risk_free_rate: FloatSeries,
    dividend_yield: FloatSeries,
    cfg: OptionBacktestConfig,
) -> FloatSeries:
    intrinsic = _intrinsic_value(option_type, spot, strike)
    time = (days_to_expiration / 365.0).clip(lower=0.0)
    sigma = implied_volatility.clip(lower=1.0e-6)
    sqrt_time = np.sqrt(time.clip(lower=1.0e-12))
    d1 = (
        np.log((spot / strike).clip(lower=1.0e-12))
        + (risk_free_rate - dividend_yield + 0.5 * sigma**2) * time
    ) / (sigma * sqrt_time)
    d2 = d1 - sigma * sqrt_time
    discount = np.exp(-risk_free_rate * time)
    dividend_discount = np.exp(-dividend_yield * time)
    is_call = option_type == "call"
    call = spot * dividend_discount * _normal_cdf(d1) - strike * discount * _normal_cdf(d2)
    put = strike * discount * _normal_cdf(-d2) - spot * dividend_discount * _normal_cdf(-d1)
    european = call.where(is_call, put)
    exercise_window = days_to_expiration <= cfg.assignment_buffer_days
    return european.where(~exercise_window, intrinsic).clip(lower=intrinsic)


def _intrinsic_value(
    option_type: pd.Series | str, spot: FloatSeries, strike: FloatSeries
) -> FloatSeries:
    is_call = option_type == "call"
    call_value = (spot - strike).clip(lower=0.0)
    put_value = (strike - spot).clip(lower=0.0)
    return call_value.where(is_call, put_value)


def _normal_cdf(values: FloatSeries) -> FloatSeries:
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(values.to_numpy() / np.sqrt(2.0)))
    return pd.Series(cdf, index=values.index)


def _aligned_assumption(value: float | FloatSeries, index: pd.Index) -> FloatSeries:
    if isinstance(value, pd.Series):
        return value.astype(float).sort_index().reindex(index, method="ffill").fillna(0.0)
    return pd.Series(float(value), index=index)


def _assignment_or_exercise_flag(frame: FloatFrame, cfg: OptionBacktestConfig) -> FloatSeries:
    if not cfg.allow_american_assignment:
        return pd.Series(False, index=frame.index)
    intrinsic = _intrinsic_value(frame["option_type"], frame["underlying_price"], frame["strike"])
    itm = intrinsic > cfg.exercise_intrinsic_threshold
    return (frame["days_to_expiration"] <= cfg.assignment_buffer_days) & itm


def _performance_by(
    trades: FloatFrame, pnl: FloatSeries, returns: FloatSeries
) -> dict[str, FloatFrame]:
    if trades.empty:
        return {}
    trade_pnl = pnl.reindex(trades.index).fillna(0.0)
    data = trades.assign(
        pnl=trade_pnl.to_numpy(), returns=returns.reindex(trades.index).fillna(0.0).to_numpy()
    )
    groups = {
        "regime": "regime",
        "confidence_bucket": "confidence_bucket",
        "tenor": "tenor_bucket",
        "moneyness": "moneyness_bucket",
        "delta": "delta_bucket",
        "liquidity_bucket": "liquidity_bucket",
        "volatility_bucket": "volatility_bucket",
        "holding_period": "holding_period_bucket",
    }
    return {name: _group_stats(data, column) for name, column in groups.items() if column in data}


def _group_stats(data: FloatFrame, column: str) -> FloatFrame:
    grouped = data.groupby(column, observed=True)
    return grouped.agg(
        trades=("contracts", "count"), pnl=("pnl", "sum"), mean_return=("returns", "mean")
    )


def _metrics(
    returns: FloatSeries,
    pnl: FloatSeries,
    equity: FloatSeries,
    capital: FloatSeries,
    cfg: OptionBacktestConfig,
) -> dict[str, float]:
    vol = returns.std(ddof=0) * np.sqrt(cfg.periods_per_year)
    ann = returns.mean() * cfg.periods_per_year
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_pnl": float(pnl.sum()),
        "total_return": float(equity.iloc[-1] / cfg.initial_capital - 1.0) if len(equity) else 0.0,
        "annualized_return": float(ann),
        "annualized_volatility": float(vol),
        "sharpe": float(ann / vol) if vol > 0 else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "average_capital_usage": float(capital.mean()) if len(capital) else 0.0,
        "peak_capital_usage": float(capital.max()) if len(capital) else 0.0,
    }


def _lookup(series: FloatSeries | None, date: object, default: object) -> object:
    if series is None:
        return default
    aligned = series.sort_index().reindex([date], method="ffill")
    value = aligned.iloc[0]
    return default if pd.isna(value) else value


def _empty_trades(index: pd.Index) -> FloatFrame:
    return pd.DataFrame(index=index[:0])
