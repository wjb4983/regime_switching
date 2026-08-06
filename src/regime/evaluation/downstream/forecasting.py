"""Paired downstream forecasting tasks for regime-aware evaluation.

The functions in this module encode a strict comparison contract for downstream
experiments that ask whether adding live regime probabilities helps a forecast or
trading decision.  Every pair compares the same model class under two information
sets:

* regime agnostic: ``f(x_t)``
* regime aware: ``f(x_t, p(z_t | F_t))``

All non-regime inputs must be identical across the pair: base features, train/test
periods, retraining cadence, execution delay, and (when a forecast becomes a trade)
cost assumptions.  Statistical forecast quality and economic decision quality are
reported separately so a lower forecast loss is not conflated with a better strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, TypeAlias

import numpy as np
import pandas as pd

from regime.evaluation.comparison import ComparisonTestResult, diebold_mariano_test
from regime.evaluation.metrics import ProbabilityKind, descriptor
from regime.evaluation.statistical import rmse

METRICS = {
    "mse": descriptor("mse", ("truth", "prediction"), "minimize"),
    "mae": descriptor("mae", ("truth", "prediction"), "minimize"),
    "rmse": descriptor("rmse", ("truth", "prediction"), "minimize"),
    "qlike": descriptor("qlike", ("truth", "prediction"), "minimize"),
    "brier": descriptor(
        "brier", ("truth", "probabilities"), "minimize", ProbabilityKind.PREDICTIVE
    ),
    "negative_spearman": descriptor("negative_spearman", ("truth", "prediction"), "minimize"),
}

ArrayLike: TypeAlias = Sequence[float] | np.ndarray | pd.Series
FrameLike: TypeAlias = np.ndarray | pd.DataFrame
TaskKind: TypeAlias = Literal[
    "equity_return",
    "realized_volatility",
    "correlation",
    "tail_event",
    "cross_sectional_ranking",
    "factor_timing",
]
LossMetric: TypeAlias = Literal["mse", "mae", "rmse", "qlike", "brier", "negative_spearman"]
EconomicMetric: TypeAlias = Literal[
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "turnover",
]


class ForecastEstimator(Protocol):
    """Minimal estimator protocol for the paired rolling evaluation."""

    def fit(
        self, features: pd.DataFrame, target: pd.Series | pd.DataFrame
    ) -> ForecastEstimator: ...

    def predict(self, features: pd.DataFrame) -> ArrayLike | FrameLike: ...


class DownstreamTaskName(StrEnum):
    """Supported paired downstream tasks."""

    EQUITY_RETURN = "equity_return"
    REALIZED_VOLATILITY = "realized_volatility"
    CORRELATION = "correlation"
    TAIL_EVENT = "tail_event"
    CROSS_SECTIONAL_RANKING = "cross_sectional_ranking"
    FACTOR_TIMING = "factor_timing"


@dataclass(frozen=True)
class ComparisonDesign:
    """Experiment settings that must be identical for both arms of a paired task."""

    train_start: pd.Timestamp | str
    train_end: pd.Timestamp | str
    test_start: pd.Timestamp | str
    test_end: pd.Timestamp | str
    retraining_cadence: str
    execution_delay: int = 0
    cost_assumptions: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DownstreamTaskSpec:
    """Configuration for one downstream paired task."""

    name: TaskKind
    target_column: str
    loss_metric: LossMetric
    feature_columns: tuple[str, ...]
    regime_probability_columns: tuple[str, ...]
    design: ComparisonDesign
    prediction_to_position: (
        Callable[[pd.Series | pd.DataFrame], pd.Series | pd.DataFrame] | None
    ) = None
    realized_return_column: str | None = None
    periods_per_year: int = 252


@dataclass(frozen=True)
class StatisticalImprovement:
    """Forecast-quality improvement of the regime-aware arm over the agnostic arm."""

    agnostic_loss: float
    aware_loss: float
    absolute_improvement: float
    relative_improvement: float
    test: ComparisonTestResult


@dataclass(frozen=True)
class EconomicImprovement:
    """Trading-decision improvement, reported independently from forecast loss."""

    agnostic_metrics: Mapping[str, float]
    aware_metrics: Mapping[str, float]
    metric_deltas: Mapping[str, float]
    cost_assumptions: Mapping[str, float]


@dataclass(frozen=True)
class PairedDownstreamResult:
    """Result for one strict ``f(x_t)`` versus ``f(x_t, p(z_t | F_t))`` task."""

    task: DownstreamTaskSpec
    agnostic_predictions: pd.Series | pd.DataFrame
    aware_predictions: pd.Series | pd.DataFrame
    statistical: StatisticalImprovement
    economic: EconomicImprovement | None


def equity_return_forecasting_task(
    *,
    target_column: str,
    feature_columns: Sequence[str],
    regime_probability_columns: Sequence[str],
    design: ComparisonDesign,
    prediction_to_position: Callable[[pd.Series | pd.DataFrame], pd.Series | pd.DataFrame]
    | None = None,
    realized_return_column: str | None = None,
) -> DownstreamTaskSpec:
    """Create a paired equity-return forecasting task using squared-error loss."""
    return _task(
        "equity_return",
        target_column,
        "mse",
        feature_columns,
        regime_probability_columns,
        design,
        prediction_to_position=prediction_to_position,
        realized_return_column=realized_return_column or target_column,
    )


def realized_volatility_forecasting_task(
    *,
    target_column: str,
    feature_columns: Sequence[str],
    regime_probability_columns: Sequence[str],
    design: ComparisonDesign,
) -> DownstreamTaskSpec:
    """Create a paired realized-volatility forecasting task using QLIKE loss."""
    return _task(
        "realized_volatility",
        target_column,
        "qlike",
        feature_columns,
        regime_probability_columns,
        design,
    )


def correlation_forecasting_task(
    *,
    target_column: str,
    feature_columns: Sequence[str],
    regime_probability_columns: Sequence[str],
    design: ComparisonDesign,
) -> DownstreamTaskSpec:
    """Create a paired correlation forecasting task using squared-error loss."""
    return _task(
        "correlation", target_column, "mse", feature_columns, regime_probability_columns, design
    )


def tail_event_prediction_task(
    *,
    target_column: str,
    feature_columns: Sequence[str],
    regime_probability_columns: Sequence[str],
    design: ComparisonDesign,
) -> DownstreamTaskSpec:
    """Create a paired drawdown or tail-event prediction task using Brier loss."""
    return _task(
        "tail_event", target_column, "brier", feature_columns, regime_probability_columns, design
    )


def cross_sectional_ranking_task(
    *,
    target_column: str,
    feature_columns: Sequence[str],
    regime_probability_columns: Sequence[str],
    design: ComparisonDesign,
) -> DownstreamTaskSpec:
    """Create a paired cross-sectional ranking task using negative Spearman score."""
    return _task(
        "cross_sectional_ranking",
        target_column,
        "negative_spearman",
        feature_columns,
        regime_probability_columns,
        design,
    )


def factor_timing_task(
    *,
    target_column: str,
    feature_columns: Sequence[str],
    regime_probability_columns: Sequence[str],
    design: ComparisonDesign,
    prediction_to_position: Callable[[pd.Series | pd.DataFrame], pd.Series | pd.DataFrame]
    | None = None,
    realized_return_column: str | None = None,
) -> DownstreamTaskSpec:
    """Create a paired factor-timing task using squared-error forecast loss."""
    return _task(
        "factor_timing",
        target_column,
        "mse",
        feature_columns,
        regime_probability_columns,
        design,
        prediction_to_position=prediction_to_position,
        realized_return_column=realized_return_column or target_column,
    )


def evaluate_paired_task(
    data: pd.DataFrame,
    task: DownstreamTaskSpec,
    *,
    agnostic_estimator: ForecastEstimator,
    aware_estimator: ForecastEstimator,
) -> PairedDownstreamResult:
    """Fit and evaluate paired regime-agnostic and regime-aware downstream forecasts."""
    _validate_task_data(data, task)
    train, test = _split_data(data, task.design)
    x_cols = list(task.feature_columns)
    aware_cols = [*x_cols, *task.regime_probability_columns]

    agnostic_estimator.fit(train[x_cols], train[task.target_column])
    aware_estimator.fit(train[aware_cols], train[task.target_column])

    y = _delay(test[task.target_column], task.design.execution_delay)
    agnostic = _as_prediction(agnostic_estimator.predict(test[x_cols]), test.index, y.name)
    aware = _as_prediction(aware_estimator.predict(test[aware_cols]), test.index, y.name)
    agnostic = _delay(agnostic, task.design.execution_delay).loc[y.index]
    aware = _delay(aware, task.design.execution_delay).loc[y.index]

    agnostic_loss = _loss_series(y, agnostic, task.loss_metric)
    aware_loss = _loss_series(y, aware, task.loss_metric)
    test_result = diebold_mariano_test(aware_loss, agnostic_loss, alternative="less")
    statistical = StatisticalImprovement(
        agnostic_loss=float(agnostic_loss.mean()),
        aware_loss=float(aware_loss.mean()),
        absolute_improvement=float(agnostic_loss.mean() - aware_loss.mean()),
        relative_improvement=_relative_improvement(
            float(agnostic_loss.mean()), float(aware_loss.mean())
        ),
        test=test_result,
    )
    economic = _economic_improvement(test, task, agnostic, aware)
    return PairedDownstreamResult(task, agnostic, aware, statistical, economic)


def recommended_quick_start_tasks(
    *,
    feature_columns: Sequence[str],
    regime_probability_columns: Sequence[str],
    design: ComparisonDesign,
) -> tuple[DownstreamTaskSpec, ...]:
    """Return downstream tasks in the recommended implementation order."""
    return (
        equity_return_forecasting_task(
            target_column="forward_return",
            feature_columns=feature_columns,
            regime_probability_columns=regime_probability_columns,
            design=design,
        ),
        realized_volatility_forecasting_task(
            target_column="realized_volatility",
            feature_columns=feature_columns,
            regime_probability_columns=regime_probability_columns,
            design=design,
        ),
        tail_event_prediction_task(
            target_column="tail_event",
            feature_columns=feature_columns,
            regime_probability_columns=regime_probability_columns,
            design=design,
        ),
        factor_timing_task(
            target_column="factor_return",
            feature_columns=feature_columns,
            regime_probability_columns=regime_probability_columns,
            design=design,
        ),
        correlation_forecasting_task(
            target_column="realized_correlation",
            feature_columns=feature_columns,
            regime_probability_columns=regime_probability_columns,
            design=design,
        ),
        cross_sectional_ranking_task(
            target_column="forward_return",
            feature_columns=feature_columns,
            regime_probability_columns=regime_probability_columns,
            design=design,
        ),
    )


def _task(
    name: TaskKind,
    target_column: str,
    loss_metric: LossMetric,
    feature_columns: Sequence[str],
    regime_probability_columns: Sequence[str],
    design: ComparisonDesign,
    *,
    prediction_to_position: Callable[[pd.Series | pd.DataFrame], pd.Series | pd.DataFrame]
    | None = None,
    realized_return_column: str | None = None,
) -> DownstreamTaskSpec:
    return DownstreamTaskSpec(
        name,
        target_column,
        loss_metric,
        tuple(feature_columns),
        tuple(regime_probability_columns),
        design,
        prediction_to_position,
        realized_return_column,
    )


def _validate_task_data(data: pd.DataFrame, task: DownstreamTaskSpec) -> None:
    required = {task.target_column, *task.feature_columns, *task.regime_probability_columns}
    if task.realized_return_column is not None:
        required.add(task.realized_return_column)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if set(task.feature_columns) & set(task.regime_probability_columns):
        raise ValueError("base features and regime probability columns must be disjoint")


def _split_data(data: pd.DataFrame, design: ComparisonDesign) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.sort_index()
    train = frame.loc[pd.Timestamp(design.train_start) : pd.Timestamp(design.train_end)]
    test = frame.loc[pd.Timestamp(design.test_start) : pd.Timestamp(design.test_end)]
    if train.empty or test.empty:
        raise ValueError("train and test periods must both contain observations")
    return train, test


def _as_prediction(
    values: ArrayLike | FrameLike, index: pd.Index, name: str
) -> pd.Series | pd.DataFrame:
    arr: np.ndarray = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != len(index):
            raise ValueError("prediction length must match test period")
        return pd.Series(arr, index=index, name=name)
    if arr.ndim == 2:
        if arr.shape[0] != len(index):
            raise ValueError("prediction row count must match test period")
        return pd.DataFrame(arr, index=index)
    raise ValueError("predictions must be one- or two-dimensional")


def _delay(values: pd.Series | pd.DataFrame, delay: int) -> pd.Series | pd.DataFrame:
    if delay < 0:
        raise ValueError("execution_delay must be non-negative")
    return values.iloc[delay:] if delay else values


def _loss_series(
    y_true: pd.Series, y_pred: pd.Series | pd.DataFrame, metric: LossMetric
) -> pd.Series:
    aligned = pd.concat([y_true.rename("y"), y_pred], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("no aligned finite forecasts remain")
    y = aligned.iloc[:, 0]
    pred = aligned.iloc[:, 1:]
    if pred.shape[1] == 1:
        p = pred.iloc[:, 0]
        values = {
            "mse": (y - p) ** 2,
            "mae": (y - p).abs(),
            "rmse": pd.Series((rmse(y, p) ** 2 for _ in range(len(y))), index=y.index),
            "qlike": y.clip(lower=1e-12) / p.clip(lower=1e-12)
            - np.log(y.clip(lower=1e-12) / p.clip(lower=1e-12))
            - 1.0,
            "brier": (y - p) ** 2,
        }
        if metric in values:
            return values[metric].astype(float)
    if metric == "negative_spearman" and pred.shape[1] == 1:
        p = pred.iloc[:, 0]
        if isinstance(y.index, pd.MultiIndex):
            scores = []
            keys = []
            for key, actual_group in y.groupby(level=0):
                forecast_group = p.loc[actual_group.index]
                if len(actual_group) > 1:
                    scores.append(-float(actual_group.rank().corr(forecast_group.rank())))
                    keys.append(key)
            if not scores:
                raise ValueError(
                    "cross-sectional ranking loss requires at least one multi-asset date"
                )
            return pd.Series(scores, index=pd.Index(keys), dtype=float)
        return pd.Series([-float(y.rank().corr(p.rank()))], dtype=float)
    raise ValueError(f"metric {metric!r} is incompatible with prediction shape")


def _economic_improvement(
    test: pd.DataFrame,
    task: DownstreamTaskSpec,
    agnostic: pd.Series | pd.DataFrame,
    aware: pd.Series | pd.DataFrame,
) -> EconomicImprovement | None:
    if task.prediction_to_position is None or task.realized_return_column is None:
        return None
    realized = _delay(test[task.realized_return_column], task.design.execution_delay)
    agnostic_returns = _strategy_returns(
        realized, task.prediction_to_position(agnostic), task.design.cost_assumptions
    )
    aware_returns = _strategy_returns(
        realized, task.prediction_to_position(aware), task.design.cost_assumptions
    )
    agnostic_metrics = _economic_metrics(agnostic_returns, task.periods_per_year)
    aware_metrics = _economic_metrics(aware_returns, task.periods_per_year)
    return EconomicImprovement(
        agnostic_metrics=agnostic_metrics,
        aware_metrics=aware_metrics,
        metric_deltas={key: aware_metrics[key] - agnostic_metrics[key] for key in aware_metrics},
        cost_assumptions=task.design.cost_assumptions,
    )


def _strategy_returns(
    realized: pd.Series, positions: pd.Series | pd.DataFrame, costs: Mapping[str, float]
) -> pd.Series:
    pos = positions.iloc[:, 0] if isinstance(positions, pd.DataFrame) else positions
    aligned = pd.concat([realized.rename("r"), pos.rename("w")], axis=1, join="inner").dropna()
    turnover = aligned["w"].diff().abs().fillna(aligned["w"].abs())
    cost_rate = float(costs.get("proportional_cost", 0.0)) + float(costs.get("half_spread", 0.0))
    net = aligned["w"] * aligned["r"] - cost_rate * turnover
    net.attrs["turnover"] = float(turnover.mean())
    return net


def _economic_metrics(returns: pd.Series, periods_per_year: int) -> Mapping[str, float]:
    if returns.empty:
        raise ValueError("strategy returns are empty")
    vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    cumulative = (1.0 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    return {
        "annualized_return": float(returns.mean() * periods_per_year),
        "annualized_volatility": vol,
        "sharpe": float((returns.mean() * periods_per_year) / vol) if vol > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(returns.attrs.get("turnover", 0.0)),
    }


def _relative_improvement(agnostic_loss: float, aware_loss: float) -> float:
    return 0.0 if agnostic_loss == 0.0 else float((agnostic_loss - aware_loss) / abs(agnostic_loss))


__all__ = [
    "ComparisonDesign",
    "DownstreamTaskName",
    "DownstreamTaskSpec",
    "EconomicImprovement",
    "ForecastEstimator",
    "PairedDownstreamResult",
    "StatisticalImprovement",
    "correlation_forecasting_task",
    "cross_sectional_ranking_task",
    "equity_return_forecasting_task",
    "evaluate_paired_task",
    "factor_timing_task",
    "realized_volatility_forecasting_task",
    "recommended_quick_start_tasks",
    "tail_event_prediction_task",
]
