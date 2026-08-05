from __future__ import annotations

import numpy as np
import pandas as pd

from regime.evaluation.downstream.forecasting import (
    ComparisonDesign,
    equity_return_forecasting_task,
    evaluate_paired_task,
    recommended_quick_start_tasks,
)


class LinearEstimator:
    def __init__(self) -> None:
        self.coef: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series | pd.DataFrame) -> LinearEstimator:
        x = np.column_stack([np.ones(len(features)), features.to_numpy(dtype=float)])
        self.coef = np.linalg.pinv(x) @ np.asarray(target, dtype=float)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("estimator must be fit before predict")
        x = np.column_stack([np.ones(len(features)), features.to_numpy(dtype=float)])
        return x @ self.coef


def test_evaluate_paired_task_separates_statistical_and_economic_improvement() -> None:
    index = pd.date_range("2024-01-01", periods=30)
    signal = np.linspace(-1.0, 1.0, len(index))
    regime = np.r_[np.zeros(15), np.ones(15)]
    forward_return = 0.01 * signal + 0.02 * regime
    data = pd.DataFrame(
        {
            "signal": signal,
            "regime_0": 1.0 - regime,
            "regime_1": regime,
            "forward_return": forward_return,
        },
        index=index,
    )
    design = ComparisonDesign(
        train_start="2024-01-01",
        train_end="2024-01-20",
        test_start="2024-01-21",
        test_end="2024-01-30",
        retraining_cadence="once",
        execution_delay=0,
        cost_assumptions={"proportional_cost": 0.0},
    )
    task = equity_return_forecasting_task(
        target_column="forward_return",
        feature_columns=("signal",),
        regime_probability_columns=("regime_0", "regime_1"),
        design=design,
        prediction_to_position=lambda prediction: np.sign(prediction),
    )

    result = evaluate_paired_task(
        data, task, agnostic_estimator=LinearEstimator(), aware_estimator=LinearEstimator()
    )

    assert result.statistical.aware_loss < result.statistical.agnostic_loss
    assert result.statistical.absolute_improvement > 0.0
    assert result.economic is not None
    assert "sharpe" in result.economic.metric_deltas


def test_recommended_quick_start_tasks_are_ordered() -> None:
    design = ComparisonDesign(
        train_start="2024-01-01",
        train_end="2024-01-31",
        test_start="2024-02-01",
        test_end="2024-02-29",
        retraining_cadence="monthly",
    )

    tasks = recommended_quick_start_tasks(
        feature_columns=("market",), regime_probability_columns=("regime_0",), design=design
    )

    assert [task.name for task in tasks] == [
        "equity_return",
        "realized_volatility",
        "tail_event",
        "factor_timing",
        "correlation",
        "cross_sectional_ranking",
    ]
