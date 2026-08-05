"""Downstream paired evaluation tasks."""

from regime.evaluation.downstream.forecasting import (
    ComparisonDesign,
    DownstreamTaskName,
    DownstreamTaskSpec,
    EconomicImprovement,
    ForecastEstimator,
    PairedDownstreamResult,
    StatisticalImprovement,
    correlation_forecasting_task,
    cross_sectional_ranking_task,
    equity_return_forecasting_task,
    evaluate_paired_task,
    factor_timing_task,
    realized_volatility_forecasting_task,
    recommended_quick_start_tasks,
    tail_event_prediction_task,
)

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
