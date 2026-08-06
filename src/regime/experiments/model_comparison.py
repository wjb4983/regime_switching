"""Helpers for config-driven multi-model regime benchmarking."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)

from regime.evaluation.regime_quality import (
    boundary_precision_recall,
    probability_entropy,
    regime_persistence,
    state_entropy,
    state_occupancy,
    switching_frequency,
)
from regime.reporting.figures import (
    bar_chart,
    distribution_by_state,
    heatmap,
    probability_area_chart,
    regime_time_series,
    table,
)
from regime.reporting.report import ReportBuilder, VisualizationMetadata


@dataclass(frozen=True)
class MetricDefinition:
    family: str
    direction: str
    title: str
    explanation: str
    meaningful_when: str


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "aligned_accuracy": MetricDefinition(
        "recovery",
        "higher",
        "Aligned accuracy",
        "Accuracy after optimally aligning nominal predicted states to the synthetic truth.",
        "Meaningful when ground-truth states are available.",
    ),
    "balanced_accuracy": MetricDefinition(
        "recovery",
        "higher",
        "Balanced accuracy",
        "Average per-state recall after state alignment so rare regimes matter.",
        "Meaningful when class imbalance matters and ground truth is available.",
    ),
    "adjusted_rand_index": MetricDefinition(
        "recovery",
        "higher",
        "Adjusted Rand index",
        "Partition similarity between inferred and true states, adjusted for chance.",
        "Meaningful when the goal is recovering the latent partition, not exact labels.",
    ),
    "normalized_mutual_information": MetricDefinition(
        "recovery",
        "higher",
        "Normalized mutual information",
        "Information overlap between inferred and true state partitions.",
        "Meaningful when comparing clustering-like state assignments.",
    ),
    "macro_f1": MetricDefinition(
        "recovery",
        "higher",
        "Macro F1",
        "Average F1 across aligned states so each regime contributes equally.",
        "Meaningful when precision and recall both matter for each regime.",
    ),
    "boundary_precision": MetricDefinition(
        "recovery",
        "higher",
        "Boundary precision",
        "Share of predicted regime changes that match true change points.",
        "Meaningful when timing of regime transitions matters.",
    ),
    "boundary_recall": MetricDefinition(
        "recovery",
        "higher",
        "Boundary recall",
        "Share of true regime changes detected by the model.",
        "Meaningful when missed transitions are costly.",
    ),
    "boundary_f1": MetricDefinition(
        "recovery",
        "higher",
        "Boundary F1",
        "Harmonic mean of boundary precision and recall.",
        "Meaningful when both false alarms and misses matter.",
    ),
    "recovery_score": MetricDefinition(
        "recovery",
        "higher",
        "Recovery score",
        "Composite mean of available truth-based recovery metrics for synthetic benchmarks.",
        "Meaningful only for experiments with known true states.",
    ),
    "regime_persistence": MetricDefinition(
        "regime_quality",
        "higher",
        "Regime persistence",
        "Fraction of adjacent observations that remain in the same inferred regime.",
        "Meaningful as a stability diagnostic, not as a standalone ranking target.",
    ),
    "switching_frequency": MetricDefinition(
        "regime_quality",
        "lower",
        "Switching frequency",
        "Fraction of adjacent observations that change inferred regime.",
        "Meaningful as a turnover diagnostic.",
    ),
    "occupancy_concentration": MetricDefinition(
        "regime_quality",
        "lower",
        "Occupancy concentration",
        "Herfindahl concentration of state occupancy; lower means less single-state collapse.",
        "Meaningful for spotting degenerate models that stay in one state.",
    ),
    "state_entropy": MetricDefinition(
        "regime_quality",
        "higher",
        "State entropy",
        "Entropy of empirical state occupancy across the sample.",
        "Meaningful for judging regime diversity, with caution on very noisy models.",
    ),
    "probability_entropy": MetricDefinition(
        "regime_quality",
        "lower",
        "Probability entropy",
        "Average uncertainty of filtered regime probabilities.",
        "Meaningful only for models that expose probabilities.",
    ),
    "transition_concentration": MetricDefinition(
        "regime_quality",
        "higher",
        "Transition concentration",
        "Average maximum one-step transition probability by current state.",
        "Meaningful for understanding how decisive inferred transitions are.",
    ),
    "confidence_calibration": MetricDefinition(
        "regime_quality",
        "lower",
        "Confidence calibration",
        "Expected calibration error of top-state confidence against correctness.",
        "Meaningful only when ground truth and probabilities are both available.",
    ),
    "fit_seconds": MetricDefinition(
        "runtime",
        "lower",
        "Fit seconds",
        "Observed wall-clock training time from the training diagnostics artifact.",
        "Meaningful for operational cost and benchmark practicality.",
    ),
    "prediction_coverage": MetricDefinition(
        "runtime",
        "higher",
        "Prediction coverage",
        "Share of rows for which the model produced a usable state prediction.",
        "Meaningful for catching partial-output failures.",
    ),
}


def discover_model_configs(root: Path, *, relative_to: Path | None = None) -> list[dict[str, Any]]:
    """Return a stable list of config-backed model entries under ``configs/models``."""
    entries: list[dict[str, Any]] = []
    base = relative_to or root.parent
    for path in sorted(root.rglob("*.yaml")):
        relative = Path(os.path.relpath(path, start=base)).as_posix()
        family = path.parent.name if path.parent != root else "core"
        label = path.stem.replace("_", "-")
        entries.append(
            {
                "config": relative,
                "label": label,
                "family": family,
            }
        )
    return entries


def flatten_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model_metrics in metrics.items():
        for metric, value in model_metrics.items():
            definition = METRIC_DEFINITIONS.get(metric)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "family": definition.family if definition else "other",
                    "direction": definition.direction if definition else "diagnostic",
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def aligned_state_mapping(truth: Sequence[int], predicted: Sequence[int]) -> dict[int, int]:
    truth_values = np.unique(np.asarray(truth, dtype=int))
    pred_values = np.unique(np.asarray(predicted, dtype=int))
    confusion = np.zeros((truth_values.size, pred_values.size), dtype=float)
    y_true = np.asarray(truth, dtype=int)
    y_pred = np.asarray(predicted, dtype=int)
    for i, truth_label in enumerate(truth_values):
        for j, pred_label in enumerate(pred_values):
            confusion[i, j] = np.sum((y_true == truth_label) & (y_pred == pred_label))
    rows, cols = linear_sum_assignment(-confusion)
    mapping = {int(pred_values[col]): int(truth_values[row]) for row, col in zip(rows, cols, strict=True)}
    for value in pred_values:
        mapping.setdefault(int(value), int(value))
    return mapping


def _boundaries(states: Sequence[int]) -> list[int]:
    array = np.asarray(states, dtype=int)
    if array.size <= 1:
        return []
    return (np.flatnonzero(array[1:] != array[:-1]) + 1).astype(int).tolist()


def _hhi(values: Mapping[int, float]) -> float:
    shares = np.asarray(list(values.values()), dtype=float)
    if shares.size == 0:
        return float("nan")
    return float(np.sum(shares**2))


def _transition_concentration(states: Sequence[int]) -> float:
    labels = np.asarray(states, dtype=int)
    if labels.size <= 1:
        return 1.0
    transition = pd.crosstab(pd.Series(labels[:-1]), pd.Series(labels[1:]), normalize="index")
    if transition.empty:
        return float("nan")
    return float(transition.max(axis=1).mean())


def _top_label_calibration_error(
    truth: Sequence[int], predicted: Sequence[int], confidence: Sequence[float], *, n_bins: int = 10
) -> float:
    correct = (np.asarray(truth, dtype=int) == np.asarray(predicted, dtype=int)).astype(float)
    probs = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps: list[float] = []
    weights: list[float] = []
    for index in range(n_bins):
        upper = probs <= bins[index + 1] if index == n_bins - 1 else probs < bins[index + 1]
        mask = (probs >= bins[index]) & upper
        if not np.any(mask):
            continue
        gaps.append(float(abs(correct[mask].mean() - probs[mask].mean())))
        weights.append(float(mask.mean()))
    if not gaps:
        return float("nan")
    return float(np.average(gaps, weights=weights))


def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    model_name: str,
    fit_seconds: float | None,
    metrics: Sequence[str],
) -> dict[str, Any]:
    """Evaluate one standardized prediction frame."""
    supported: dict[str, float] = {}
    unsupported: dict[str, str] = {}
    if "state" not in frame:
        return {
            "model": model_name,
            "status": "failed",
            "metrics": supported,
            "unsupported_metrics": {metric: "missing state predictions" for metric in metrics},
        }
    work = frame.copy()
    work = work.dropna(subset=["state"]).reset_index(drop=True)
    states = work["state"].astype(int).to_numpy()
    prediction_coverage = float(len(work) / max(len(frame), 1))
    supported["prediction_coverage"] = prediction_coverage
    if fit_seconds is not None:
        supported["fit_seconds"] = float(fit_seconds)
    if len(work) > 0:
        occupancy = state_occupancy(states)
        supported["regime_persistence"] = float(regime_persistence(states))
        supported["switching_frequency"] = float(switching_frequency(states))
        supported["occupancy_concentration"] = float(_hhi(occupancy))
        supported["state_entropy"] = float(state_entropy(states))
        supported["transition_concentration"] = float(_transition_concentration(states))
    probability_columns = [column for column in work.columns if str(column).startswith("prob_")]
    if probability_columns:
        probabilities = work[probability_columns].to_numpy(dtype=float)
        supported["probability_entropy"] = float(probability_entropy(probabilities))
    else:
        unsupported["probability_entropy"] = "model did not expose filtered probabilities"
    if "true_state" in work.columns and work["true_state"].notna().any():
        truth = work["true_state"].astype(int).to_numpy()
        mapping = aligned_state_mapping(truth, states)
        aligned = np.asarray([mapping[int(value)] for value in states], dtype=int)
        supported["aligned_accuracy"] = float(np.mean(aligned == truth))
        unique_truth = np.unique(truth)
        unique_aligned = np.unique(aligned)
        if unique_truth.size < 2 or unique_aligned.size < 2:
            supported["balanced_accuracy"] = supported["aligned_accuracy"]
        else:
            supported["balanced_accuracy"] = float(balanced_accuracy_score(truth, aligned))
        supported["adjusted_rand_index"] = float(adjusted_rand_score(truth, states))
        supported["normalized_mutual_information"] = float(
            normalized_mutual_info_score(truth, states)
        )
        supported["macro_f1"] = float(f1_score(truth, aligned, average="macro"))
        precision, recall = boundary_precision_recall(_boundaries(truth), _boundaries(aligned))
        supported["boundary_precision"] = float(precision)
        supported["boundary_recall"] = float(recall)
        supported["boundary_f1"] = float(
            0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        )
        recovery_parts = [
            supported[name]
            for name in (
                "aligned_accuracy",
                "balanced_accuracy",
                "macro_f1",
                "normalized_mutual_information",
                "boundary_f1",
            )
        ]
        recovery_parts.append((supported["adjusted_rand_index"] + 1.0) / 2.0)
        supported["recovery_score"] = float(np.mean(recovery_parts))
        if probability_columns:
            confidence = work[probability_columns].max(axis=1).to_numpy(dtype=float)
            supported["confidence_calibration"] = float(
                _top_label_calibration_error(truth, aligned, confidence)
            )
    else:
        for metric in (
            "aligned_accuracy",
            "balanced_accuracy",
            "adjusted_rand_index",
            "normalized_mutual_information",
            "macro_f1",
            "boundary_precision",
            "boundary_recall",
            "boundary_f1",
            "recovery_score",
            "confidence_calibration",
        ):
            unsupported[metric] = "ground-truth states are unavailable"
    for metric in metrics:
        if metric not in supported and metric not in unsupported:
            unsupported[metric] = "metric not computable from available artifacts"
    return {
        "model": model_name,
        "status": "completed",
        "metrics": supported,
        "unsupported_metrics": unsupported,
        "rows": int(len(work)),
        "probability_columns": probability_columns,
    }


def rank_models(evaluation_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create comparison tables from the evaluation payload."""
    completed = [
        model for model in evaluation_payload.get("models", []) if model.get("status") == "completed"
    ]
    failures = [
        model for model in evaluation_payload.get("models", []) if model.get("status") != "completed"
    ]

    def table_for(metric_names: Sequence[str], *, sort_by: str | None = None, ascending: bool = False) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for model in completed:
            row = {"model": model["model"]}
            row.update({name: model.get("metrics", {}).get(name) for name in metric_names})
            rows.append(row)
        if sort_by is not None:
            rows.sort(
                key=lambda item: (-np.inf if item.get(sort_by) is None and not ascending else np.inf)
                if item.get(sort_by) is None
                else float(item.get(sort_by)),
                reverse=not ascending,
            )
        return rows

    recovery_metric = "recovery_score" if any(
        "recovery_score" in model.get("metrics", {}) for model in completed
    ) else None
    recovery = table_for(
        [
            "recovery_score",
            "aligned_accuracy",
            "balanced_accuracy",
            "adjusted_rand_index",
            "normalized_mutual_information",
            "macro_f1",
            "boundary_f1",
        ],
        sort_by=recovery_metric,
    )
    unsupervised = table_for(
        [
            "regime_persistence",
            "switching_frequency",
            "occupancy_concentration",
            "state_entropy",
            "probability_entropy",
            "transition_concentration",
            "confidence_calibration",
        ]
    )
    runtime = table_for(["fit_seconds", "prediction_coverage"], sort_by="fit_seconds", ascending=True)
    return {
        "ranking": {
            "primary_metric": recovery_metric or "fit_seconds",
            "recovery": recovery,
            "regime_quality": unsupervised,
            "runtime": runtime,
        },
        "failures": [
            {
                "model": item.get("model"),
                "status": item.get("status"),
                "reason": item.get("reason") or item.get("unsupported_metrics") or "unknown failure",
            }
            for item in failures
        ],
    }


def render_comparison_report(
    comparison_payload: Mapping[str, Any],
    *,
    title: str,
    output: Path,
) -> Path:
    """Render the HTML comparison report."""
    builder = ReportBuilder(title, subtitle="Config-driven synthetic model benchmark")
    config_hash_input = json.dumps(comparison_payload.get("config", {}), sort_keys=True, default=str)
    period = str(comparison_payload.get("data_period", "Synthetic benchmark"))
    metadata = VisualizationMetadata.from_config(
        research_question="Which regime model best recovers the synthetic regimes and remains usable in practice?",
        interpretation="Compare truth-based recovery separately from unsupervised regime diagnostics and runtime cost.",
        data_period=period,
        model_version="comparison",
        config={"comparison": comparison_payload.get("config", {}), "models": comparison_payload.get("model_names", [])},
        probability_kind="filtered",
    )
    definitions = []
    for name, definition in METRIC_DEFINITIONS.items():
        definitions.append(
            {
                "metric": name,
                "family": definition.family,
                "direction": definition.direction,
                "what it means": definition.explanation,
                "meaningful when": definition.meaningful_when,
            }
        )
    builder.add(
        table(
            pd.DataFrame(definitions).set_index("metric"),
            metadata,
            title="Metric guide",
        )
    )
    ranking = comparison_payload.get("ranking", {})
    for name, title_text in (
        ("recovery", "Recovery ranking"),
        ("regime_quality", "Regime-quality comparison"),
        ("runtime", "Runtime and coverage"),
    ):
        rows = ranking.get(name, [])
        if not rows:
            continue
        frame = pd.DataFrame(rows).set_index("model")
        builder.add(table(frame, metadata, title=title_text))
        numeric = frame.select_dtypes(include=["number"]).dropna(axis=1, how="all")
        if not numeric.empty:
            builder.add(bar_chart(numeric, metadata, title=f"{title_text} chart"))
    failures = comparison_payload.get("failures", [])
    if failures:
        builder.add(
            table(
                pd.DataFrame(failures).set_index("model"),
                metadata,
                title="Failures and unsupported models",
            )
        )
    for model in comparison_payload.get("models", []):
        frame_path = model.get("evaluation_frame")
        if not frame_path:
            continue
        path = Path(str(frame_path))
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        if "timestamp" in frame:
            frame = frame.copy()
            frame.index = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        model_meta = VisualizationMetadata.from_config(
            research_question="How does this model segment the synthetic series over time?",
            interpretation="Timelines and state diagnostics explain why each model scored the way it did.",
            data_period=period,
            model_version=str(model.get("model")),
            config={"model": model.get("model"), "family": model.get("family")},
            probability_kind="filtered",
        )
        if {"close", "state"} <= set(frame.columns):
            builder.add(
                regime_time_series(
                    frame["close"],
                    frame["state"],
                    model_meta,
                    title=f"Regime timeline ({model['model']})",
                )
            )
        elif {"return_1d", "state"} <= set(frame.columns):
            builder.add(
                regime_time_series(
                    frame["return_1d"],
                    frame["state"],
                    model_meta,
                    title=f"Regime timeline ({model['model']})",
                )
            )
        if "state" in frame.columns:
            occupancy = pd.Series(state_occupancy(frame["state"].dropna().astype(int)), name="share")
            builder.add(
                bar_chart(
                    occupancy.sort_index(),
                    model_meta,
                    title=f"State occupancy ({model['model']})",
                    y_label="Share",
                )
            )
            groups = frame["state"].ne(frame["state"].shift()).cumsum()
            durations = frame.groupby(groups, dropna=False)["state"].agg(["first", "size"]).dropna()
            if not durations.empty:
                duration_sizes = durations["size"].reset_index(drop=True)
                duration_states = durations["first"].reset_index(drop=True)
                builder.add(
                    distribution_by_state(
                        duration_sizes,
                        duration_states,
                        model_meta,
                        title=f"State duration distribution ({model['model']})",
                        x_label="Observations",
                    )
                )
            transition = pd.crosstab(frame["state"].shift(), frame["state"], normalize="index")
            if not transition.empty:
                builder.add(heatmap(transition, model_meta, title=f"Transition heatmap ({model['model']})"))
        probability_columns = [column for column in frame.columns if str(column).startswith("prob_")]
        if probability_columns:
            builder.add(
                probability_area_chart(
                    frame[probability_columns],
                    model_meta,
                    title=f"Filtered probabilities ({model['model']})",
                )
            )
    return builder.write(output)
