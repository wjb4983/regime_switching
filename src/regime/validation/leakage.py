"""Point-in-time and training-window guards for leakage-sensitive workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import pandas as pd

ProbabilityType = Literal["filtered", "smoothed", "classifier_live", "retrospective"]
EstimatorT = TypeVar("EstimatorT")


def point_in_time_snapshot(
    data: pd.DataFrame,
    decision_time: object,
    *,
    key_columns: Sequence[str],
    publication_column: str = "publication_ts",
    vendor_received_column: str = "vendor_received_ts",
) -> pd.DataFrame:
    """Return the latest revision of rows actually known by ``decision_time``.

    Publication time is mandatory.  Vendor receipt time is an additional availability
    constraint when the column/value is present.  Selecting a revision happens *after*
    these constraints are applied, so a later restatement cannot replace the vintage
    visible at the decision time.
    """
    missing = [column for column in (*key_columns, publication_column) if column not in data]
    if missing:
        raise ValueError(f"point-in-time data missing columns: {missing}")

    cutoff = pd.Timestamp(decision_time)
    frame = data.copy()
    publication = pd.to_datetime(frame[publication_column], utc=cutoff.tz is not None)
    available = publication.notna() & publication.le(cutoff)
    order_columns = [publication_column]
    if vendor_received_column in frame:
        received = pd.to_datetime(frame[vendor_received_column], utc=cutoff.tz is not None)
        available &= received.isna() | received.le(cutoff)
        order_columns.append(vendor_received_column)

    visible = frame.loc[available].sort_values(order_columns, kind="stable")
    return visible.drop_duplicates(list(key_columns), keep="last").reset_index(drop=True)


def fit_on_training_window(
    estimator: EstimatorT,
    features: Any,
    train_indices: Sequence[int],
    labels: Any | None = None,
) -> EstimatorT:
    """Fit an estimator exclusively on positional rows in a declared training window."""
    if not train_indices:
        raise ValueError("train_indices must not be empty")
    training_features = _take_rows(features, train_indices)
    if labels is None:
        estimator.fit(training_features)  # type: ignore[attr-defined]
    else:
        estimator.fit(training_features, _take_rows(labels, train_indices))  # type: ignore[attr-defined]
    return estimator


def require_adjustment_status(data: pd.DataFrame, expected: str) -> None:
    """Reject price data whose declared corporate-action adjustment is inconsistent."""
    if "adjustment_status" not in data:
        raise ValueError("price data must declare adjustment_status")
    actual = set(data["adjustment_status"].dropna().astype(str))
    if actual != {expected}:
        raise ValueError(f"expected adjustment_status={expected!r}, found {sorted(actual)!r}")


def validate_probability_usage(
    probability_type: ProbabilityType,
    *,
    live_equivalent: bool = True,
    allow_hindsight: bool = False,
) -> None:
    """Block hindsight-only probability products from live-equivalent evaluation."""
    hindsight_only = probability_type in {"smoothed", "retrospective"}
    if live_equivalent and hindsight_only and not allow_hindsight:
        raise ValueError(
            f"{probability_type} probabilities are hindsight-only and cannot be used in "
            "a live-equivalent backtest"
        )


def _take_rows(values: Any, indices: Sequence[int]) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[list(indices)]
    try:
        return values[list(indices)]
    except (TypeError, IndexError):
        return [values[index] for index in indices]
