"""Recurring-state jump segmentation models.

Jump segmentation assigns every observation to one of a fixed set of reusable
latent states.  Unlike non-recurring change-point segmentation, a state label may
appear in many disjoint time intervals (for example ``0, 0, 1, 1, 0``).  Boundary
or change-point detectors instead create chronological segment identifiers that
are not intended to recur.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from math import pi
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field, PositiveInt, model_validator
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from regime.models.base import ModelMetadata, RegimeModel, RegimeModelConfig

ArrayLike = pd.DataFrame | pd.Series | Sequence[Sequence[float]] | Sequence[float] | np.ndarray
Parameterization = Literal["centroid", "gaussian_diag"]


class JumpSegmentationConfig(RegimeModelConfig):
    """Configuration for recurring-state jump segmentation.

    ``jump_penalty`` is a global cost for changing state between adjacent
    observations. ``transition_penalty`` optionally adds state-pair-specific
    costs, either as a scalar off-diagonal penalty or a square matrix where entry
    ``[i, j]`` is the extra cost of moving from state ``i`` to state ``j``.
    """

    model_name: str = "jump_segmentation"
    n_states: PositiveInt = 2
    parameterization: Parameterization = "centroid"
    jump_penalty: float = Field(default=1.0, ge=0.0)
    transition_penalty: float | tuple[tuple[float, ...], ...] = Field(default=0.0)
    max_iter: PositiveInt = 25
    tol: float = Field(default=1e-4, ge=0.0)
    scale: bool = True
    feature_names: tuple[str, ...] | None = None
    covariance_regularization: float = Field(default=1e-6, gt=0.0)

    @model_validator(mode="after")
    def _validate_transition_penalty(self) -> Self:
        if isinstance(self.transition_penalty, tuple):
            if len(self.transition_penalty) != self.n_states:
                raise ValueError("transition_penalty matrix must have n_states rows")
            if any(len(row) != self.n_states for row in self.transition_penalty):
                raise ValueError("transition_penalty matrix must be square")
            if any(value < 0 for row in self.transition_penalty for value in row):
                raise ValueError("transition_penalty costs must be non-negative")
        elif self.transition_penalty < 0:
            raise ValueError("transition_penalty must be non-negative")
        return self


@dataclass(frozen=True)
class JumpSegmentationResult:
    """Fitted recurring-state segmentation output and summaries."""

    assignments: tuple[int, ...]
    objective: float
    centroids: Mapping[int, Mapping[str, float]]
    distribution_parameters: Mapping[int, Mapping[str, Mapping[str, float]]]
    occupancy: Mapping[int, float]
    transitions: Mapping[str, Any]
    state_statistics: Mapping[str, Mapping[str, float]]


def _as_frame(data: ArrayLike, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "value")
    else:
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        names = list(feature_names or [f"feature_{i}" for i in range(array.shape[1])])
        frame = pd.DataFrame(array, columns=names)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("data must contain at least one numeric feature")
    return numeric.ffill().bfill().fillna(0.0)


def _state_occupancy(assignments: np.ndarray, n_states: int) -> dict[int, float]:
    total = max(1, len(assignments))
    return {state: float(np.sum(assignments == state) / total) for state in range(n_states)}


def _transition_summary(assignments: np.ndarray, n_states: int) -> dict[str, Any]:
    counts = np.zeros((n_states, n_states), dtype=int)
    for left, right in pairwise(assignments):
        counts[int(left), int(right)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)
    return {"labels": tuple(range(n_states)), "counts": counts.tolist(), "matrix": matrix.tolist()}


def align_jump_labels(reference: Sequence[int], candidate: Sequence[int]) -> tuple[int, ...]:
    """Relabel candidate recurring states to maximize overlap with a reference path."""
    ref = np.asarray(reference, dtype=int)
    cand = np.asarray(candidate, dtype=int)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must have the same shape")
    ref_labels = sorted(int(x) for x in np.unique(ref))
    cand_labels = sorted(int(x) for x in np.unique(cand))
    cost = np.zeros((len(ref_labels), len(cand_labels)), dtype=int)
    for i, ref_label in enumerate(ref_labels):
        for j, cand_label in enumerate(cand_labels):
            cost[i, j] = -int(np.sum((ref == ref_label) & (cand == cand_label)))
    rows, cols = linear_sum_assignment(cost)
    mapping = {cand_labels[col]: ref_labels[row] for row, col in zip(rows, cols, strict=True)}
    next_label = max(ref_labels, default=-1) + 1
    for label in cand_labels:
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
    return tuple(int(mapping[int(label)]) for label in cand)


class JumpSegmentationModel(RegimeModel):
    """Dynamic-programming recurring-state segmentation with penalized jumps.

    The model alternates between estimating reusable state parameters and finding
    the lowest-cost state path.  It is a recurring-state model, not a change-point
    detector: the same state can be revisited after intervening regimes.
    """

    def __init__(self, config: JumpSegmentationConfig | None = None) -> None:
        self.config = config or JumpSegmentationConfig()
        self.scaler: StandardScaler | None = None
        self.feature_names: tuple[str, ...] = ()
        self.centers_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.result: JumpSegmentationResult | None = None
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            config_hash=self.configuration_hash(self.config),
            attributes={"segmentation_type": "recurring_state_jump"},
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def fit(self, dataset: ArrayLike, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        if config is not None:
            payload = config.model_dump() if hasattr(config, "model_dump") else config
            self.config = JumpSegmentationConfig.model_validate(payload)
        frame, values = self._prepare_fit(dataset)
        labels = KMeans(
            n_clusters=self.config.n_states,
            random_state=self.config.random_seed,
            n_init=10,
        ).fit_predict(values)
        previous = np.inf
        objective = np.inf
        for _ in range(self.config.max_iter):
            self._estimate_parameters(values, labels)
            costs = self._emission_cost(values)
            labels, objective = self._decode(costs)
            if abs(previous - objective) <= self.config.tol * max(1.0, abs(previous)):
                break
            previous = objective
        self._estimate_parameters(values, labels)
        self._finalize(frame, labels, objective)
        return self

    def predict(self, dataset: ArrayLike) -> tuple[int, ...]:
        _, values = self._prepare_predict(dataset)
        labels, _ = self._decode(self._emission_cost(values))
        return tuple(int(x) for x in labels)

    def predict_aligned(self, dataset: ArrayLike, reference: Sequence[int]) -> tuple[int, ...]:
        """Predict labels and align them to a reference recurring-state labeling."""
        return align_jump_labels(reference, self.predict(dataset))

    def align_to(
        self, reference: Sequence[int], candidate: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        """Align a candidate path, or the fitted path, to a reference path."""
        if candidate is None:
            if self.result is None:
                raise ValueError("model is not fitted")
            candidate = self.result.assignments
        return align_jump_labels(reference, candidate)

    def transition_matrix(self) -> Sequence[Sequence[float]]:
        if self.result is None:
            raise ValueError("model is not fitted")
        return self.result.transitions["matrix"]

    def state_statistics(self) -> Mapping[str, Mapping[str, float]]:
        if self.result is None:
            raise ValueError("model is not fitted")
        return self.result.state_statistics

    def transition_summary(self) -> Mapping[str, Any]:
        """Return counts and probabilities for one-step recurring-state transitions."""
        if self.result is None:
            raise ValueError("model is not fitted")
        return self.result.transitions

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"saved object is not a {cls.__name__}")
        return obj

    def _prepare_fit(self, data: ArrayLike) -> tuple[pd.DataFrame, np.ndarray]:
        frame = _as_frame(data, self.config.feature_names)
        self.feature_names = tuple(str(column) for column in frame.columns)
        values = frame.to_numpy(dtype=float)
        if self.config.scale:
            self.scaler = StandardScaler().fit(values)
            values = self.scaler.transform(values)
        return frame, values

    def _prepare_predict(self, data: ArrayLike) -> tuple[pd.DataFrame, np.ndarray]:
        if self.centers_ is None:
            raise ValueError("model is not fitted")
        frame = _as_frame(data, self.feature_names or self.config.feature_names)
        values = frame.to_numpy(dtype=float)
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return frame, values

    def _transition_cost(self) -> np.ndarray:
        k = self.config.n_states
        base = np.full((k, k), float(self.config.jump_penalty))
        np.fill_diagonal(base, 0.0)
        penalty = self.config.transition_penalty
        if isinstance(penalty, tuple):
            return base + np.asarray(penalty, dtype=float)
        extra = np.full((k, k), float(penalty))
        np.fill_diagonal(extra, 0.0)
        return base + extra

    def _estimate_parameters(self, values: np.ndarray, labels: np.ndarray) -> None:
        centers = np.zeros((self.config.n_states, values.shape[1]), dtype=float)
        variances = np.zeros_like(centers)
        for state in range(self.config.n_states):
            members = values[labels == state]
            if len(members) == 0:
                random_index = np.random.default_rng(self.config.random_seed).integers(len(values))
                centers[state] = values[random_index]
                variances[state] = np.var(values, axis=0) + self.config.covariance_regularization
            else:
                centers[state] = np.mean(members, axis=0)
                variances[state] = np.var(members, axis=0) + self.config.covariance_regularization
        self.centers_ = centers
        self.variances_ = variances

    def _emission_cost(self, values: np.ndarray) -> np.ndarray:
        if self.centers_ is None or self.variances_ is None:
            raise ValueError("model is not fitted")
        diff = values[:, None, :] - self.centers_[None, :, :]
        if self.config.parameterization == "centroid":
            return np.sum(diff**2, axis=2)
        scaled_distance = diff**2 / self.variances_[None, :, :]
        log_variance = np.log(2 * pi * self.variances_[None, :, :])
        return 0.5 * np.sum(scaled_distance + log_variance, axis=2)

    def _decode(self, emission_cost: np.ndarray) -> tuple[np.ndarray, float]:
        n, k = emission_cost.shape
        transition = self._transition_cost()
        dp = np.zeros((n, k), dtype=float)
        back = np.zeros((n, k), dtype=int)
        dp[0] = emission_cost[0]
        for t in range(1, n):
            scores = dp[t - 1][:, None] + transition
            back[t] = np.argmin(scores, axis=0)
            dp[t] = emission_cost[t] + scores[back[t], np.arange(k)]
        labels = np.zeros(n, dtype=int)
        labels[-1] = int(np.argmin(dp[-1]))
        for t in range(n - 2, -1, -1):
            labels[t] = back[t + 1, labels[t + 1]]
        return labels, float(np.min(dp[-1]))

    def _finalize(self, frame: pd.DataFrame, labels: np.ndarray, objective: float) -> None:
        original = frame.to_numpy(dtype=float)
        stats: dict[str, dict[str, float]] = {}
        centroids: dict[int, dict[str, float]] = {}
        distributions: dict[int, dict[str, dict[str, float]]] = {}
        for state in range(self.config.n_states):
            members = original[labels == state]
            if len(members) == 0:
                mean = np.full(original.shape[1], np.nan)
                std = np.full(original.shape[1], np.nan)
            else:
                mean = np.mean(members, axis=0)
                std = np.std(members, axis=0)
            centroids[state] = {
                name: float(value) for name, value in zip(self.feature_names, mean, strict=True)
            }
            distributions[state] = {
                name: {"mean": float(mean_i), "std": float(std_i)}
                for name, mean_i, std_i in zip(self.feature_names, mean, std, strict=True)
            }
            stats[str(state)] = {
                "count": float(len(members)),
                "occupancy": float(len(members) / max(1, len(labels))),
                **{
                    f"{name}_mean": float(value)
                    for name, value in zip(self.feature_names, mean, strict=True)
                },
                **{
                    f"{name}_std": float(value)
                    for name, value in zip(self.feature_names, std, strict=True)
                },
            }
        self.result = JumpSegmentationResult(
            assignments=tuple(int(x) for x in labels),
            objective=float(objective),
            centroids=centroids,
            distribution_parameters=distributions,
            occupancy=_state_occupancy(labels, self.config.n_states),
            transitions=_transition_summary(labels, self.config.n_states),
            state_statistics=stats,
        )
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            fitted_at=datetime.now(UTC),
            training_observations=len(frame),
            feature_names=self.feature_names,
            config_hash=self.configuration_hash(self.config),
            attributes={
                "segmentation_type": "recurring_state_jump",
                "not_change_point_segmentation": True,
                "parameterization": self.config.parameterization,
                "jump_penalty": self.config.jump_penalty,
            },
        )
