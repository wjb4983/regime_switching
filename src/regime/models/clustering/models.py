"""Clustering models for recurring regime labels.

Cluster identifiers are arbitrary.  Use :func:`align_labels` before comparing
assignments from separate refits or from different model classes.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from pydantic import Field, PositiveInt, model_validator
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.covariance import GraphicalLasso
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from regime.models.base import ModelMetadata, RegimeModel, RegimeModelConfig

ArrayLike = pd.DataFrame | pd.Series | Sequence[Sequence[float]] | Sequence[float] | np.ndarray


class ClusteringConfig(RegimeModelConfig):
    """Configuration shared by clustering regime models."""

    model_name: str = "clustering_model"
    scale: bool = True
    feature_names: tuple[str, ...] | None = None
    covariance_type: str = "full"
    linkage: str = "ward"
    min_cluster_size: PositiveInt = 5
    smoothing_window: PositiveInt = 1
    jump_penalty: float = Field(default=0.0, ge=0.0)
    max_iter: PositiveInt = 100
    n_init: PositiveInt = 10
    window_size: PositiveInt = 5
    graphical_lasso_alpha: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.covariance_type not in {"full", "tied", "diag", "spherical"}:
            raise ValueError("covariance_type must be valid for sklearn GaussianMixture")
        if self.linkage not in {"ward", "complete", "average", "single"}:
            raise ValueError("linkage must be valid for AgglomerativeClustering")
        return self


@dataclass(frozen=True)
class ClusterModelResult:
    """Fitted clustering output with summaries derived from assignments."""

    assignments: tuple[int, ...]
    probabilities: tuple[tuple[float, ...], ...] | None
    centroids: Mapping[int, Mapping[str, float]]
    occupancy: Mapping[int, float]
    transitions: Mapping[str, Any]
    entropy: tuple[float, ...] | None


def _as_frame(data: ArrayLike, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "value")
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = list(feature_names or [f"feature_{i}" for i in range(arr.shape[1])])
        frame = pd.DataFrame(arr, columns=names)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("data must contain at least one numeric feature")
    return numeric.ffill().bfill().fillna(0.0)


def _entropy(probs: np.ndarray) -> np.ndarray:
    clean = np.clip(probs.astype(float), 1e-12, 1.0)
    return -np.sum(clean * np.log(clean), axis=1)


def assignment_entropy(probabilities: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return Shannon entropy for each probability row."""
    return tuple(float(x) for x in _entropy(np.asarray(probabilities, dtype=float)))


def state_occupancy(assignments: Sequence[int]) -> dict[int, float]:
    """Return the fraction of observations assigned to each state label."""
    labels, counts = np.unique(np.asarray(assignments, dtype=int), return_counts=True)
    total = max(1, int(np.sum(counts)))
    return {int(label): float(count / total) for label, count in zip(labels, counts, strict=True)}


def transition_summary(assignments: Sequence[int]) -> dict[str, Any]:
    """Summarize one-step transitions between arbitrary state labels."""
    arr = np.asarray(assignments, dtype=int)
    labels = sorted(int(x) for x in np.unique(arr))
    index = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for left, right in pairwise(arr):
        counts[index[int(left)], index[int(right)]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)
    return {"labels": tuple(labels), "counts": counts.tolist(), "matrix": probs.tolist()}


def state_centroid_summary(
    data: ArrayLike, assignments: Sequence[int]
) -> dict[int, dict[str, float]]:
    """Return per-state feature means in the original feature scale."""
    frame = _as_frame(data)
    arr = np.asarray(assignments, dtype=int)
    if len(frame) != len(arr):
        raise ValueError("assignments length must match data length")
    return {
        int(label): {
            str(col): float(value) for col, value in frame.loc[arr == label].mean().items()
        }
        for label in sorted(np.unique(arr))
    }


def smooth_assignments(assignments: Sequence[int], window: int = 3) -> tuple[int, ...]:
    """Apply centered majority-vote temporal smoothing to assignments."""
    arr = np.asarray(assignments, dtype=int)
    if window <= 1 or len(arr) == 0:
        return tuple(int(x) for x in arr)
    radius = window // 2
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i - radius), min(len(arr), i + radius + 1)
        vals, counts = np.unique(arr[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return tuple(int(x) for x in out)


def align_labels(reference: Sequence[int], candidate: Sequence[int]) -> tuple[int, ...]:
    """Relabel ``candidate`` to maximize overlap with ``reference`` labels."""
    ref = np.asarray(reference, dtype=int)
    cand = np.asarray(candidate, dtype=int)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must have the same shape")
    ref_labels = sorted(int(x) for x in np.unique(ref))
    cand_labels = sorted(int(x) for x in np.unique(cand))
    cost = np.zeros((len(ref_labels), len(cand_labels)), dtype=int)
    for i, r in enumerate(ref_labels):
        for j, c in enumerate(cand_labels):
            cost[i, j] = -int(np.sum((ref == r) & (cand == c)))
    rows, cols = linear_sum_assignment(cost)
    mapping = {cand_labels[c]: ref_labels[r] for r, c in zip(rows, cols, strict=True)}
    next_label = (max(ref_labels) + 1) if ref_labels else 0
    for label in cand_labels:
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
    return tuple(int(mapping[int(x)]) for x in cand)


class _BaseClusteringRegimeModel(RegimeModel):
    def __init__(self, config: ClusteringConfig | None = None) -> None:
        self.config = config or ClusteringConfig()
        self.scaler: StandardScaler | None = None
        self.estimator: Any | None = None
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            config_hash=self.configuration_hash(self.config),
        )
        self.result: ClusterModelResult | None = None
        self.feature_names: tuple[str, ...] = ()

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def _prepare_fit(self, data: ArrayLike) -> tuple[pd.DataFrame, np.ndarray]:
        frame = _as_frame(data, self.config.feature_names)
        self.feature_names = tuple(str(c) for c in frame.columns)
        values = frame.to_numpy(dtype=float)
        if self.config.scale:
            self.scaler = StandardScaler().fit(values)
            values = self.scaler.transform(values)
        return frame, values

    def _prepare_predict(self, data: ArrayLike) -> tuple[pd.DataFrame, np.ndarray]:
        frame = _as_frame(data, self.feature_names or self.config.feature_names)
        values = frame.to_numpy(dtype=float)
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return frame, values

    def _finalize(
        self,
        frame: pd.DataFrame,
        labels: Sequence[int],
        probabilities: np.ndarray | None = None,
    ) -> Self:
        labels_tuple = smooth_assignments(labels, self.config.smoothing_window)
        probs_tuple = (
            None
            if probabilities is None
            else tuple(tuple(float(x) for x in row) for row in probabilities)
        )
        self.result = ClusterModelResult(
            assignments=labels_tuple,
            probabilities=probs_tuple,
            centroids=state_centroid_summary(frame, labels_tuple),
            occupancy=state_occupancy(labels_tuple),
            transitions=transition_summary(labels_tuple),
            entropy=None if probabilities is None else assignment_entropy(probabilities),
        )
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            fitted_at=datetime.now(UTC),
            training_observations=len(frame),
            feature_names=self.feature_names,
            config_hash=self.configuration_hash(self.config),
        )
        return self

    def fit(self, dataset: ArrayLike, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        if config is not None:
            payload = config.model_dump() if hasattr(config, "model_dump") else config
            self.config = ClusteringConfig.model_validate(payload)
        frame, values = self._prepare_fit(dataset)
        labels, probs = self._fit_values(values)
        return self._finalize(frame, labels, probs)

    def predict(self, dataset: ArrayLike) -> tuple[int, ...]:
        _, values = self._prepare_predict(dataset)
        return tuple(int(x) for x in self._predict_values(values))

    def predict_proba(self, dataset: ArrayLike) -> tuple[tuple[float, ...], ...]:
        _, values = self._prepare_predict(dataset)
        probs = self._predict_proba_values(values)
        return tuple(tuple(float(x) for x in row) for row in probs)

    def state_statistics(self) -> Mapping[str, Mapping[str, float]]:
        if self.result is None:
            raise ValueError("model is not fitted")
        return {str(k): v for k, v in self.result.centroids.items()}

    def transition_matrix(self) -> Sequence[Sequence[float]]:
        if self.result is None:
            raise ValueError("model is not fitted")
        return self.result.transitions["matrix"]

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

    def _fit_values(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def _predict_values(self, values: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("model is not fitted")
        return self.estimator.predict(values)

    def _predict_proba_values(self, values: np.ndarray) -> np.ndarray:
        labels = self._predict_values(values)
        probs = np.zeros((len(labels), self.config.n_states))
        probs[np.arange(len(labels)), labels] = 1.0
        return probs


class KMeansRegimeModel(_BaseClusteringRegimeModel):
    """K-means recurring-regime clustering with training-window scaling."""

    def _fit_values(self, values: np.ndarray) -> tuple[np.ndarray, None]:
        self.estimator = KMeans(
            n_clusters=self.config.n_states,
            random_state=self.config.random_seed,
            n_init=self.config.n_init,
        )
        return self.estimator.fit_predict(values), None


class GaussianMixtureRegimeModel(_BaseClusteringRegimeModel):
    """Gaussian mixture model with posterior probabilities and entropy."""

    def _fit_values(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.estimator = GaussianMixture(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            random_state=self.config.random_seed,
        ).fit(values)
        return self.estimator.predict(values), self.estimator.predict_proba(values)

    def _predict_proba_values(self, values: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("model is not fitted")
        return self.estimator.predict_proba(values)


class HierarchicalClusteringRegimeModel(_BaseClusteringRegimeModel):
    """Agglomerative hierarchical clustering for fit-time segmentation."""

    def _fit_values(self, values: np.ndarray) -> tuple[np.ndarray, None]:
        self.estimator = AgglomerativeClustering(
            n_clusters=self.config.n_states, linkage=self.config.linkage
        )
        return self.estimator.fit_predict(values), None

    def _predict_values(self, values: np.ndarray) -> np.ndarray:
        if self.result is None or (self.scaler is None and not self.result.centroids):
            raise ValueError("model is not fitted")
        centroids = np.asarray(
            [
                [v[name] for name in self.feature_names]
                for _, v in sorted(self.result.centroids.items())
            ]
        )
        if self.scaler is not None:
            centroids = self.scaler.transform(centroids)
        distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.asarray(sorted(self.result.centroids), dtype=int)
        return labels[np.argmin(distances, axis=1)]


class HDBSCANRegimeModel(_BaseClusteringRegimeModel):
    """HDBSCAN clustering when the optional ``hdbscan`` package is installed."""

    def _fit_values(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        try:
            from hdbscan import HDBSCAN
        except ImportError as exc:
            raise ImportError(
                "Install regime-switching[clustering] to use HDBSCANRegimeModel"
            ) from exc
        self.estimator = HDBSCAN(
            min_cluster_size=self.config.min_cluster_size, prediction_data=True
        ).fit(values)
        labels = np.asarray(self.estimator.labels_, dtype=int)
        return labels, None


class JumpPenalizedKMeansRegimeModel(KMeansRegimeModel):
    """K-means followed by dynamic-programming smoothing that penalizes state jumps."""

    def _fit_values(self, values: np.ndarray) -> tuple[np.ndarray, None]:
        super()._fit_values(values)
        centers = np.asarray(self.estimator.cluster_centers_)
        cost = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        n, k = cost.shape
        dp = np.zeros((n, k))
        back = np.zeros((n, k), dtype=int)
        dp[0] = cost[0]
        for t in range(1, n):
            switch_cost = np.arange(k)[:, None] != np.arange(k)[None, :]
            penalties = dp[t - 1][:, None] + self.config.jump_penalty * switch_cost
            back[t] = np.argmin(penalties, axis=0)
            dp[t] = cost[t] + penalties[back[t], np.arange(k)]
        labels = np.zeros(n, dtype=int)
        labels[-1] = int(np.argmin(dp[-1]))
        for t in range(n - 2, -1, -1):
            labels[t] = back[t + 1, labels[t + 1]]
        return labels, None


class TICCRegimeModel(JumpPenalizedKMeansRegimeModel):
    """TICC-like covariance-structure segmentation using lagged windows.

    This equivalent lightweight implementation clusters flattened rolling windows,
    applies jump penalties, and estimates sparse inverse covariance matrices per state.
    """

    def _prepare_fit(self, data: ArrayLike) -> tuple[pd.DataFrame, np.ndarray]:
        frame, values = super()._prepare_fit(data)
        w = self.config.window_size
        if len(values) < w:
            raise ValueError("data length must be at least window_size")
        windows = np.asarray([values[i - w + 1 : i + 1].ravel() for i in range(w - 1, len(values))])
        self._ticc_prefix = w - 1
        return frame, windows

    def _finalize(
        self,
        frame: pd.DataFrame,
        labels: Sequence[int],
        probabilities: np.ndarray | None = None,
    ) -> Self:
        prefix = getattr(self, "_ticc_prefix", 0)
        padded = tuple([int(labels[0])] * prefix + [int(x) for x in labels])
        fitted = super()._finalize(frame, padded, probabilities)
        raw_values = frame.to_numpy(dtype=float)
        values = self.scaler.transform(raw_values) if self.scaler is not None else raw_values
        covariances: dict[int, list[list[float]]] = {}
        for label in sorted(set(padded)):
            segment = values[np.asarray(padded) == label]
            if len(segment) > values.shape[1]:
                covariances[int(label)] = (
                    GraphicalLasso(alpha=self.config.graphical_lasso_alpha)
                    .fit(segment)
                    .covariance_.tolist()
                )
        self._metadata = self._metadata.model_copy(
            update={"attributes": {"state_covariances": covariances}}
        )
        return fitted
