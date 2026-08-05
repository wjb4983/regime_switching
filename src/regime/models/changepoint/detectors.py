"""Change-point boundary/event detectors.

The APIs in this module intentionally describe boundary events instead of
recurring-state classification.  A detected point means "a new chronological
segment starts here"; repeated segment identifiers are never reused as regime
labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp, lgamma, log, pi, sqrt
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pydantic import Field, PositiveInt, model_validator
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

from regime.config.base import RegimeBaseConfig
from regime.models.base import UnsupportedModelOperation


@dataclass(frozen=True)
class ChangePointDetectionResult:
    """Output from a boundary/event detector.

    ``segment_ids`` are chronological identifiers induced by boundaries. They are
    not recurring latent classes and should not be interpreted as risk-on/risk-off
    or hidden Markov states.
    """

    scores: tuple[float, ...]
    change_probabilities: tuple[float, ...]
    boundary_timestamps: tuple[Any, ...]
    boundary_indices: tuple[int, ...]
    segment_ids: tuple[int, ...]
    detection_delays: tuple[int, ...] | None = None
    false_alarm_rate: float | None = None
    metadata: dict[str, Any] | None = None


class BoundaryEventDetector(Protocol):
    """Protocol for detectors that emit boundary events, not state classes."""

    def detect(
        self,
        data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
        *,
        ground_truth: Sequence[Any] | None = None,
    ) -> ChangePointDetectionResult: ...


class BaseChangePointConfig(RegimeBaseConfig):
    """Shared configuration for change-point boundary detectors."""

    feature: str | None = None
    min_size: PositiveInt = 5
    threshold: float | None = None
    tolerance: PositiveInt = 5


class CUSUMConfig(BaseChangePointConfig):
    """Configuration for two-sided online CUSUM boundary detection."""

    drift: float = Field(default=0.0, ge=0.0)
    reference_mean: float | None = None
    reference_std: float | None = Field(default=None, gt=0.0)
    threshold: float = Field(default=5.0, gt=0.0)


class PageHinkleyConfig(BaseChangePointConfig):
    """Configuration for online Page-Hinkley mean-shift boundary detection."""

    delta: float = Field(default=0.005, ge=0.0)
    alpha: float = Field(default=0.999, gt=0.0, le=1.0)
    threshold: float = Field(default=50.0, gt=0.0)


class BinarySegmentationConfig(BaseChangePointConfig):
    """Configuration for offline binary segmentation."""

    max_breakpoints: PositiveInt = 5
    threshold: float = Field(default=1.0, gt=0.0)
    use_ruptures: bool = False
    model: str = "l2"


class PELTConfig(BaseChangePointConfig):
    """Configuration for offline PELT change-point detection."""

    penalty: float = Field(default=3.0, gt=0.0)
    use_ruptures: bool = False
    model: str = "l2"


class DistributionalChangePointConfig(BaseChangePointConfig):
    """Configuration for kernel/distributional two-sample scanning."""

    window: PositiveInt = 20
    threshold: float = Field(default=0.75, gt=0.0)
    method: str = "kernel_mmd"
    gamma: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _validate_method(self) -> DistributionalChangePointConfig:
        if self.method not in {"kernel_mmd", "ks"}:
            raise ValueError("method must be 'kernel_mmd' or 'ks'")
        return self


class BayesianOnlineChangePointConfig(BaseChangePointConfig):
    """Configuration for Bayesian online change-point detection (BOCPD)."""

    hazard: float = Field(default=0.01, gt=0.0, lt=1.0)
    threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    prior_mean: float = 0.0
    prior_kappa: float = Field(default=1.0, gt=0.0)
    prior_alpha: float = Field(default=1.0, gt=0.0)
    prior_beta: float = Field(default=1.0, gt=0.0)


def _as_frame(data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name or "value")
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        return pd.DataFrame({"value": array})
    return pd.DataFrame(array)


def _values_and_index(
    data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray, feature: str | None
) -> tuple[np.ndarray, pd.Index]:
    frame = _as_frame(data)
    if feature is not None:
        if feature not in frame:
            raise ValueError(f"feature {feature!r} not found in data")
        selected = frame[[feature]]
    else:
        selected = frame.select_dtypes(include=[np.number])
    if selected.empty:
        raise ValueError("data must contain at least one numeric feature")
    values = selected.to_numpy(dtype=float)
    if np.isnan(values).any():
        values = pd.DataFrame(values).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    return values, frame.index


def _segment_ids(n: int, boundaries: Iterable[int]) -> tuple[int, ...]:
    boundary_set = set(boundaries)
    current = 0
    ids = []
    for i in range(n):
        if i in boundary_set:
            current += 1
        ids.append(current)
    return tuple(ids)


def _metrics(
    boundaries: Sequence[int], truth: Sequence[Any] | None, n: int, tolerance: int
) -> tuple[tuple[int, ...] | None, float | None]:
    if truth is None:
        return None, None
    truth_indices = sorted(int(x) for x in truth)
    unmatched = set(truth_indices)
    delays: list[int] = []
    false_alarms = 0
    for boundary in boundaries:
        candidates = [t for t in unmatched if abs(boundary - t) <= tolerance]
        if candidates:
            matched = min(candidates, key=lambda t: abs(boundary - t))
            unmatched.remove(matched)
            delays.append(boundary - matched)
        else:
            false_alarms += 1
    return tuple(delays), false_alarms / max(1, n)


def _finalize(
    scores: np.ndarray,
    boundaries: Sequence[int],
    index: pd.Index,
    ground_truth: Sequence[Any] | None,
    tolerance: int,
    metadata: dict[str, Any],
) -> ChangePointDetectionResult:
    n = len(scores)
    clean = np.nan_to_num(scores.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    max_score = float(np.max(clean)) if clean.size else 0.0
    probs = clean / max_score if max_score > 0 else clean
    unique_boundaries = tuple(sorted({b for b in boundaries if 0 < b < n}))
    delays, false_alarm_rate = _metrics(unique_boundaries, ground_truth, n, tolerance)
    return ChangePointDetectionResult(
        scores=tuple(float(x) for x in clean),
        change_probabilities=tuple(float(min(1.0, max(0.0, x))) for x in probs),
        boundary_timestamps=tuple(index[b] for b in unique_boundaries),
        boundary_indices=unique_boundaries,
        segment_ids=_segment_ids(n, unique_boundaries),
        detection_delays=delays,
        false_alarm_rate=false_alarm_rate,
        metadata=metadata,
    )


class CUSUMDetector:
    """Two-sided CUSUM detector for online boundary events."""

    def __init__(self, config: CUSUMConfig | None = None) -> None:
        self.config = config or CUSUMConfig()

    def detect(
        self,
        data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
        *,
        ground_truth: Sequence[Any] | None = None,
    ) -> ChangePointDetectionResult:
        values, index = _values_and_index(data, self.config.feature)
        x = values[:, 0]
        baseline = x[: max(1, min(len(x), self.config.min_size))]
        mean = (
            float(np.mean(baseline))
            if self.config.reference_mean is None
            else self.config.reference_mean
        )
        std = (
            float(np.std(baseline, ddof=1) or 1.0)
            if self.config.reference_std is None
            else self.config.reference_std
        )
        pos = neg = 0.0
        scores = np.zeros(len(x))
        boundaries: list[int] = []
        for i, value in enumerate(x):
            z = (value - mean) / std
            pos = max(0.0, pos + z - self.config.drift)
            neg = min(0.0, neg + z + self.config.drift)
            score = max(pos, -neg)
            scores[i] = score
            if i >= self.config.min_size and score >= self.config.threshold:
                boundaries.append(i)
                pos = neg = 0.0
                mean = float(value)
        return _finalize(
            scores,
            boundaries,
            index,
            ground_truth,
            self.config.tolerance,
            {"detector": "cusum", "semantics": "boundary_event"},
        )


class PageHinkleyDetector:
    """Page-Hinkley detector for online mean-shift boundary events."""

    def __init__(self, config: PageHinkleyConfig | None = None) -> None:
        self.config = config or PageHinkleyConfig()

    def detect(
        self,
        data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
        *,
        ground_truth: Sequence[Any] | None = None,
    ) -> ChangePointDetectionResult:
        values, index = _values_and_index(data, self.config.feature)
        x = values[:, 0]
        mean = 0.0
        cumulative = minimum = 0.0
        scores = np.zeros(len(x))
        boundaries: list[int] = []
        for i, value in enumerate(x, start=1):
            mean += (value - mean) / i
            cumulative = self.config.alpha * cumulative + value - mean - self.config.delta
            minimum = min(minimum, cumulative)
            score = cumulative - minimum
            scores[i - 1] = score
            if i > self.config.min_size and score >= self.config.threshold:
                boundaries.append(i - 1)
                cumulative = minimum = 0.0
                mean = float(value)
        return _finalize(
            scores,
            boundaries,
            index,
            ground_truth,
            self.config.tolerance,
            {"detector": "page_hinkley", "semantics": "boundary_event"},
        )


def _mean_cost(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    centered = values - np.mean(values, axis=0)
    return float(np.sum(centered * centered))


def _split_score(values: np.ndarray, split: int) -> float:
    return _mean_cost(values) - _mean_cost(values[:split]) - _mean_cost(values[split:])


class BinarySegmentationDetector:
    """Offline binary segmentation boundary detector."""

    def __init__(self, config: BinarySegmentationConfig | None = None) -> None:
        self.config = config or BinarySegmentationConfig()

    def detect(
        self,
        data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
        *,
        ground_truth: Sequence[Any] | None = None,
    ) -> ChangePointDetectionResult:
        values, index = _values_and_index(data, self.config.feature)
        if self.config.use_ruptures:
            return RupturesAdapter(
                "Binseg", self.config.model, min_size=self.config.min_size
            ).detect(
                values,
                index,
                ground_truth=ground_truth,
                tolerance=self.config.tolerance,
                n_bkps=self.config.max_breakpoints,
            )
        scores = np.zeros(len(values))
        boundaries: list[int] = []
        segments = [(0, len(values))]
        for _ in range(self.config.max_breakpoints):
            best = (0.0, None, None)
            for start, end in segments:
                if end - start < 2 * self.config.min_size:
                    continue
                for split in range(start + self.config.min_size, end - self.config.min_size + 1):
                    score = _split_score(values[start:end], split - start)
                    scores[split] = max(scores[split], score)
                    if score > best[0]:
                        best = (score, split, (start, end))
            if best[1] is None or best[0] < self.config.threshold:
                break
            boundaries.append(best[1])
            segments.remove(best[2])  # type: ignore[arg-type]
            segments.extend([(best[2][0], best[1]), (best[1], best[2][1])])  # type: ignore[index]
        return _finalize(
            scores,
            boundaries,
            index,
            ground_truth,
            self.config.tolerance,
            {"detector": "binary_segmentation", "semantics": "boundary_event"},
        )


class PELTDetector:
    """Offline PELT detector with an L2 mean-shift cost."""

    def __init__(self, config: PELTConfig | None = None) -> None:
        self.config = config or PELTConfig()

    def detect(
        self,
        data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
        *,
        ground_truth: Sequence[Any] | None = None,
    ) -> ChangePointDetectionResult:
        values, index = _values_and_index(data, self.config.feature)
        if self.config.use_ruptures:
            return RupturesAdapter("Pelt", self.config.model, min_size=self.config.min_size).detect(
                values,
                index,
                ground_truth=ground_truth,
                tolerance=self.config.tolerance,
                pen=self.config.penalty,
            )
        n = len(values)
        costs = {0: -self.config.penalty}
        paths: dict[int, list[int]] = {0: []}
        for t in range(self.config.min_size, n + 1):
            options = [
                (costs[tau] + _mean_cost(values[tau:t]) + self.config.penalty, tau)
                for tau in costs
                if t - tau >= self.config.min_size
            ]
            if not options:
                continue
            best_cost, best_tau = min(options)
            costs[t] = best_cost
            paths[t] = [*paths[best_tau], best_tau] if best_tau else []
        boundaries = [b for b in paths.get(n, []) if b > 0]
        scores = np.zeros(n)
        for i in range(self.config.min_size, n - self.config.min_size):
            scores[i] = max(
                0.0,
                _split_score(
                    values[i - self.config.min_size : i + self.config.min_size],
                    self.config.min_size,
                ),
            )
        return _finalize(
            scores,
            boundaries,
            index,
            ground_truth,
            self.config.tolerance,
            {"detector": "pelt", "semantics": "boundary_event"},
        )


class DistributionalChangePointDetector:
    """Kernel MMD or KS distributional boundary detector."""

    def __init__(self, config: DistributionalChangePointConfig | None = None) -> None:
        self.config = config or DistributionalChangePointConfig()

    def detect(
        self,
        data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
        *,
        ground_truth: Sequence[Any] | None = None,
    ) -> ChangePointDetectionResult:
        values, index = _values_and_index(data, self.config.feature)
        n = len(values)
        scores = np.zeros(n)
        for i in range(self.config.window, n - self.config.window):
            left = values[i - self.config.window : i]
            right = values[i : i + self.config.window]
            if self.config.method == "ks":
                scores[i] = float(ks_2samp(left[:, 0], right[:, 0]).statistic)
            else:
                gamma = self.config.gamma or 1.0 / max(float(np.var(values)), 1e-6)
                kxx = np.exp(-gamma * cdist(left, left, "sqeuclidean")).mean()
                kyy = np.exp(-gamma * cdist(right, right, "sqeuclidean")).mean()
                kxy = np.exp(-gamma * cdist(left, right, "sqeuclidean")).mean()
                scores[i] = float(sqrt(max(0.0, kxx + kyy - 2.0 * kxy)))
        candidates = np.flatnonzero(scores >= self.config.threshold)
        boundaries = _non_max_suppression(scores, candidates, self.config.window)
        return _finalize(
            scores,
            boundaries,
            index,
            ground_truth,
            self.config.tolerance,
            {
                "detector": "distributional",
                "method": self.config.method,
                "semantics": "boundary_event",
            },
        )


class BayesianOnlineChangePointDetector:
    """Bayesian online change-point detector using a conjugate Gaussian model."""

    def __init__(self, config: BayesianOnlineChangePointConfig | None = None) -> None:
        self.config = config or BayesianOnlineChangePointConfig()

    def detect(
        self,
        data: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
        *,
        ground_truth: Sequence[Any] | None = None,
    ) -> ChangePointDetectionResult:
        values, index = _values_and_index(data, self.config.feature)
        x = values[:, 0]
        log_r = np.array([0.0])
        mus = np.array([self.config.prior_mean])
        kappas = np.array([self.config.prior_kappa])
        alphas = np.array([self.config.prior_alpha])
        betas = np.array([self.config.prior_beta])
        probs = np.zeros(len(x))
        boundaries: list[int] = []
        for t, value in enumerate(x):
            pred = np.array(
                [
                    _student_t_logpdf(value, mus[i], kappas[i], alphas[i], betas[i])
                    for i in range(len(mus))
                ]
            )
            growth = log_r + pred + log(1.0 - self.config.hazard)
            cp = _logsumexp(log_r + pred + log(self.config.hazard))
            new_log_r = np.concatenate([[cp], growth])
            new_log_r -= _logsumexp(new_log_r)
            probs[t] = exp(new_log_r[0])
            if t >= self.config.min_size and probs[t] >= self.config.threshold:
                boundaries.append(t)
            mus, kappas, alphas, betas = _update_niw(value, mus, kappas, alphas, betas, self.config)
            log_r = new_log_r
        return _finalize(
            probs,
            boundaries,
            index,
            ground_truth,
            self.config.tolerance,
            {"detector": "bayesian_online", "semantics": "boundary_event"},
        )


class RupturesAdapter:
    """Adapter that makes optional ``ruptures`` support explicit.

    Only offline boundary detection is supported. Online filtering,
    probabilities, detection delay estimation without supplied ground truth, and
    recurring-state classification are deliberately unsupported.
    """

    def __init__(self, algorithm: str, model: str = "l2", *, min_size: int = 5) -> None:
        self.algorithm = algorithm
        self.model = model
        self.min_size = min_size

    def detect(
        self,
        values: np.ndarray,
        index: pd.Index,
        *,
        ground_truth: Sequence[Any] | None = None,
        tolerance: int = 5,
        **predict_kwargs: Any,
    ) -> ChangePointDetectionResult:
        try:
            import ruptures as rpt
        except ImportError as exc:
            raise UnsupportedModelOperation(
                "ruptures adapter requires the optional 'changepoint' extra"
            ) from exc
        if self.algorithm not in {"Binseg", "Pelt"}:
            raise UnsupportedModelOperation(f"ruptures algorithm {self.algorithm}")
        algo = getattr(rpt, self.algorithm)(model=self.model, min_size=self.min_size).fit(values)
        endpoints = algo.predict(**predict_kwargs)
        boundaries = [int(point) for point in endpoints if point < len(values)]
        scores = np.zeros(len(values))
        for boundary in boundaries:
            scores[boundary] = 1.0
        return _finalize(
            scores,
            boundaries,
            index,
            ground_truth,
            tolerance,
            {
                "detector": f"ruptures_{self.algorithm.lower()}",
                "model": self.model,
                "semantics": "boundary_event",
                "unsupported": [
                    "online_filtering",
                    "recurring_state_classification",
                    "native_change_probabilities",
                ],
            },
        )


def _non_max_suppression(
    scores: np.ndarray, candidates: np.ndarray, radius: int
) -> tuple[int, ...]:
    selected: list[int] = []
    for candidate in sorted(candidates, key=lambda i: scores[i], reverse=True):
        if all(abs(int(candidate) - existing) > radius for existing in selected):
            selected.append(int(candidate))
    return tuple(sorted(selected))


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + log(float(np.sum(np.exp(values - maximum))))


def _student_t_logpdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    nu = 2.0 * alpha
    scale = sqrt(beta * (kappa + 1.0) / (alpha * kappa))
    z = (x - mu) / scale
    return (
        lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * log(nu * pi)
        - log(scale)
        - ((nu + 1.0) / 2.0) * log1p((z * z) / nu)
    )


def log1p(value: float) -> float:
    return float(np.log1p(value))


def _update_niw(
    value: float,
    mus: np.ndarray,
    kappas: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    config: BayesianOnlineChangePointConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    new_kappas = kappas + 1.0
    new_mus = (kappas * mus + value) / new_kappas
    new_alphas = alphas + 0.5
    new_betas = betas + (kappas * (value - mus) ** 2) / (2.0 * new_kappas)
    return (
        np.concatenate([[config.prior_mean], new_mus]),
        np.concatenate([[config.prior_kappa], new_kappas]),
        np.concatenate([[config.prior_alpha], new_alphas]),
        np.concatenate([[config.prior_beta], new_betas]),
    )
