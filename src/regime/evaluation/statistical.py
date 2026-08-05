"""Statistical forecast-evaluation metrics for regime-switching models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy.stats import norm

ArrayLike = Sequence[float] | np.ndarray


class PredictiveDistribution(Protocol):
    """Minimal protocol for predictive distributions used by scoring rules."""

    def logpdf(self, x: ArrayLike) -> np.ndarray: ...

    def cdf(self, x: ArrayLike) -> np.ndarray: ...

    def ppf(self, q: ArrayLike) -> np.ndarray: ...


def _as_float_array(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return array


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(np.broadcast_shapes(*(array.shape for array in arrays)), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(np.broadcast_to(array, mask.shape))
    return mask


def _mean_valid(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("metric is undefined because no finite observations remain")
    return float(np.mean(finite))


def out_of_sample_log_likelihood(
    log_likelihoods: ArrayLike | None = None,
    *,
    observations: ArrayLike | None = None,
    predictive_distribution: PredictiveDistribution | None = None,
    average: bool = False,
) -> float:
    """Compute held-out log likelihood from per-observation values or a distribution."""
    if log_likelihoods is None:
        if observations is None or predictive_distribution is None:
            raise ValueError("provide log_likelihoods or observations with predictive_distribution")
        values = predictive_distribution.logpdf(_as_float_array(observations, name="observations"))
    else:
        values = _as_float_array(log_likelihoods, name="log_likelihoods")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("log likelihood is undefined because no finite values remain")
    result = np.mean(finite) if average else np.sum(finite)
    return float(result)


def predictive_log_score(
    observations: ArrayLike,
    predictive_density: ArrayLike | None = None,
    *,
    predictive_distribution: PredictiveDistribution | None = None,
    eps: float = 1e-12,
) -> float:
    """Return the mean negative log predictive density, lower is better."""
    y = _as_float_array(observations, name="observations")
    if predictive_density is None:
        if predictive_distribution is None:
            raise ValueError("provide predictive_density or predictive_distribution")
        log_pdf = np.asarray(predictive_distribution.logpdf(y), dtype=float)
    else:
        density = np.asarray(predictive_density, dtype=float)
        log_pdf = np.log(np.clip(density, eps, None))
    return -_mean_valid(log_pdf)


def brier_score(y_true: ArrayLike, probabilities: ArrayLike) -> float:
    """Mean squared error of probabilistic binary or multiclass forecasts."""
    truth = np.asarray(y_true)
    probs = _as_float_array(probabilities, name="probabilities")
    if probs.ndim == 1:
        obs = np.asarray(truth, dtype=float)
        mask = _finite_mask(obs, probs)
        return float(np.mean((probs[mask] - obs[mask]) ** 2))
    labels = truth.astype(int)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and y_true must have the same number of rows")
    one_hot = np.eye(probs.shape[1], dtype=float)[labels]
    mask = np.isfinite(probs).all(axis=1)
    return float(np.mean(np.sum((probs[mask] - one_hot[mask]) ** 2, axis=1)))


def calibration_error(y_true: ArrayLike, probabilities: ArrayLike, *, n_bins: int = 10) -> float:
    """Maximum absolute bin-wise calibration gap for binary forecasts."""
    return _calibration_gap(y_true, probabilities, n_bins=n_bins, weighted=False, maximum=True)


def expected_calibration_error(
    y_true: ArrayLike, probabilities: ArrayLike, *, n_bins: int = 10
) -> float:
    """Weighted average absolute bin-wise calibration gap for binary forecasts."""
    return _calibration_gap(y_true, probabilities, n_bins=n_bins, weighted=True, maximum=False)


def _calibration_gap(
    y_true: ArrayLike, probabilities: ArrayLike, *, n_bins: int, weighted: bool, maximum: bool
) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    obs = _as_float_array(y_true, name="y_true")
    probs = _as_float_array(probabilities, name="probabilities")
    mask = _finite_mask(obs, probs)
    obs, probs = obs[mask], np.clip(probs[mask], 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps: list[float] = []
    weights: list[float] = []
    for idx in range(n_bins):
        upper_mask = probs <= bins[idx + 1] if idx == n_bins - 1 else probs < bins[idx + 1]
        in_bin = (probs >= bins[idx]) & upper_mask
        if np.any(in_bin):
            gaps.append(float(abs(np.mean(obs[in_bin]) - np.mean(probs[in_bin]))))
            weights.append(float(np.mean(in_bin)))
    if not gaps:
        raise ValueError("calibration error is undefined because no finite values remain")
    if maximum:
        return float(max(gaps))
    return float(np.average(gaps, weights=weights if weighted else None))


def qlike(realized: ArrayLike, forecast: ArrayLike, *, eps: float = 1e-12) -> float:
    """Quasi-likelihood loss for positive variance or volatility forecasts."""
    y = np.clip(_as_float_array(realized, name="realized"), eps, None)
    f = np.clip(_as_float_array(forecast, name="forecast"), eps, None)
    mask = _finite_mask(y, f)
    ratio = y[mask] / f[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    y = _as_float_array(y_true, name="y_true")
    pred = _as_float_array(y_pred, name="y_pred")
    mask = _finite_mask(y, pred)
    return float(np.mean(np.abs(y[mask] - pred[mask])))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    y = _as_float_array(y_true, name="y_true")
    pred = _as_float_array(y_pred, name="y_pred")
    mask = _finite_mask(y, pred)
    return float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2)))


def crps(
    observations: ArrayLike,
    *,
    predictive_distribution: PredictiveDistribution | None = None,
    samples: np.ndarray | None = None,
    mean: ArrayLike | None = None,
    std: ArrayLike | None = None,
) -> float:
    """Continuous ranked probability score using samples, normals, or numeric integration."""
    y = _as_float_array(observations, name="observations")
    if samples is not None:
        draws = np.asarray(samples, dtype=float)
        if draws.ndim == 1:
            draws = draws[:, np.newaxis]
        if draws.shape[0] == y.shape[0]:
            draws = draws.T
        term1 = np.mean(np.abs(draws - y), axis=0)
        pairwise = np.abs(draws[:, None, :] - draws[None, :, :])
        return float(np.mean(term1 - 0.5 * np.mean(pairwise, axis=(0, 1))))
    if mean is not None and std is not None:
        mu = np.asarray(mean, dtype=float)
        sigma = np.clip(np.asarray(std, dtype=float), 1e-12, None)
        z = (y - mu) / sigma
        values = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        return _mean_valid(values)
    if predictive_distribution is None:
        raise ValueError("provide samples, normal mean/std, or predictive_distribution")
    grid = np.linspace(1e-4, 1.0 - 1e-4, 512)
    quantiles = np.asarray(predictive_distribution.ppf(grid), dtype=float)
    if quantiles.ndim == 1:
        quantiles = quantiles[:, np.newaxis]
    indicators = (y[np.newaxis, :] <= quantiles).astype(float)
    values = 2.0 * np.trapz((indicators - grid[:, np.newaxis]) * (quantiles - y), grid, axis=0)
    return _mean_valid(values)


def coverage(observations: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Fraction of observations inside predictive intervals."""
    y = _as_float_array(observations, name="observations")
    lo = _as_float_array(lower, name="lower")
    hi = _as_float_array(upper, name="upper")
    mask = _finite_mask(y, lo, hi)
    return float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))


def sharpness(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean predictive interval width, lower is sharper."""
    lo = _as_float_array(lower, name="lower")
    hi = _as_float_array(upper, name="upper")
    mask = _finite_mask(lo, hi)
    return float(np.mean(hi[mask] - lo[mask]))


__all__ = [
    "brier_score",
    "calibration_error",
    "coverage",
    "crps",
    "expected_calibration_error",
    "mae",
    "out_of_sample_log_likelihood",
    "predictive_log_score",
    "qlike",
    "rmse",
    "sharpness",
]
