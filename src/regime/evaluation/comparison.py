"""Out-of-sample model-comparison tests and multiple-testing controls.

The helpers in this module deliberately operate on aligned, finite observations only:
inputs may be ``pandas`` objects with indexes or NumPy-like arrays.  When indexes are
available, series are inner-joined by timestamp before any statistic is computed;
otherwise arrays must have the same out-of-sample length.  Missing and non-finite rows
are dropped after alignment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, skew

ArrayLike = Sequence[float] | np.ndarray | pd.Series
MatrixLike = Sequence[Sequence[float]] | np.ndarray | pd.DataFrame
BootstrapKind = Literal["block", "stationary"]
FdrMethod = Literal["bonferroni", "holm", "benjamini-hochberg", "benjamini-yekutieli"]


@dataclass(frozen=True)
class TestDocumentation:
    """Methodology notes that travel with each comparison result."""

    null_hypothesis: str
    required_input_series: str
    assumptions: str
    limitations: str
    valid_use_cases: str
    invalid_use_cases: str
    interpretation: str


@dataclass(frozen=True)
class ComparisonTestResult:
    """Generic scalar test result with methodology documentation."""

    statistic: float
    p_value: float
    n_obs: int
    method: str
    documentation: TestDocumentation
    extra: Mapping[str, float | tuple[str, ...] | tuple[float, ...]]


@dataclass(frozen=True)
class ModelConfidenceSetResult:
    """Model Confidence Set result.

    ``included_models`` is the confidence set that survived sequential elimination at
    ``alpha``.  ``elimination_order`` records removed models from first to last.
    """

    included_models: tuple[str, ...]
    excluded_models: tuple[str, ...]
    elimination_order: tuple[str, ...]
    p_values: Mapping[str, float]
    alpha: float
    n_obs: int
    documentation: TestDocumentation


@dataclass(frozen=True)
class MultipleTestingResult:
    """False-discovery or family-wise-error control for experiment grids."""

    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float
    method: FdrMethod
    documentation: TestDocumentation


def _series(values: ArrayLike, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.rename(name).astype(float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return pd.Series(arr, name=name, dtype=float)


def _frame(values: MatrixLike, *, model_names: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        frame = values.astype(float).copy()
        if model_names is not None:
            if len(model_names) != frame.shape[1]:
                raise ValueError("model_names length must match the number of columns")
            frame.columns = list(model_names)
        return frame
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("loss matrix must be two-dimensional: observations by models")
    columns = (
        list(model_names)
        if model_names is not None
        else [f"model_{i}" for i in range(arr.shape[1])]
    )
    if len(columns) != arr.shape[1]:
        raise ValueError("model_names length must match the number of columns")
    return pd.DataFrame(arr, columns=columns)


def _aligned_series(*series: pd.Series) -> pd.DataFrame:
    aligned = pd.concat(series, axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        raise ValueError("no aligned finite out-of-sample observations remain")
    return aligned


def align_loss_matrix(
    losses: MatrixLike, *, model_names: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return finite, index-aligned out-of-sample losses with rows containing any NaN removed."""
    frame = _frame(losses, model_names=model_names).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("losses must contain at least one aligned finite out-of-sample row")
    if frame.shape[1] < 2:
        raise ValueError("at least two models are required")
    return frame


def block_bootstrap_indices(
    n_obs: int, *, block_length: int, n_bootstrap: int, random_state: int | None = None
) -> np.ndarray:
    """Moving block bootstrap indexes for dependent out-of-sample series.

    Null hypothesis: this is a resampling scheme, not a hypothesis test.  Required input
    series: the aligned out-of-sample sample length.  Assumptions: dependence is captured
    by fixed-length adjacent blocks and the series is weakly stationary over the evaluation
    window.  Limitations: block length is user-chosen and boundary wrapping can duplicate
    observations.  Valid use cases: forecast-loss or return series with serial dependence.
    Invalid use cases: non-time-ordered data or structural breaks not represented in the
    resamples.  Interpretation: downstream bootstrap p-values are tail frequencies under
    the centered empirical resampling distribution.
    """
    if n_obs <= 0 or block_length <= 0 or n_bootstrap <= 0:
        raise ValueError("n_obs, block_length, and n_bootstrap must be positive")
    rng = np.random.default_rng(random_state)
    out = np.empty((n_bootstrap, n_obs), dtype=int)
    starts = np.arange(n_obs)
    for boot in range(n_bootstrap):
        pieces: list[np.ndarray] = []
        while sum(piece.size for piece in pieces) < n_obs:
            start = int(rng.choice(starts))
            pieces.append((start + np.arange(block_length)) % n_obs)
        out[boot] = np.concatenate(pieces)[:n_obs]
    return out


def stationary_bootstrap_indices(
    n_obs: int, *, average_block_length: float, n_bootstrap: int, random_state: int | None = None
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indexes with geometric block lengths.

    Null hypothesis: this is a resampling scheme, not a hypothesis test.  Required input
    series: the aligned out-of-sample sample length.  Assumptions: observations are ordered
    and approximately stationary, with dependence summarized by a mean block length.
    Limitations: average block length materially affects inference.  Valid use cases:
    dependent forecast-loss or strategy-return series.  Invalid use cases: shuffled panels,
    strongly nonstationary periods, or tiny samples.  Interpretation: downstream p-values
    are bootstrap tail probabilities computed from these pseudo-samples.
    """
    if n_obs <= 0 or average_block_length <= 0 or n_bootstrap <= 0:
        raise ValueError("n_obs, average_block_length, and n_bootstrap must be positive")
    rng = np.random.default_rng(random_state)
    restart_probability = min(1.0, 1.0 / average_block_length)
    out = np.empty((n_bootstrap, n_obs), dtype=int)
    for boot in range(n_bootstrap):
        idx = int(rng.integers(n_obs))
        for pos in range(n_obs):
            if pos > 0 and rng.random() < restart_probability:
                idx = int(rng.integers(n_obs))
            out[boot, pos] = idx
            idx = (idx + 1) % n_obs
    return out


def _bootstrap_indices(
    n: int, kind: BootstrapKind, block_length: float, b: int, seed: int | None
) -> np.ndarray:
    if kind == "block":
        return block_bootstrap_indices(
            n, block_length=max(1, round(block_length)), n_bootstrap=b, random_state=seed
        )
    return stationary_bootstrap_indices(
        n, average_block_length=block_length, n_bootstrap=b, random_state=seed
    )


def diebold_mariano_test(
    loss_model: ArrayLike,
    loss_benchmark: ArrayLike,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    bandwidth: int | None = None,
) -> ComparisonTestResult:
    """Diebold-Mariano equal predictive-accuracy test.

    Null hypothesis: the expected aligned out-of-sample loss differential is zero.
    Required input series: two loss series for the same forecast origins; lower losses are
    better.  Assumptions: covariance-stationary loss differentials and enough observations
    for a HAC long-run-variance estimate.  Limitations: low power in small samples and
    sensitivity to loss-function choice and HAC bandwidth.  Valid use cases: comparing two
    non-nested forecasts on a common holdout period.  Invalid use cases: in-sample fit,
    unaligned forecast windows, or losses computed with future information.  Interpretation:
    small p-values reject equal predictive accuracy in the requested direction.
    """
    data = _aligned_series(_series(loss_model, "model"), _series(loss_benchmark, "benchmark"))
    diff = (data["model"] - data["benchmark"]).to_numpy()
    n = diff.size
    lag = int(np.floor(n ** (1 / 3))) if bandwidth is None else bandwidth
    centered = diff - np.mean(diff)
    lrv = float(np.dot(centered, centered) / n)
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    if lrv <= 0:
        raise ValueError(
            "Diebold-Mariano statistic is undefined because long-run variance is not positive"
        )
    stat = float(np.mean(diff) / np.sqrt(lrv / n))
    if alternative == "less":
        p_value = float(norm.cdf(stat))
    elif alternative == "greater":
        p_value = float(norm.sf(stat))
    else:
        p_value = float(2.0 * norm.sf(abs(stat)))
    return ComparisonTestResult(
        stat,
        p_value,
        n,
        "Diebold-Mariano",
        _dm_doc(),
        {"mean_loss_difference": float(np.mean(diff)), "bandwidth": float(lag)},
    )


# attach concise docs without duplicating dataclass creation at call sites
def _doc(
    null: str, req: str, ass: str, lim: str, valid: str, invalid: str, interp: str
) -> TestDocumentation:
    return TestDocumentation(null, req, ass, lim, valid, invalid, interp)


def _dm_doc() -> TestDocumentation:
    return _doc(
        "Expected loss differential equals zero.",
        "Two aligned out-of-sample loss series.",
        "Stationary loss differential; HAC variance is adequate.",
        "Small-sample and bandwidth sensitive.",
        "Pairwise holdout forecast comparison.",
        "In-sample, unaligned, or look-ahead contaminated losses.",
        "Small p-values reject equal predictive accuracy.",
    )


def reality_check(
    losses: MatrixLike,
    benchmark: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    block_length: float = 10,
    bootstrap: BootstrapKind = "stationary",
    random_state: int | None = None,
) -> ComparisonTestResult:
    """White Reality Check for data-snooped benchmark outperformance."""
    frame = align_loss_matrix(losses)
    bench = _series(benchmark, "benchmark")
    data = (
        pd.concat([frame, bench], axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    )
    models = data.iloc[:, :-1]
    diff = (
        data["benchmark"].to_numpy()[:, None] - models.to_numpy()
    )  # positive means model improves
    mean_diff = diff.mean(axis=0)
    stat = float(np.sqrt(len(data)) * np.max(mean_diff))
    centered = diff - mean_diff
    idx = _bootstrap_indices(len(data), bootstrap, block_length, n_bootstrap, random_state)
    boot_stats = np.sqrt(len(data)) * np.max(centered[idx].mean(axis=1), axis=1)
    p_value = float((1 + np.sum(boot_stats >= stat)) / (n_bootstrap + 1))
    doc = _doc(
        "No candidate improves on the benchmark after accounting for search.",
        "Aligned candidate and benchmark out-of-sample loss series.",
        "Stationary dependence captured by bootstrap blocks.",
        "Can be conservative; bootstrap choices matter.",
        "Many candidate strategies/models versus one benchmark.",
        "Unaligned windows, non-loss metrics where larger is better unless transformed.",
        "Small p-values indicate at least one candidate beats the benchmark.",
    )
    return ComparisonTestResult(
        stat,
        p_value,
        len(data),
        "Reality Check",
        doc,
        {
            "best_model": (str(models.columns[int(np.argmax(mean_diff))]),),
            "best_mean_improvement": float(np.max(mean_diff)),
        },
    )


def superior_predictive_ability_test(
    losses: MatrixLike,
    benchmark: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    block_length: float = 10,
    bootstrap: BootstrapKind = "stationary",
    random_state: int | None = None,
) -> ComparisonTestResult:
    """Hansen Superior Predictive Ability test with studentized loss improvements."""
    frame = align_loss_matrix(losses)
    data = pd.concat([frame, _series(benchmark, "benchmark")], axis=1, join="inner").dropna()
    diff = data["benchmark"].to_numpy()[:, None] - data.iloc[:, :-1].to_numpy()
    n = diff.shape[0]
    scale = np.maximum(diff.std(axis=0, ddof=1), 1e-12)
    stat = float(np.max(np.sqrt(n) * np.maximum(diff.mean(axis=0), 0.0) / scale))
    centered = diff - diff.mean(axis=0)
    idx = _bootstrap_indices(n, bootstrap, block_length, n_bootstrap, random_state)
    boot = np.max(np.sqrt(n) * np.maximum(centered[idx].mean(axis=1), 0.0) / scale, axis=1)
    p_value = float((1 + np.sum(boot >= stat)) / (n_bootstrap + 1))
    doc = _doc(
        "No candidate has superior expected predictive ability over the benchmark.",
        "Aligned candidate and benchmark out-of-sample losses.",
        "Stationary loss improvements and valid block bootstrap.",
        "Studentization can be unstable with near-zero variance.",
        "Large experiment grids compared with a benchmark.",
        "In-sample tuning evidence or dependent panels without temporal order.",
        "Small p-values indicate at least one superior candidate.",
    )
    return ComparisonTestResult(stat, p_value, n, "Superior Predictive Ability", doc, {})


def model_confidence_set(
    losses: MatrixLike,
    *,
    alpha: float = 0.10,
    n_bootstrap: int = 1000,
    block_length: float = 10,
    bootstrap: BootstrapKind = "stationary",
    random_state: int | None = None,
) -> ModelConfidenceSetResult:
    """Hansen-Lunde-Nason Model Confidence Set using sequential range elimination."""
    frame = align_loss_matrix(losses)
    rng_seed = random_state
    remaining = list(frame.columns)
    removed: list[str] = []
    pvals: dict[str, float] = {}
    while len(remaining) > 1:
        cur = frame[remaining].to_numpy()
        means = cur.mean(axis=0)
        stat = float(np.max(means) - np.min(means))
        centered = cur - means
        idx = _bootstrap_indices(cur.shape[0], bootstrap, block_length, n_bootstrap, rng_seed)
        boot = np.max(centered[idx].mean(axis=1), axis=1) - np.min(
            centered[idx].mean(axis=1), axis=1
        )
        p_value = float((1 + np.sum(boot >= stat)) / (n_bootstrap + 1))
        worst = remaining[int(np.argmax(means))]
        pvals[worst] = p_value
        if p_value > alpha:
            break
        removed.append(worst)
        remaining.remove(worst)
        rng_seed = None if rng_seed is None else rng_seed + 1
    doc = _doc(
        "All models in the current set have equal expected loss.",
        "Aligned out-of-sample loss matrix, lower is better.",
        "Bootstrap captures serial dependence; losses are comparable across models.",
        "Simplified range statistic; results depend on alpha and bootstrap settings.",
        "Reporting a set of statistically indistinguishable best models.",
        "Unaligned forecasts, different target periods, or metrics where larger is better.",
        "The confidence set contains models not rejected as inferior at alpha.",
    )
    return ModelConfidenceSetResult(
        tuple(remaining), tuple(removed), tuple(removed), pvals, alpha, len(frame), doc
    )


def probabilistic_sharpe_ratio(
    returns: ArrayLike, *, benchmark_sharpe: float = 0.0, periods_per_year: float = 1.0
) -> ComparisonTestResult:
    """Probabilistic Sharpe Ratio: probability true Sharpe exceeds a benchmark."""
    r = _series(returns, "returns").dropna().to_numpy()
    n = r.size
    sr = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods_per_year))
    sr_period = sr / np.sqrt(periods_per_year)
    bench_period = benchmark_sharpe / np.sqrt(periods_per_year)
    denom = np.sqrt(
        max(
            1e-12,
            1
            - skew(r) * sr_period
            + ((np.mean(((r - r.mean()) / r.std(ddof=1)) ** 4) - 1) / 4) * sr_period**2,
        )
    )
    prob = float(norm.cdf((sr_period - bench_period) * np.sqrt(n - 1) / denom))
    doc = _doc(
        "True Sharpe ratio is at or below the benchmark Sharpe.",
        "Aligned out-of-sample strategy returns.",
        "Returns are representative with finite moments; skew/kurtosis adjustment is adequate.",
        "Not robust to severe nonstationarity or underestimated costs.",
        "Single strategy performance significance versus a hurdle.",
        "In-sample optimized returns or non-return loss series.",
        "The reported p-value is one minus the probability true Sharpe exceeds the hurdle.",
    )
    return ComparisonTestResult(
        sr, 1.0 - prob, n, "Probabilistic Sharpe Ratio", doc, {"probability": prob}
    )


def deflated_sharpe_ratio(
    returns: ArrayLike, *, n_trials: int, periods_per_year: float = 1.0
) -> ComparisonTestResult:
    """Deflated Sharpe Ratio adjusting for non-normality and multiple trials."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    expected_max = norm.ppf(1 - 1 / max(np.e * n_trials, 1.000001)) / np.sqrt(periods_per_year)
    result = probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe=float(expected_max * np.sqrt(periods_per_year)),
        periods_per_year=periods_per_year,
    )
    doc = _doc(
        "Observed Sharpe does not exceed the expected best Sharpe from multiple trials.",
        "Out-of-sample returns for the selected strategy plus number of tried configurations.",
        "Trial count approximates the effective search breadth; finite moments.",
        "Effective trials are hard to estimate and correlated searches reduce precision.",
        "Post-selection strategy assessment after large grids.",
        "Ignoring transaction costs, live drift, or using in-sample returns.",
        "Small p-values indicate Sharpe remains significant after deflation.",
    )
    return ComparisonTestResult(
        result.statistic,
        result.p_value,
        result.n_obs,
        "Deflated Sharpe Ratio",
        doc,
        {
            "n_trials": float(n_trials),
            "benchmark_sharpe": float(expected_max * np.sqrt(periods_per_year)),
        },
    )


def false_discovery_control(
    p_values: ArrayLike, *, alpha: float = 0.05, method: FdrMethod = "benjamini-hochberg"
) -> MultipleTestingResult:
    """Adjust p-values for large experiment grids using FWER/FDR controls."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0 or np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("p_values must be a one-dimensional finite array in [0, 1]")
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    if method == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    elif method == "holm":
        vals = np.maximum.accumulate((m - np.arange(m)) * ranked)
        adj = np.empty(m)
        adj[order] = np.minimum(vals, 1.0)
    else:
        c_m = float(np.sum(1 / np.arange(1, m + 1))) if method == "benjamini-yekutieli" else 1.0
        vals = np.minimum.accumulate((m * c_m / np.arange(m, 0, -1)) * ranked[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(vals, 1.0)
    doc = _doc(
        "Each individual null hypothesis is true unless rejected after multiplicity adjustment.",
        "P-values from aligned out-of-sample experiments.",
        "BH assumes independent/positive dependence; BY is valid under arbitrary dependence.",
        "Controls statistical discoveries, not economic materiality or data leakage.",
        "Large model-selection and feature-search grids.",
        "Replacing proper holdout design or correcting biased p-values.",
        "Reject entries whose adjusted p-value is at most alpha.",
    )
    return MultipleTestingResult(
        tuple(float(x) for x in adj), tuple(bool(x) for x in adj <= alpha), alpha, method, doc
    )


__all__ = [
    "ComparisonTestResult",
    "ModelConfidenceSetResult",
    "MultipleTestingResult",
    "TestDocumentation",
    "align_loss_matrix",
    "block_bootstrap_indices",
    "deflated_sharpe_ratio",
    "diebold_mariano_test",
    "false_discovery_control",
    "model_confidence_set",
    "probabilistic_sharpe_ratio",
    "reality_check",
    "stationary_bootstrap_indices",
    "superior_predictive_ability_test",
]
