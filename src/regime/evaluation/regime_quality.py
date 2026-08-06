"""Regime-sequence quality, stability, and synthetic-truth diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations, pairwise

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, mutual_info_score

from regime.evaluation.metrics import ProbabilityKind, descriptor

METRICS = {
    "regime_persistence": descriptor("regime_persistence", ("states",), "maximize", display=".2%"),
    "switching_frequency": descriptor(
        "switching_frequency", ("states",), "diagnostic", display=".2%"
    ),
    "state_entropy": descriptor("state_entropy", ("states",), "diagnostic"),
    "probability_entropy": descriptor(
        "probability_entropy", ("probabilities",), "diagnostic", ProbabilityKind.FILTERED
    ),
    "state_recurrence": descriptor(
        "state_recurrence", ("states",), "diagnostic", aggregation="pooled"
    ),
    "rolling_refit_stability": descriptor(
        "rolling_refit_stability", ("state_runs",), "maximize", aggregation="none"
    ),
    "transition_stability": descriptor(
        "transition_stability", ("transition_matrices",), "minimize", aggregation="none"
    ),
}

ArrayLike = Sequence[float] | Sequence[int] | np.ndarray


def _labels(states: ArrayLike, *, name: str = "states") -> np.ndarray:
    labels = np.asarray(states, dtype=int)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    return labels


def _runs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    ends = np.r_[starts[1:], states.size]
    return states[starts], starts, ends - starts


def regime_persistence(states: ArrayLike) -> float:
    """Fraction of adjacent time steps that remain in the same regime."""
    labels = _labels(states)
    if labels.size < 2:
        return 1.0
    return float(np.mean(labels[1:] == labels[:-1]))


def duration_distribution(states: ArrayLike) -> dict[int, tuple[int, ...]]:
    """Run-length samples observed for each state."""
    labels = _labels(states)
    run_labels, _, durations = _runs(labels)
    result: dict[int, list[int]] = {}
    for label, duration in zip(run_labels, durations, strict=True):
        result.setdefault(int(label), []).append(int(duration))
    return {label: tuple(values) for label, values in result.items()}


def transition_stability(transition_matrices: Sequence[ArrayLike]) -> float:
    """Mean Frobenius distance between consecutive transition matrices."""
    matrices = [np.asarray(matrix, dtype=float) for matrix in transition_matrices]
    if len(matrices) < 2:
        return 0.0
    distances = [np.linalg.norm(b - a) for a, b in pairwise(matrices)]
    return float(np.mean(distances))


def state_occupancy(states: ArrayLike) -> dict[int, float]:
    """Empirical share of time spent in each state."""
    labels = _labels(states)
    values, counts = np.unique(labels, return_counts=True)
    return {
        int(value): float(count / labels.size) for value, count in zip(values, counts, strict=True)
    }


def state_entropy(states: ArrayLike, *, base: float = 2.0) -> float:
    """Entropy of empirical state occupancy."""
    probs = np.asarray(list(state_occupancy(states).values()), dtype=float)
    return _entropy(probs, base=base)


def probability_entropy(probabilities: ArrayLike, *, base: float = 2.0) -> float:
    """Mean entropy of per-time regime probability vectors."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    row_sums = probs.sum(axis=1, keepdims=True)
    normalized = np.divide(probs, row_sums, out=np.zeros_like(probs), where=row_sums > 0)
    return float(np.mean([_entropy(row, base=base) for row in normalized]))


def switching_frequency(states: ArrayLike) -> float:
    """Number of switches per adjacent time step."""
    return 1.0 - regime_persistence(states)


def detection_delay(true_boundaries: Sequence[int], detected_boundaries: Sequence[int]) -> float:
    """Mean non-negative delay from each true boundary to the next detected boundary."""
    detected = np.sort(np.asarray(detected_boundaries, dtype=int))
    delays = [
        detected[detected >= true].min() - true
        for true in true_boundaries
        if np.any(detected >= true)
    ]
    return float(np.mean(delays)) if delays else float("inf")


def false_alarm_rate(
    true_boundaries: Sequence[int], detected_boundaries: Sequence[int], *, tolerance: int = 0
) -> float:
    """Share of detected boundaries not within tolerance of a true boundary."""
    true = np.asarray(true_boundaries, dtype=int)
    detected = np.asarray(detected_boundaries, dtype=int)
    if detected.size == 0:
        return 0.0
    false = [not np.any(np.abs(true - boundary) <= tolerance) for boundary in detected]
    return float(np.mean(false))


def boundary_precision_recall(
    true_boundaries: Sequence[int], detected_boundaries: Sequence[int], *, tolerance: int = 0
) -> tuple[float, float]:
    """Precision and recall for synthetic regime boundaries."""
    true = list(map(int, true_boundaries))
    detected = list(map(int, detected_boundaries))
    matched_true: set[int] = set()
    true_positive = 0
    for boundary in detected:
        candidates = [
            idx
            for idx, value in enumerate(true)
            if idx not in matched_true and abs(value - boundary) <= tolerance
        ]
        if candidates:
            best = min(candidates, key=lambda idx: abs(true[idx] - boundary))
            matched_true.add(best)
            true_positive += 1
    precision = true_positive / len(detected) if detected else 0.0
    recall = true_positive / len(true) if true else 0.0
    return float(precision), float(recall)


def variation_of_information(
    labels_true: ArrayLike, labels_pred: ArrayLike, *, base: float = 2.0
) -> float:
    """Information distance between two state partitions."""
    true = _labels(labels_true, name="labels_true")
    pred = _labels(labels_pred, name="labels_pred")
    if true.shape != pred.shape:
        raise ValueError("label sequences must have equal length")
    return (
        state_entropy(true, base=base)
        + state_entropy(pred, base=base)
        - 2.0 * mutual_info_score(true, pred) / np.log(base)
    )


def adjusted_rand_index(labels_true: ArrayLike, labels_pred: ArrayLike) -> float:
    """Adjusted Rand index between synthetic truth and inferred labels."""
    return float(
        adjusted_rand_score(
            _labels(labels_true, name="labels_true"),
            _labels(labels_pred, name="labels_pred"),
        )
    )


def state_separability(features: ArrayLike, states: ArrayLike) -> float:
    """Between-centroid separation divided by average within-state dispersion."""
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    labels = _labels(states)
    if x.shape[0] != labels.size:
        raise ValueError("features and states must have the same number of rows")
    unique = np.unique(labels)
    centroids = np.vstack([x[labels == label].mean(axis=0) for label in unique])
    grand = x.mean(axis=0)
    between = float(np.mean(np.linalg.norm(centroids - grand, axis=1)))
    within_values = [
        np.linalg.norm(x[labels == label] - centroids[idx], axis=1).mean()
        for idx, label in enumerate(unique)
    ]
    within = float(np.mean(within_values))
    return between / max(within, 1e-12)


def state_recurrence(states: ArrayLike) -> dict[int, int]:
    """Number of distinct runs observed for each state."""
    run_labels, _, _ = _runs(_labels(states))
    values, counts = np.unique(run_labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def rare_state_sample_size(states: ArrayLike, *, threshold: float = 0.05) -> dict[int, int]:
    """Counts for states whose occupancy is at or below ``threshold``."""
    labels = _labels(states)
    occ = state_occupancy(labels)
    return {
        label: int(np.sum(labels == label)) for label, share in occ.items() if share <= threshold
    }


def parameter_uncertainty(samples: Mapping[str, ArrayLike] | ArrayLike) -> dict[str, float] | float:
    """Posterior/bootstrap parameter uncertainty as sample standard deviations."""
    if isinstance(samples, Mapping):
        return {
            key: float(np.nanstd(np.asarray(value, dtype=float), ddof=1))
            for key, value in samples.items()
        }
    return float(np.nanmean(np.nanstd(np.asarray(samples, dtype=float), axis=0, ddof=1)))


def random_initialization_stability(label_runs: Sequence[ArrayLike]) -> float:
    """Mean pairwise label-aligned ARI across random initializations."""
    return _mean_pairwise_ari(label_runs)


def rolling_refit_stability(label_runs: Sequence[ArrayLike]) -> float:
    """Mean pairwise label-aligned ARI across rolling refits."""
    return _mean_pairwise_ari(label_runs)


def label_alignment_stability(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Share of labels equal after optimal confusion-matrix alignment."""
    ref = _labels(reference, name="reference")
    cand = _labels(candidate, name="candidate")
    if ref.shape != cand.shape:
        raise ValueError("label sequences must have equal length")
    ref_values = np.unique(ref)
    cand_values = np.unique(cand)
    matrix = np.zeros((ref_values.size, cand_values.size), dtype=float)
    for i, ref_label in enumerate(ref_values):
        for j, cand_label in enumerate(cand_values):
            matrix[i, j] = np.sum((ref == ref_label) & (cand == cand_label))
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / ref.size)


def _mean_pairwise_ari(label_runs: Sequence[ArrayLike]) -> float:
    runs = [_labels(run, name="label_runs") for run in label_runs]
    if len(runs) < 2:
        return 1.0
    return float(np.mean([adjusted_rand_index(a, b) for a, b in combinations(runs, 2)]))


def _entropy(probabilities: np.ndarray, *, base: float) -> float:
    probs = probabilities[np.isfinite(probabilities) & (probabilities > 0)]
    if probs.size == 0:
        return 0.0
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)) / np.log(base))


__all__ = [
    "adjusted_rand_index",
    "boundary_precision_recall",
    "detection_delay",
    "duration_distribution",
    "false_alarm_rate",
    "label_alignment_stability",
    "parameter_uncertainty",
    "probability_entropy",
    "random_initialization_stability",
    "rare_state_sample_size",
    "regime_persistence",
    "rolling_refit_stability",
    "state_entropy",
    "state_occupancy",
    "state_recurrence",
    "state_separability",
    "switching_frequency",
    "transition_stability",
    "variation_of_information",
]
