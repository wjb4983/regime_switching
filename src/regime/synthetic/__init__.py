"""Deterministic synthetic generators for regime-switching workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Container returned by all synthetic generators."""

    observations: FloatArray
    latent_states: IntArray | None
    transition_matrix: FloatArray | None
    duration_distribution: FloatArray | None
    true_change_points: IntArray | None
    metadata: dict[str, Any]
    seed: int | None


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _normalise_rows(matrix: FloatArray) -> FloatArray:
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("transition matrix rows must have positive sums")
    return matrix / row_sums


def _default_transition(n_states: int, persistence: float = 0.9) -> FloatArray:
    if n_states < 1:
        raise ValueError("n_states must be positive")
    if n_states == 1:
        return np.ones((1, 1), dtype=np.float64)
    off = (1.0 - persistence) / (n_states - 1)
    matrix = np.full((n_states, n_states), off, dtype=np.float64)
    np.fill_diagonal(matrix, persistence)
    return matrix


def _sample_markov_states(
    n_steps: int,
    transition_matrix: FloatArray,
    generator: np.random.Generator,
    initial_probs: FloatArray | None = None,
) -> IntArray:
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    transition_matrix = _normalise_rows(transition_matrix)
    n_states = transition_matrix.shape[0]
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError("transition_matrix must be square")
    probs = (
        np.full(n_states, 1.0 / n_states) if initial_probs is None else np.asarray(initial_probs)
    )
    states = np.empty(n_steps, dtype=np.int_)
    states[0] = generator.choice(n_states, p=probs / probs.sum())
    for idx in range(1, n_steps):
        states[idx] = generator.choice(n_states, p=transition_matrix[states[idx - 1]])
    return states


def _change_points(states: IntArray) -> IntArray:
    return np.flatnonzero(states[1:] != states[:-1]).astype(np.int_) + 1


def _state_parameters(n_states: int, n_features: int) -> tuple[FloatArray, FloatArray]:
    means = np.linspace(-2.0, 2.0, n_states, dtype=np.float64)[:, None]
    means = np.repeat(means, n_features, axis=1)
    scales = np.linspace(0.6, 1.4, n_states, dtype=np.float64)[:, None]
    scales = np.repeat(scales, n_features, axis=1)
    return means, scales


def gaussian_hmm(
    n_steps: int = 200,
    n_states: int = 3,
    n_features: int = 1,
    *,
    transition_matrix: FloatArray | None = None,
    seed: int | None = None,
) -> SyntheticDataset:
    """Generate Gaussian emissions from a hidden Markov model."""
    generator = _rng(seed)
    transition = (
        _default_transition(n_states)
        if transition_matrix is None
        else _normalise_rows(transition_matrix)
    )
    states = _sample_markov_states(n_steps, transition, generator)
    means, scales = _state_parameters(n_states, n_features)
    observations = generator.normal(means[states], scales[states]).astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        transition,
        None,
        _change_points(states),
        {"kind": "gaussian_hmm", "means": means, "scales": scales},
        seed,
    )


def student_t_hmm(
    n_steps: int = 200,
    n_states: int = 3,
    n_features: int = 1,
    *,
    df: float = 5.0,
    transition_matrix: FloatArray | None = None,
    seed: int | None = None,
) -> SyntheticDataset:
    """Generate Student-t emissions from a hidden Markov model."""
    generator = _rng(seed)
    transition = (
        _default_transition(n_states)
        if transition_matrix is None
        else _normalise_rows(transition_matrix)
    )
    states = _sample_markov_states(n_steps, transition, generator)
    means, scales = _state_parameters(n_states, n_features)
    observations = (
        means[states] + scales[states] * generator.standard_t(df, (n_steps, n_features))
    ).astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        transition,
        None,
        _change_points(states),
        {"kind": "student_t_hmm", "df": df, "means": means, "scales": scales},
        seed,
    )


def hsmm_non_geometric(
    n_steps: int = 200,
    n_states: int = 3,
    n_features: int = 1,
    *,
    duration_pmf: FloatArray | None = None,
    seed: int | None = None,
) -> SyntheticDataset:
    """Generate an HSMM with explicit non-geometric duration probabilities."""
    generator = _rng(seed)
    durations = np.asarray(
        duration_pmf
        if duration_pmf is not None
        else np.tile([0.05, 0.15, 0.35, 0.30, 0.15], (n_states, 1)),
        dtype=np.float64,
    )
    durations = _normalise_rows(durations)
    states_list: list[int] = []
    current = int(generator.integers(n_states))
    while len(states_list) < n_steps:
        dur = int(generator.choice(np.arange(1, durations.shape[1] + 1), p=durations[current]))
        states_list.extend([current] * dur)
        current = (current + int(generator.integers(1, n_states))) % n_states
    states = np.asarray(states_list[:n_steps], dtype=np.int_)
    means, scales = _state_parameters(n_states, n_features)
    observations = generator.normal(means[states], scales[states]).astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        None,
        durations,
        _change_points(states),
        {"kind": "hsmm_non_geometric", "support": np.arange(1, durations.shape[1] + 1)},
        seed,
    )


def markov_switching_ar(
    n_steps: int = 200, n_states: int = 2, *, seed: int | None = None
) -> SyntheticDataset:
    """Generate a univariate Markov-switching AR(1) process."""
    generator = _rng(seed)
    transition = _default_transition(n_states, 0.92)
    states = _sample_markov_states(n_steps, transition, generator)
    coeffs = np.linspace(-0.4, 0.8, n_states)
    intercepts = np.linspace(-1.0, 1.0, n_states)
    noise = np.linspace(0.3, 0.9, n_states)
    y = np.empty(n_steps, dtype=np.float64)
    y[0] = generator.normal(intercepts[states[0]], noise[states[0]])
    for idx in range(1, n_steps):
        s = states[idx]
        y[idx] = intercepts[s] + coeffs[s] * y[idx - 1] + generator.normal(0.0, noise[s])
    return SyntheticDataset(
        y[:, None],
        states,
        transition,
        None,
        _change_points(states),
        {"kind": "markov_switching_ar", "ar_coefficients": coeffs, "intercepts": intercepts},
        seed,
    )


def switching_stochastic_volatility(
    n_steps: int = 200, n_states: int = 2, *, seed: int | None = None
) -> SyntheticDataset:
    """Generate returns with latent Markov-switching log-volatility levels."""
    generator = _rng(seed)
    transition = _default_transition(n_states, 0.95)
    states = _sample_markov_states(n_steps, transition, generator)
    log_vol = np.linspace(-1.5, 0.8, n_states)[states] + generator.normal(0.0, 0.15, n_steps)
    observations = (np.exp(log_vol) * generator.normal(size=n_steps))[:, None].astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        transition,
        None,
        _change_points(states),
        {"kind": "switching_stochastic_volatility", "log_volatility": log_vol},
        seed,
    )


def switching_covariance_matrices(
    n_steps: int = 200, n_states: int = 3, n_features: int = 3, *, seed: int | None = None
) -> SyntheticDataset:
    """Generate multivariate observations with state-dependent covariance matrices."""
    generator = _rng(seed)
    transition = _default_transition(n_states, 0.9)
    states = _sample_markov_states(n_steps, transition, generator)
    covariances = np.stack([np.eye(n_features) * (idx + 1) for idx in range(n_states)]).astype(
        np.float64
    )
    observations = np.vstack(
        [generator.multivariate_normal(np.zeros(n_features), covariances[s]) for s in states]
    ).astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        transition,
        None,
        _change_points(states),
        {"kind": "switching_covariance_matrices", "covariances": covariances},
        seed,
    )


def abrupt_change_points(
    n_steps: int = 200,
    n_features: int = 1,
    *,
    change_points: tuple[int, ...] = (60, 130),
    seed: int | None = None,
) -> SyntheticDataset:
    """Generate piecewise-constant regimes with abrupt jumps at known change points."""
    generator = _rng(seed)
    bounds = (0, *change_points, n_steps)
    states = np.concatenate(
        [np.full(bounds[i + 1] - bounds[i], i, dtype=np.int_) for i in range(len(bounds) - 1)]
    )
    means, _ = _state_parameters(len(bounds) - 1, n_features)
    observations = generator.normal(means[states], 0.35).astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        None,
        None,
        np.asarray(change_points, dtype=np.int_),
        {"kind": "abrupt_change_points", "means": means},
        seed,
    )


def gradual_transitions(
    n_steps: int = 200,
    n_features: int = 1,
    *,
    center: int | None = None,
    width: float = 15.0,
    seed: int | None = None,
) -> SyntheticDataset:
    """Generate two regimes connected by a smooth logistic transition."""
    generator = _rng(seed)
    center = n_steps // 2 if center is None else center
    weights = 1.0 / (1.0 + np.exp(-(np.arange(n_steps) - center) / width))
    baseline = (-2.0 * (1.0 - weights) + 2.0 * weights)[:, None]
    observations = np.repeat(baseline, n_features, axis=1) + generator.normal(
        0.0, 0.25, (n_steps, n_features)
    )
    states = (weights >= 0.5).astype(np.int_)
    return SyntheticDataset(
        observations.astype(np.float64),
        states,
        None,
        None,
        np.asarray([center], dtype=np.int_),
        {"kind": "gradual_transitions", "weights": weights, "width": width},
        seed,
    )


def recurring_regimes(
    n_steps: int = 200, period: int = 40, n_features: int = 1, *, seed: int | None = None
) -> SyntheticDataset:
    """Generate regimes that recur on a fixed cycle."""
    generator = _rng(seed)
    states = ((np.arange(n_steps) // (period // 2)) % 2).astype(np.int_)
    means, _ = _state_parameters(2, n_features)
    observations = generator.normal(means[states], 0.4).astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        None,
        None,
        _change_points(states),
        {"kind": "recurring_regimes", "period": period},
        seed,
    )


def non_recurring_regimes(
    n_steps: int = 200, n_regimes: int = 4, n_features: int = 1, *, seed: int | None = None
) -> SyntheticDataset:
    """Generate regimes that appear once in sequence."""
    cps = tuple(np.linspace(0, n_steps, n_regimes + 1, dtype=int)[1:-1].tolist())
    data = abrupt_change_points(n_steps, n_features, change_points=cps, seed=seed)
    metadata = data.metadata | {"kind": "non_recurring_regimes", "n_regimes": n_regimes}
    return SyntheticDataset(
        data.observations, data.latent_states, None, None, data.true_change_points, metadata, seed
    )


def rare_crisis_states(
    n_steps: int = 300,
    n_features: int = 1,
    *,
    crisis_probability: float = 0.03,
    seed: int | None = None,
) -> SyntheticDataset:
    """Generate mostly calm observations with rare high-volatility crisis states."""
    generator = _rng(seed)
    states = (generator.random(n_steps) < crisis_probability).astype(np.int_)
    observations = generator.normal(
        np.where(states == 1, -4.0, 0.0)[:, None],
        np.where(states == 1, 3.0, 0.5)[:, None],
        (n_steps, n_features),
    ).astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        None,
        None,
        _change_points(states),
        {"kind": "rare_crisis_states", "crisis_probability": crisis_probability},
        seed,
    )


def overlapping_emissions(
    n_steps: int = 200, n_states: int = 2, n_features: int = 1, *, seed: int | None = None
) -> SyntheticDataset:
    """Generate HMM states whose emissions intentionally overlap."""
    generator = _rng(seed)
    transition = _default_transition(n_states, 0.9)
    states = _sample_markov_states(n_steps, transition, generator)
    means = np.linspace(-0.3, 0.3, n_states)[:, None].repeat(n_features, axis=1)
    observations = generator.normal(means[states], 1.0).astype(np.float64)
    return SyntheticDataset(
        observations,
        states,
        transition,
        None,
        _change_points(states),
        {"kind": "overlapping_emissions", "means": means, "scale": 1.0},
        seed,
    )


def missing_observations(
    base: SyntheticDataset | None = None,
    *,
    missing_probability: float = 0.1,
    seed: int | None = None,
) -> SyntheticDataset:
    """Inject missing observations into a generated dataset."""
    data = gaussian_hmm(seed=seed) if base is None else base
    generator = _rng(seed)
    observations = data.observations.copy()
    mask = generator.random(observations.shape) < missing_probability
    observations[mask] = np.nan
    return SyntheticDataset(
        observations,
        data.latent_states,
        data.transition_matrix,
        data.duration_distribution,
        data.true_change_points,
        data.metadata
        | {
            "kind": "missing_observations",
            "missing_mask": mask,
            "missing_probability": missing_probability,
        },
        seed,
    )


def outliers(
    base: SyntheticDataset | None = None,
    *,
    outlier_probability: float = 0.03,
    magnitude: float = 8.0,
    seed: int | None = None,
) -> SyntheticDataset:
    """Inject additive outliers into a generated dataset."""
    data = gaussian_hmm(seed=seed) if base is None else base
    generator = _rng(seed)
    observations = data.observations.copy()
    mask = cast(NDArray[np.bool_], generator.random(observations.shape[0]) < outlier_probability)
    observations[mask] += generator.choice(
        [-magnitude, magnitude], size=(mask.sum(), observations.shape[1])
    )
    return SyntheticDataset(
        observations,
        data.latent_states,
        data.transition_matrix,
        data.duration_distribution,
        data.true_change_points,
        data.metadata | {"kind": "outliers", "outlier_mask": mask, "magnitude": magnitude},
        seed,
    )


def structural_drift(
    n_steps: int = 200, n_features: int = 1, *, seed: int | None = None
) -> SyntheticDataset:
    """Generate observations with a persistent linear structural drift."""
    generator = _rng(seed)
    drift = np.linspace(0.0, 3.0, n_steps)[:, None]
    observations = np.repeat(drift, n_features, axis=1) + generator.normal(
        0.0, 0.4, (n_steps, n_features)
    )
    return SyntheticDataset(
        observations.astype(np.float64),
        None,
        None,
        None,
        None,
        {"kind": "structural_drift", "drift": drift[:, 0]},
        seed,
    )


def misspecified_state_count_scenarios(
    true_states: int = 3, assumed_states: int = 2, n_steps: int = 200, *, seed: int | None = None
) -> SyntheticDataset:
    """Generate a dataset whose metadata records a deliberately wrong modeled state count."""
    data = gaussian_hmm(n_steps=n_steps, n_states=true_states, seed=seed)
    metadata = data.metadata | {
        "kind": "misspecified_state_count_scenarios",
        "true_states": true_states,
        "assumed_states": assumed_states,
    }
    return SyntheticDataset(
        data.observations,
        data.latent_states,
        data.transition_matrix,
        None,
        data.true_change_points,
        metadata,
        seed,
    )


__all__ = [
    "SyntheticDataset",
    "abrupt_change_points",
    "gaussian_hmm",
    "gradual_transitions",
    "hsmm_non_geometric",
    "markov_switching_ar",
    "missing_observations",
    "misspecified_state_count_scenarios",
    "non_recurring_regimes",
    "outliers",
    "overlapping_emissions",
    "rare_crisis_states",
    "recurring_regimes",
    "structural_drift",
    "student_t_hmm",
    "switching_covariance_matrices",
    "switching_stochastic_volatility",
]
