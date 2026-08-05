"""Experimental CPU reference implementations of switching state-space models.

These models are prototypes until parameter and state recovery are validated on a
documented synthetic benchmark.  They use an interacting-multiple-model (IMM)
switching Kalman filter and an approximate discrete-state backward smoother.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PositiveInt
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from regime.models.base import ModelMetadata, RegimeModel, RegimeModelConfig

Array = NDArray[np.float64]
_EPS = 1e-10


class StateSpaceConfig(RegimeModelConfig):
    """Configuration shared by the experimental switching estimators."""

    model_name: str = "switching_linear_dynamical_system"
    state_dim: PositiveInt = 1
    max_iter: PositiveInt = 10
    covariance_regularization: float = Field(default=1e-5, gt=0)
    transition_regularization: float = Field(default=1.0, ge=0)
    max_duration: PositiveInt = 50


@dataclass(frozen=True)
class StateSpaceParameters:
    """Host-side parameters for a regime-dependent linear Gaussian system."""

    transition_matrix: Array
    dynamics: Array
    observation: Array
    process_covariance: Array
    observation_covariance: Array
    initial_mean: Array
    initial_covariance: Array


@dataclass(frozen=True)
class StateSpaceResult:
    """Complete batch inference output."""

    filtered_probabilities: Array
    smoothed_probabilities: Array
    filtered_state_means: Array
    filtered_state_covariances: Array
    smoothed_state_means: Array
    log_likelihood: float
    numerical_diagnostics: Mapping[str, float | int | bool | str]


def _data(value: Any) -> Array:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or len(out) == 0 or not np.isfinite(out).all():
        raise ValueError("observations must be a non-empty, finite 1D or 2D array")
    return out


def _normal_logpdf(value: Array, mean: Array, covariance: Array) -> float:
    covariance = (covariance + covariance.T) / 2
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("innovation covariance is not positive definite")
    residual = value - mean
    return float(
        -0.5
        * (
            len(value) * np.log(2 * np.pi)
            + logdet
            + residual @ np.linalg.solve(covariance, residual)
        )
    )


class SwitchingKalmanFilter:
    """IMM switching Kalman filter usable independently of an estimator."""

    def __init__(self, parameters: StateSpaceParameters) -> None:
        self.parameters = parameters

    def filter(
        self, observations: Any, transition_sequence: Array | None = None
    ) -> StateSpaceResult:
        y = _data(observations)
        p = self.parameters
        regimes, state_dim = p.dynamics.shape[:2]
        means = np.broadcast_to(p.initial_mean, (regimes, state_dim)).copy()
        covariances = np.broadcast_to(p.initial_covariance, (regimes, state_dim, state_dim)).copy()
        probabilities = np.full(regimes, 1 / regimes)
        filtered = np.empty((len(y), regimes))
        state_means = np.empty((len(y), regimes, state_dim))
        state_covariances = np.empty((len(y), regimes, state_dim, state_dim))
        log_likelihood = 0.0
        jitter_count = 0
        min_innovation_eigenvalue = np.inf
        transitions = []
        for t, observation in enumerate(y):
            trans = p.transition_matrix if transition_sequence is None else transition_sequence[t]
            transitions.append(trans)
            predicted_probabilities = probabilities @ trans
            mixed_means = np.empty_like(means)
            mixed_covariances = np.empty_like(covariances)
            for j in range(regimes):
                weights = probabilities * trans[:, j] / max(predicted_probabilities[j], _EPS)
                mixed_means[j] = weights @ means
                mixed_covariances[j] = sum(
                    weights[i]
                    * (
                        covariances[i]
                        + np.outer(means[i] - mixed_means[j], means[i] - mixed_means[j])
                    )
                    for i in range(regimes)
                )
            log_weights = np.empty(regimes)
            for j in range(regimes):
                predicted_mean = p.dynamics[j] @ mixed_means[j]
                predicted_covariance = (
                    p.dynamics[j] @ mixed_covariances[j] @ p.dynamics[j].T + p.process_covariance[j]
                )
                innovation_covariance = (
                    p.observation[j] @ predicted_covariance @ p.observation[j].T
                    + p.observation_covariance[j]
                )
                eigenvalue = float(np.linalg.eigvalsh(innovation_covariance).min())
                min_innovation_eigenvalue = min(min_innovation_eigenvalue, eigenvalue)
                if eigenvalue < _EPS:
                    innovation_covariance += np.eye(len(observation)) * (_EPS - eigenvalue)
                    jitter_count += 1
                gain = np.linalg.solve(
                    innovation_covariance, p.observation[j] @ predicted_covariance
                ).T
                innovation = observation - p.observation[j] @ predicted_mean
                means[j] = predicted_mean + gain @ innovation
                covariances[j] = (
                    predicted_covariance - gain @ p.observation[j] @ predicted_covariance
                )
                covariances[j] = (covariances[j] + covariances[j].T) / 2
                log_weights[j] = np.log(max(predicted_probabilities[j], _EPS)) + _normal_logpdf(
                    observation, p.observation[j] @ predicted_mean, innovation_covariance
                )
            increment = float(logsumexp(log_weights))
            log_likelihood += increment
            probabilities = np.exp(log_weights - increment)
            filtered[t], state_means[t], state_covariances[t] = probabilities, means, covariances
        smoothed = filtered.copy()
        for t in range(len(y) - 2, -1, -1):
            prediction = filtered[t] @ transitions[t + 1]
            smoothed[t] = filtered[t] * (
                transitions[t + 1] @ (smoothed[t + 1] / np.maximum(prediction, _EPS))
            )
            smoothed[t] /= smoothed[t].sum()
        smoothed_means = np.einsum("tk,tkd->td", smoothed, state_means)
        diagnostics: dict[str, float | int | bool | str] = {
            "algorithm": "IMM approximate smoother",
            "log_likelihood": log_likelihood,
            "jitter_count": jitter_count,
            "minimum_innovation_eigenvalue": float(min_innovation_eigenvalue),
            "probabilities_normalized": bool(np.allclose(filtered.sum(axis=1), 1)),
        }
        return StateSpaceResult(
            filtered,
            smoothed,
            state_means,
            state_covariances,
            smoothed_means,
            log_likelihood,
            diagnostics,
        )


class SwitchingLinearDynamicalSystem(RegimeModel):
    """EXPERIMENTAL switching LDS with a CPU IMM inference implementation."""

    experimental = True

    def __init__(self, config: StateSpaceConfig | None = None) -> None:
        self.config = config or StateSpaceConfig()
        self.parameters_: StateSpaceParameters | None = None
        self.result_: StateSpaceResult | None = None
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            config_hash=self.config.config_hash(),
            attributes={"experimental": True, "validation": "synthetic recovery pending"},
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def state_space_parameters(self) -> StateSpaceParameters:
        if self.parameters_ is None:
            raise ValueError("model must be fitted before parameters are available")
        return self.parameters_

    @property
    def numerical_diagnostics(self) -> Mapping[str, float | int | bool | str]:
        return {} if self.result_ is None else self.result_.numerical_diagnostics

    def _fit_observation_projection(self, observations: Array) -> tuple[Array, Array]:
        state_dim = self.config.state_dim
        observation_dim = observations.shape[1]
        observation = np.zeros((self.config.n_states, observation_dim, state_dim))
        observation[:, : min(state_dim, observation_dim), : min(state_dim, observation_dim)] = (
            np.eye(min(state_dim, observation_dim))
        )
        latent = observations[:, :state_dim]
        if latent.shape[1] < state_dim:
            latent = np.pad(latent, ((0, 0), (0, state_dim - latent.shape[1])))
        return observation, latent

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        if config is not None:
            self.config = StateSpaceConfig(**config.model_dump())
        y = _data(dataset)
        k, d, reg = (
            self.config.n_states,
            self.config.state_dim,
            self.config.covariance_regularization,
        )
        observation, latent = self._fit_observation_projection(y)
        labels = KMeans(n_clusters=k, n_init=10, random_state=self.config.random_seed).fit_predict(
            y
        )
        transition_counts = np.full((k, k), self.config.transition_regularization)
        np.add.at(transition_counts, (labels[:-1], labels[1:]), 1)
        transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        dynamics = np.zeros((k, d, d))
        process = np.empty((k, d, d))
        noise = np.empty((k, y.shape[1], y.shape[1]))
        for regime in range(k):
            mask = labels[1:] == regime
            x0, x1 = latent[:-1][mask], latent[1:][mask]
            dynamics[regime] = (
                np.linalg.lstsq(x0, x1, rcond=None)[0].T if len(x0) >= d else np.eye(d)
            )
            state_residual = x1 - x0 @ dynamics[regime].T
            process[regime] = (
                state_residual.T @ state_residual / max(len(state_residual), 1)
            ) + np.eye(d) * reg
            residual = y[labels == regime] - latent[labels == regime] @ observation[regime].T
            noise[regime] = (residual.T @ residual / max(len(residual), 1)) + np.eye(
                y.shape[1]
            ) * reg
        self.parameters_ = StateSpaceParameters(
            transition, dynamics, observation, process, noise, latent[0], np.eye(d)
        )
        self.result_ = self._run_filter(y)
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=k,
            fitted_at=datetime.now(UTC),
            training_observations=len(y),
            config_hash=self.config.config_hash(),
            attributes={
                "experimental": True,
                "validation": "synthetic recovery pending",
                "log_likelihood": self.result_.log_likelihood,
            },
        )
        return self

    def _transition_sequence(self, observations: Array) -> Array | None:
        return None

    def _run_filter(self, observations: Array) -> StateSpaceResult:
        return SwitchingKalmanFilter(self.state_space_parameters).filter(
            observations, self._transition_sequence(observations)
        )

    def infer(self, dataset: Any) -> StateSpaceResult:
        return self._run_filter(_data(dataset))

    def predict(self, dataset: Any) -> Sequence[int]:
        return self.infer(dataset).filtered_probabilities.argmax(axis=1).tolist()

    def predict_proba(self, dataset: Any) -> Sequence[Sequence[float]]:
        return self.infer(dataset).filtered_probabilities.tolist()

    def smooth(self, dataset: Any) -> StateSpaceResult:  # type: ignore[override]
        return self.infer(dataset)

    def transition_matrix(self) -> Sequence[Sequence[float]]:
        return self.state_space_parameters.transition_matrix.tolist()

    def state_statistics(self) -> Mapping[str, Mapping[str, float]]:
        p = self.state_space_parameters
        return {
            f"state_{i}": {
                "process_variance": float(np.trace(p.process_covariance[i])),
                "observation_variance": float(np.trace(p.observation_covariance[i])),
            }
            for i in range(self.config.n_states)
        }

    def save(self, path: str | Path) -> None:
        p = self.state_space_parameters
        payload = {
            "class": type(self).__name__,
            "config": self.config.model_dump(mode="json"),
            "parameters": {name: getattr(p, name).tolist() for name in p.__dataclass_fields__},
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["class"] != cls.__name__:
            raise TypeError(f"serialized model is {payload['class']}, not {cls.__name__}")
        model = cls(StateSpaceConfig(**payload["config"]))
        model.parameters_ = StateSpaceParameters(
            **{key: np.asarray(value, dtype=float) for key, value in payload["parameters"].items()}
        )
        return model


class SwitchingDynamicFactorModel(SwitchingLinearDynamicalSystem):
    """EXPERIMENTAL switching LDS using PCA factor observations."""

    def _fit_observation_projection(self, observations: Array) -> tuple[Array, Array]:
        centered = observations - observations.mean(axis=0)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        loadings = right[: self.config.state_dim].T
        latent = centered @ loadings
        observation = np.broadcast_to(loadings, (self.config.n_states, *loadings.shape)).copy()
        return observation, latent


class RecurrentSwitchingLinearDynamicalSystem(SwitchingLinearDynamicalSystem):
    """EXPERIMENTAL recurrent SLDS with observation-conditioned transitions."""

    recurrent_weights_: Array | None = None

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        result = super().fit(dataset, config)
        self.recurrent_weights_ = np.zeros((self.config.n_states, _data(dataset).shape[1]))
        return result

    def _transition_sequence(self, observations: Array) -> Array | None:
        if self.recurrent_weights_ is None:
            return None
        logits = observations @ self.recurrent_weights_.T
        modulation = np.exp(logits - logits.max(axis=1, keepdims=True))
        sequence = (
            self.state_space_parameters.transition_matrix[None, :, :] * modulation[:, None, :]
        )
        return sequence / sequence.sum(axis=2, keepdims=True)


class ExplicitDurationSwitchingLinearDynamicalSystem(SwitchingLinearDynamicalSystem):
    """EXPERIMENTAL duration-aware SLDS using truncated explicit duration hazards."""

    duration_probabilities_: Array | None = None

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        result = super().fit(dataset, config)
        labels = np.asarray(self.predict(dataset))
        counts = np.ones((self.config.n_states, self.config.max_duration)) * _EPS
        start = 0
        for end in range(1, len(labels) + 1):
            if end == len(labels) or labels[end] != labels[start]:
                counts[labels[start], min(end - start, self.config.max_duration) - 1] += 1
                start = end
        self.duration_probabilities_ = counts / counts.sum(axis=1, keepdims=True)
        return result

    @property
    def expected_durations(self) -> Array:
        if self.duration_probabilities_ is None:
            raise ValueError("model must be fitted before durations are available")
        return self.duration_probabilities_ @ np.arange(1, self.config.max_duration + 1)
