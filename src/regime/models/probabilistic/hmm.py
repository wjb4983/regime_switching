"""Numerically stable probabilistic HMM-family regime models.

The implementations favor small, dependency-light EM estimators with explicit access to
filtered probabilities, smoothed probabilities, transitions, durations, and state summaries.
They are intended as transparent research baselines rather than highly optimized vendor
wrappers.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PositiveInt
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from regime.models.base import ModelMetadata, RegimeInferenceResult, RegimeModel, RegimeModelConfig

_EPS = 1e-12
Array = NDArray[np.float64]


class ProbabilisticHMMConfig(RegimeModelConfig):
    """Configuration for probabilistic latent-state estimators."""

    max_iter: PositiveInt = 50
    tol: float = Field(default=1e-4, gt=0.0)
    n_init: PositiveInt = 1
    covariance_regularization: float = Field(default=1e-6, gt=0.0)
    sticky_strength: float = Field(default=0.0, ge=0.0)
    n_mixtures: PositiveInt = 2
    ar_order: PositiveInt = 1
    student_t_dof: float = Field(default=8.0, gt=2.0)
    duration_mean: float | None = Field(default=None, gt=0.0)
    max_duration: PositiveInt = 50


@dataclass
class _ForwardBackward:
    log_likelihood: float
    gamma: Array
    xi_sum: Array
    filtered: Array
    smoothed: Array


def _as_2d(dataset: Any) -> Array:
    x = np.asarray(dataset, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("dataset must be a non-empty 1D or 2D numeric array")
    if not np.all(np.isfinite(x)):
        raise ValueError("dataset contains NaN or infinite values")
    return x


def _normalize_rows(a: Array) -> Array:
    out = np.maximum(a, _EPS)
    return out / out.sum(axis=1, keepdims=True)


def _entropy(p: Array) -> float:
    p = np.maximum(p, _EPS)
    return float(-(p * np.log(p)).sum())


def _log_gaussian(x: Array, means: Array, covars: Array) -> Array:
    n, d = x.shape
    out = np.empty((n, len(means)), dtype=np.float64)
    eye = np.eye(d)
    for k, (mean, cov) in enumerate(zip(means, covars, strict=True)):
        c = np.asarray(cov, dtype=np.float64) + eye * _EPS
        sign, logdet = np.linalg.slogdet(c)
        if sign <= 0:
            c = c + eye * 1e-6
            _, logdet = np.linalg.slogdet(c)
        diff = x - mean
        solved = np.linalg.solve(c, diff.T).T
        out[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + (diff * solved).sum(axis=1))
    return out


def _weighted_cov(x: Array, weights: Array, mean: Array, reg: float) -> Array:
    denom = max(float(weights.sum()), _EPS)
    diff = x - mean
    return (diff * weights[:, None]).T @ diff / denom + np.eye(x.shape[1]) * reg


class GaussianHMM(RegimeModel):
    """Gaussian-emission HMM fitted with multiple-initialization EM."""

    def __init__(self, config: ProbabilisticHMMConfig | None = None) -> None:
        self.config = config or ProbabilisticHMMConfig(model_name=self.__class__.__name__)
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            config_hash=self.config.config_hash(),
        )
        self.startprob_: Array | None = None
        self.transmat_: Array | None = None
        self.means_: Array | None = None
        self.covars_: Array | None = None
        self.log_likelihood_: float | None = None
        self._filter_probs: Array | None = None
        self._last_filtered: Array | None = None

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        if config is not None:
            self.config = ProbabilisticHMMConfig(**config.model_dump())
        x = self._transform_fit_data(_as_2d(dataset))
        best: tuple[float, Array, Array, Array, Array] | None = None
        base_seed = self.config.random_seed
        for init in range(self.config.n_init):
            rng = np.random.default_rng(None if base_seed is None else base_seed + init)
            start, trans, means, covars = self._initial_parameters(x, rng)
            prev = -np.inf
            for _ in range(self.config.max_iter):
                fb = self._forward_backward_from_params(x, start, trans, means, covars)
                start, trans, means, covars = self._m_step(x, fb.gamma, fb.xi_sum, means, covars)
                if abs(fb.log_likelihood - prev) < self.config.tol:
                    break
                prev = fb.log_likelihood
            fb = self._forward_backward_from_params(x, start, trans, means, covars)
            if best is None or fb.log_likelihood > best[0]:
                best = (fb.log_likelihood, start, trans, means, covars)
        assert best is not None
        self.log_likelihood_, self.startprob_, self.transmat_, self.means_, self.covars_ = best
        fb = self._forward_backward_from_params(
            x, self.startprob_, self.transmat_, self.means_, self.covars_
        )
        self._filter_probs = fb.filtered
        self._last_filtered = fb.filtered[-1]
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            fitted_at=datetime.now(UTC),
            training_observations=len(x),
            config_hash=self.config.config_hash(),
            attributes={"log_likelihood": float(self.log_likelihood_)},
        )
        return self

    def _transform_fit_data(self, x: Array) -> Array:
        return x

    def _initial_parameters(
        self, x: Array, rng: np.random.Generator
    ) -> tuple[Array, Array, Array, Array]:
        k = self.config.n_states
        labels = KMeans(
            n_clusters=k, n_init=1, random_state=int(rng.integers(0, 2**31 - 1))
        ).fit_predict(x)
        means = np.vstack(
            [
                x[labels == j].mean(axis=0) if np.any(labels == j) else x[rng.integers(len(x))]
                for j in range(k)
            ]
        )
        covars = np.stack(
            [
                _weighted_cov(
                    x, (labels == j).astype(float), means[j], self.config.covariance_regularization
                )
                for j in range(k)
            ]
        )
        start = np.full(k, 1.0 / k)
        trans = np.full((k, k), 1.0 / k) + np.eye(k) * self.config.sticky_strength
        return start, _normalize_rows(trans), means, covars

    def _log_emission(self, x: Array, means: Array, covars: Array) -> Array:
        return _log_gaussian(x, means, covars)

    def _forward_backward_from_params(
        self, x: Array, start: Array, trans: Array, means: Array, covars: Array
    ) -> _ForwardBackward:
        log_b = self._log_emission(x, means, covars)
        log_start = np.log(np.maximum(start, _EPS))
        log_trans = np.log(np.maximum(trans, _EPS))
        n, k = log_b.shape
        la = np.empty((n, k))
        lb = np.zeros((n, k))
        la[0] = log_start + log_b[0]
        for t in range(1, n):
            la[t] = log_b[t] + logsumexp(la[t - 1][:, None] + log_trans, axis=0)
        ll = float(logsumexp(la[-1]))
        for t in range(n - 2, -1, -1):
            lb[t] = logsumexp(log_trans + log_b[t + 1] + lb[t + 1], axis=1)
        gamma = np.exp(la + lb - ll)
        filtered = np.exp(la - logsumexp(la, axis=1, keepdims=True))
        xi = np.zeros((k, k))
        for t in range(n - 1):
            log_xi = la[t][:, None] + log_trans + log_b[t + 1] + lb[t + 1] - ll
            xi += np.exp(log_xi)
        return _ForwardBackward(ll, gamma / gamma.sum(axis=1, keepdims=True), xi, filtered, gamma)

    def _m_step(
        self, x: Array, gamma: Array, xi_sum: Array, means: Array, covars: Array
    ) -> tuple[Array, Array, Array, Array]:
        weights = gamma.sum(axis=0) + _EPS
        means = gamma.T @ x / weights[:, None]
        covars = np.stack(
            [
                _weighted_cov(x, gamma[:, j], means[j], self.config.covariance_regularization)
                for j in range(self.config.n_states)
            ]
        )
        trans = xi_sum + np.eye(self.config.n_states) * self.config.sticky_strength + _EPS
        return gamma[0] + _EPS, _normalize_rows(trans), means, covars

    def predict(self, dataset: Any) -> Sequence[int]:
        return np.asarray(self.predict_proba(dataset)).argmax(axis=1).tolist()

    def predict_proba(self, dataset: Any) -> Sequence[Sequence[float]]:
        self._require_fit()
        x = self._transform_fit_data(_as_2d(dataset))
        fb = self._forward_backward_from_params(
            x, self.startprob_, self.transmat_, self.means_, self.covars_
        )  # type: ignore[arg-type]
        return fb.filtered.tolist()

    def smooth(self, dataset: Any) -> Sequence[RegimeInferenceResult]:
        self._require_fit()
        x = self._transform_fit_data(_as_2d(dataset))
        fb = self._forward_backward_from_params(
            x, self.startprob_, self.transmat_, self.means_, self.covars_
        )  # type: ignore[arg-type]
        return [self._result(row, smoothed=fb.smoothed[i]) for i, row in enumerate(fb.filtered)]

    def filter(self, observation: Any) -> RegimeInferenceResult:
        self._require_fit()
        x = self._transform_fit_data(_as_2d(observation))
        probs = self._last_filtered if self._last_filtered is not None else self.startprob_
        out = probs.copy()  # type: ignore[union-attr]
        for row in x:
            logp = (
                np.log(np.maximum(out @ self.transmat_, _EPS))
                + self._log_emission(row[None, :], self.means_, self.covars_)[0]
            )  # type: ignore[arg-type,operator]
            out = np.exp(logp - logsumexp(logp))
        self._last_filtered = out
        return self._result(out)

    def transition_matrix(self) -> Sequence[Sequence[float]]:
        self._require_fit()
        return self.transmat_.tolist()  # type: ignore[union-attr]

    def expected_durations(self) -> Sequence[float]:
        self._require_fit()
        return (1.0 / np.maximum(1.0 - np.diag(self.transmat_), _EPS)).tolist()  # type: ignore[arg-type]

    def state_statistics(self) -> Mapping[str, Mapping[str, float]]:
        self._require_fit()
        return {
            f"state_{i}": {
                "mean": float(np.mean(self.means_[i])),
                "variance": float(np.mean(np.diag(self.covars_[i]))),
                "expected_duration": float(self.expected_durations()[i]),
            }
            for i in range(self.config.n_states)
        }  # type: ignore[index]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    def _result(self, probs: Array, smoothed: Array | None = None) -> RegimeInferenceResult:
        state = int(np.argmax(probs))
        change = 0.0 if self._last_filtered is None else float(1.0 - self._last_filtered[state])
        return RegimeInferenceResult(
            state=state,
            filtered_probabilities=tuple(map(float, probs)),
            smoothed_probabilities=None if smoothed is None else tuple(map(float, smoothed)),
            change_probability=max(0.0, min(1.0, change)),
            expected_regime_duration=float(self.expected_durations()[state]),
            transition_matrix=tuple(tuple(map(float, r)) for r in self.transition_matrix()),
            entropy=_entropy(probs),
            confidence=float(np.max(probs)),
            state_statistics=self.state_statistics(),
            model_version=self.config.model_version,
            configuration_hash=self.config.config_hash(),
        )

    def _require_fit(self) -> None:
        if (
            self.startprob_ is None
            or self.transmat_ is None
            or self.means_ is None
            or self.covars_ is None
        ):
            raise ValueError("model must be fitted before inference")


class StickyHMM(GaussianHMM):
    """Gaussian HMM with self-transition pseudo-counts."""

    def __init__(self, config: ProbabilisticHMMConfig | None = None) -> None:
        super().__init__(
            config or ProbabilisticHMMConfig(model_name="StickyHMM", sticky_strength=10.0)
        )


class StudentTHMM(GaussianHMM):
    """Student-t emission HMM with fixed degrees of freedom."""

    def _log_emission(self, x: Array, means: Array, covars: Array) -> Array:
        n, d = x.shape
        nu = self.config.student_t_dof
        out = np.empty((n, len(means)))
        for k, (mean, cov) in enumerate(zip(means, covars, strict=True)):
            c = cov + np.eye(d) * _EPS
            _, logdet = np.linalg.slogdet(c)
            diff = x - mean
            q = (diff * np.linalg.solve(c, diff.T).T).sum(axis=1)
            out[:, k] = (
                gammaln((nu + d) / 2)
                - gammaln(nu / 2)
                - 0.5 * (d * np.log(nu * np.pi) + logdet)
                - ((nu + d) / 2) * np.log1p(q / nu)
            )
        return out


class ARHMM(GaussianHMM):
    """Autoregressive HMM using lagged features and Gaussian residual emissions."""

    def _transform_fit_data(self, x: Array) -> Array:
        p = self.config.ar_order
        if len(x) <= p:
            raise ValueError("dataset length must exceed ar_order")
        return np.hstack([x[p:], *(x[p - lag : len(x) - lag] for lag in range(1, p + 1))])


class InputOutputHMM(GaussianHMM):
    """Input-output HMM baseline that conditions emissions on concatenated inputs and outputs."""

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        y, u = dataset if isinstance(dataset, tuple) and len(dataset) == 2 else (dataset, None)
        x = _as_2d(y) if u is None else np.hstack([_as_2d(y), _as_2d(u)])
        return super().fit(x, config)


class GMMHMM(GaussianHMM):
    """GMM-emission HMM initialized from a Gaussian mixture adapter."""

    def _initial_parameters(
        self, x: Array, rng: np.random.Generator
    ) -> tuple[Array, Array, Array, Array]:
        gm = GaussianMixture(
            n_components=self.config.n_states,
            n_init=1,
            random_state=int(rng.integers(0, 2**31 - 1)),
            reg_covar=self.config.covariance_regularization,
        ).fit(x)
        trans = np.full((self.config.n_states, self.config.n_states), 1.0 / self.config.n_states)
        return gm.weights_ + _EPS, trans, gm.means_, gm.covariances_


class HSMM(GaussianHMM):
    """Hidden semi-Markov approximation with explicit geometric/Poisson duration priors."""

    def _m_step(
        self, x: Array, gamma: Array, xi_sum: Array, means: Array, covars: Array
    ) -> tuple[Array, Array, Array, Array]:
        start, trans, means, covars = super()._m_step(x, gamma, xi_sum, means, covars)
        if self.config.duration_mean is not None:
            stay = 1.0 - 1.0 / self.config.duration_mean
            np.fill_diagonal(trans, np.clip(stay, _EPS, 1.0 - _EPS))
            trans = _normalize_rows(trans)
        return start, trans, means, covars

    def duration_pmf(self) -> Sequence[Sequence[float]]:
        mean = np.asarray(self.expected_durations())[:, None]
        d = np.arange(1, self.config.max_duration + 1)[None, :]
        p = (1.0 / np.maximum(mean, 1.0)) * (1.0 - 1.0 / np.maximum(mean, 1.0)) ** (d - 1)
        return (p / p.sum(axis=1, keepdims=True)).tolist()


class ExplicitDurationLatentStateModel(HSMM):
    """Alias-style explicit-duration latent-state model with HSMM duration exposure."""


class HDPHMMAdapter(GaussianHMM):
    """Placeholder adapter for future Bayesian nonparametric HDP-HMM integrations."""

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        raise NotImplementedError(
            "HDP-HMM requires an optional Bayesian backend that is not installed"
        )
