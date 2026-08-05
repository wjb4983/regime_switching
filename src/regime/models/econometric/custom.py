"""Lightweight custom econometric estimators kept separate from vendor adapters."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from regime.models.base import ModelMetadata, RegimeModel, RegimeModelConfig
from regime.models.econometric.config import EconometricModelConfig

Array = NDArray[np.float64]
_EPS = 1e-12


def _as_2d(dataset: Any) -> Array:
    x = np.asarray(dataset, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0 or not np.all(np.isfinite(x)):
        raise ValueError("dataset must be a non-empty finite 1D or 2D numeric array")
    return x


def _lagged(y: Array, order: int) -> tuple[Array, Array]:
    if y.shape[0] <= order:
        raise ValueError("dataset length must exceed ar_order")
    target = y[order:, 0]
    design = np.column_stack(
        [np.ones(len(target)), *(y[order - lag : -lag, 0] for lag in range(1, order + 1))]
    )
    return target, design


def _ridge_solve(x: Array, y: Array, regularization: float) -> Array:
    penalty = np.eye(x.shape[1]) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


class _BaseEconometricModel(RegimeModel):
    """Shared persistence and metadata for econometric estimators."""

    def __init__(self, config: EconometricModelConfig | None = None) -> None:
        self.config = config or EconometricModelConfig(model_name=self.__class__.__name__)
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            config_hash=self.config.config_hash(),
        )
        self.labels_: Array | None = None
        self.params_: dict[str, Any] = {}

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        if config is not None:
            self.config = EconometricModelConfig(**config.model_dump())
        labels = self._fit_array(_as_2d(dataset))
        self.labels_ = labels.astype(np.float64)
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            fitted_at=datetime.now(UTC),
            training_observations=len(labels),
            config_hash=self.config.config_hash(),
            attributes=self.state_statistics(),
        )
        return self

    def _fit_array(self, x: Array) -> Array:
        raise NotImplementedError

    def predict(self, dataset: Any) -> Sequence[int]:
        return np.asarray(self.predict_proba(dataset)).argmax(axis=1).tolist()

    def predict_proba(self, dataset: Any) -> Sequence[Sequence[float]]:
        labels = self._predict_labels(_as_2d(dataset)).astype(int)
        probs = np.zeros((len(labels), self.config.n_states))
        probs[np.arange(len(labels)), np.clip(labels, 0, self.config.n_states - 1)] = 1.0
        return probs.tolist()

    def _predict_labels(self, x: Array) -> Array:
        if self.labels_ is None:
            raise ValueError("model must be fitted before inference")
        return self.labels_[-len(x) :]

    def state_statistics(self) -> Mapping[str, Mapping[str, float]]:
        if self.labels_ is None:
            return {}
        return {
            f"state_{state}": {"frequency": float(np.mean(self.labels_ == state))}
            for state in range(self.config.n_states)
        }

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


class ThresholdAutoregression(_BaseEconometricModel):
    """Self-exciting threshold autoregression with one AR fit per threshold regime."""

    def _fit_array(self, x: Array) -> Array:
        y, design = _lagged(x, self.config.ar_order)
        threshold_source = design[:, 1]
        cuts = (
            [self.config.threshold]
            if self.config.threshold is not None
            else np.quantile(threshold_source, np.linspace(0, 1, self.config.n_states + 1)[1:-1])
        )
        labels = np.digitize(threshold_source, cuts).astype(int)
        self.params_["thresholds"] = np.asarray(cuts, dtype=float).tolist()
        self.params_["coefficients"] = [
            _ridge_solve(
                design[labels == state], y[labels == state], self.config.regularization
            ).tolist()
            if np.any(labels == state)
            else np.zeros(design.shape[1]).tolist()
            for state in range(self.config.n_states)
        ]
        return labels.astype(np.float64)

    def _predict_labels(self, x: Array) -> Array:
        _, design = _lagged(x, self.config.ar_order)
        return np.digitize(design[:, 1], self.params_["thresholds"]).astype(np.float64)


class SmoothTransitionAutoregression(ThresholdAutoregression):
    """Logistic smooth-transition AR represented as soft two-regime probabilities."""

    def predict_proba(self, dataset: Any) -> Sequence[Sequence[float]]:
        if self.config.n_states != 2:
            return super().predict_proba(dataset)
        _, design = _lagged(_as_2d(dataset), self.config.ar_order)
        center = float(self.params_.get("thresholds", [0.0])[0])
        p1 = 1.0 / (1.0 + np.exp(-self.config.smoothness * (design[:, 1] - center)))
        return np.column_stack([1.0 - p1, p1]).tolist()


class RegimeSwitchingCorrelation(_BaseEconometricModel):
    """Correlation-regime baseline using rolling absolute average correlation quantiles."""

    def _fit_array(self, x: Array) -> Array:
        if x.shape[1] < 2:
            raise ValueError("correlation regimes require at least two columns")
        levels: list[float] = []
        for i in range(len(x)):
            window = x[max(0, i - 20) : i + 1]
            if len(window) < 2:
                levels.append(0.0)
                continue
            corr = np.nan_to_num(np.corrcoef(window.T), nan=0.0)
            off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
            levels.append(float(np.mean(np.abs(off_diag))))
        corr_level = np.asarray(levels, dtype=np.float64)
        cuts = np.quantile(corr_level, np.linspace(0, 1, self.config.n_states + 1)[1:-1])
        self.params_["thresholds"] = cuts.tolist()
        return np.digitize(corr_level, cuts).astype(np.float64)


class RegimeSwitchingCopula(RegimeSwitchingCorrelation):
    """Copula-regime placeholder using rank-correlation states.

    A dedicated optional copula backend can replace this rank-correlation baseline.
    """


class SwitchingStochasticVolatility(_BaseEconometricModel):
    """Fragile/custom SV approximation based on log-squared-return volatility states."""

    def _fit_array(self, x: Array) -> Array:
        lv = np.log(x[:, 0] ** 2 + _EPS)
        cuts = np.quantile(lv, np.linspace(0, 1, self.config.n_states + 1)[1:-1])
        self.params_["log_variance_thresholds"] = cuts.tolist()
        return np.digitize(lv, cuts).astype(np.float64)


class RegimeSwitchingJumpDiffusion(_BaseEconometricModel):
    """Explicit placeholder for future regime-switching jump-diffusion estimators."""

    def _fit_array(self, x: Array) -> Array:
        raise NotImplementedError(
            "regime-switching jump-diffusion estimation requires a future optional backend"
        )
