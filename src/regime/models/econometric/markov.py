"""Statsmodels-backed Markov-switching econometric adapters."""

from __future__ import annotations

import importlib
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from regime.models.base import RegimeModelConfig
from regime.models.econometric.config import EconometricModelConfig
from regime.models.econometric.custom import _as_2d, _BaseEconometricModel
from regime.models.probabilistic import GaussianHMM, ProbabilisticHMMConfig


def _to_hmm_config(config: EconometricModelConfig) -> ProbabilisticHMMConfig:
    return ProbabilisticHMMConfig(
        model_name=config.model_name,
        n_states=config.n_states,
        random_seed=config.random_seed,
        ar_order=config.ar_order,
        max_iter=config.max_iter,
        tol=config.tol,
        covariance_regularization=max(config.regularization, 1e-9),
    )


Array = NDArray[np.float64]


class MarkovSwitchingRegression(_BaseEconometricModel):
    """Adapter around :class:`statsmodels.tsa.regime_switching.MarkovRegression`."""

    def __init__(self, config: EconometricModelConfig | None = None) -> None:
        super().__init__(config)
        self.result_: Any | None = None

    def _fit_array(self, x: Array) -> Array:
        sm = importlib.import_module("statsmodels.tsa.regime_switching.markov_regression")
        y = x[:, 0]
        exog = x[:, 1:] if x.shape[1] > 1 else None
        model = sm.MarkovRegression(
            y,
            k_regimes=self.config.n_states,
            exog=exog,
            switching_exog=self.config.exog_switching,
            switching_variance=self.config.switching_variance,
        )
        self.result_ = model.fit(
            method=self.config.optimizer,
            maxiter=self.config.max_iter,
            disp=False,
            search_reps=self.config.search_reps,
            cov_type=self.config.covariance_type,
            **self.config.optimizer_options,
        )
        return np.asarray(self.result_.smoothed_marginal_probabilities).argmax(axis=1)

    def predict_proba(self, dataset: Any) -> list[list[float]]:
        if self.result_ is None:
            raise ValueError("model must be fitted before inference")
        return np.asarray(self.result_.smoothed_marginal_probabilities).tolist()

    def transition_matrix(self) -> list[list[float]]:
        if self.result_ is None:
            raise ValueError("model must be fitted before inference")
        return np.asarray(self.result_.regime_transition).squeeze().tolist()


class MarkovSwitchingAR(MarkovSwitchingRegression):
    """Statsmodels Markov autoregression adapter."""

    def _fit_array(self, x: Array) -> Array:
        sm = importlib.import_module("statsmodels.tsa.regime_switching.markov_autoregression")
        model = sm.MarkovAutoregression(
            x[:, 0],
            k_regimes=self.config.n_states,
            order=self.config.ar_order,
            switching_ar=True,
            switching_variance=self.config.switching_variance,
        )
        self.result_ = model.fit(
            method=self.config.optimizer,
            maxiter=self.config.max_iter,
            disp=False,
            search_reps=self.config.search_reps,
            cov_type=self.config.covariance_type,
            **self.config.optimizer_options,
        )
        return np.asarray(self.result_.smoothed_marginal_probabilities).argmax(axis=1)


class MarkovSwitchingVAR(GaussianHMM):
    """VAR-regime baseline: Gaussian HMM over VAR residual and lagged-feature vectors."""

    def __init__(self, config: EconometricModelConfig | None = None) -> None:
        super().__init__(
            _to_hmm_config(config or EconometricModelConfig(model_name=self.__class__.__name__))
        )

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        if config is not None:
            self.config = _to_hmm_config(EconometricModelConfig(**config.model_dump()))
        return super().fit(dataset)


class MarkovSwitchingGARCH(GaussianHMM):
    """Optional-backend boundary for MSGARCH, with HMM volatility-feature fallback."""

    @staticmethod
    def _volatility_features(dataset: Any) -> Array:
        x = _as_2d(dataset)[:, 0]
        return np.column_stack([x, x * x, np.abs(x)])

    def __init__(self, config: EconometricModelConfig | None = None) -> None:
        super().__init__(
            _to_hmm_config(config or EconometricModelConfig(model_name=self.__class__.__name__))
        )

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        return super().fit(self._volatility_features(dataset), config)

    def predict(self, dataset: Any) -> list[int]:
        return np.asarray(self.predict_proba(dataset)).argmax(axis=1).tolist()

    def predict_proba(self, dataset: Any) -> list[list[float]]:
        return list(GaussianHMM.predict_proba(self, self._volatility_features(dataset)))


class MarkovSwitchingHAR(GaussianHMM):
    """Markov-switching HAR baseline for realized volatility features."""

    @staticmethod
    def _har_features(dataset: Any) -> Array:
        rv = np.maximum(_as_2d(dataset)[:, 0], 0.0)
        if len(rv) < 22:
            raise ValueError("HAR realized-volatility model requires at least 22 observations")
        weekly = np.convolve(rv, np.ones(5) / 5, mode="valid")
        monthly = np.convolve(rv, np.ones(22) / 22, mode="valid")
        return np.column_stack([rv[21:], weekly[17:], monthly])

    def __init__(self, config: EconometricModelConfig | None = None) -> None:
        super().__init__(
            _to_hmm_config(config or EconometricModelConfig(model_name=self.__class__.__name__))
        )

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:  # type: ignore[override]
        return super().fit(self._har_features(dataset), config)

    def predict(self, dataset: Any) -> list[int]:
        return np.asarray(self.predict_proba(dataset)).argmax(axis=1).tolist()

    def predict_proba(self, dataset: Any) -> list[list[float]]:
        return list(GaussianHMM.predict_proba(self, self._har_features(dataset)))
