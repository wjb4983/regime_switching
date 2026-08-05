from __future__ import annotations

import numpy as np
import pytest

from regime.models.econometric import (
    EconometricModelConfig,
    MarkovSwitchingGARCH,
    MarkovSwitchingHAR,
    MarkovSwitchingRegression,
    RegimeSwitchingCorrelation,
    RegimeSwitchingJumpDiffusion,
    SmoothTransitionAutoregression,
    SwitchingStochasticVolatility,
    ThresholdAutoregression,
)


def _series() -> np.ndarray:
    rng = np.random.default_rng(123)
    return np.r_[rng.normal(-0.5, 0.2, 35), rng.normal(0.8, 0.3, 35)]


def _config(**kwargs: object) -> EconometricModelConfig:
    params = {"n_states": 2, "ar_order": 2, "max_iter": 20, "random_seed": 9}
    params.update(kwargs)
    return EconometricModelConfig(**params)


def test_threshold_and_smooth_transition_ar() -> None:
    x = _series()
    tar = ThresholdAutoregression(_config()).fit(x)
    assert len(tar.predict(x)) == len(x) - 2
    assert "thresholds" in tar.params_

    star = SmoothTransitionAutoregression(_config(smoothness=3.0)).fit(x)
    probs = np.asarray(star.predict_proba(x))
    assert probs.shape == (len(x) - 2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_statsmodels_markov_regression_adapter() -> None:
    x = _series()
    model = MarkovSwitchingRegression(_config(max_iter=5, search_reps=0)).fit(x)
    probs = np.asarray(model.predict_proba(x))
    assert probs.shape == (len(x), 2)
    assert np.asarray(model.transition_matrix()).shape == (2, 2)


def test_custom_fragile_estimators_and_placeholder() -> None:
    x = _series()
    assert len(MarkovSwitchingGARCH(_config()).fit(x).predict(x)) == len(x)
    assert (
        len(MarkovSwitchingHAR(_config()).fit(np.abs(x) + 0.01).predict(np.abs(x) + 0.01))
        == len(x) - 21
    )
    assert len(SwitchingStochasticVolatility(_config()).fit(x).predict(x)) == len(x)

    y = np.column_stack([x, x + np.linspace(0, 1, len(x))])
    assert len(RegimeSwitchingCorrelation(_config()).fit(y).predict(y)) == len(y)

    with pytest.raises(NotImplementedError):
        RegimeSwitchingJumpDiffusion(_config()).fit(x)
