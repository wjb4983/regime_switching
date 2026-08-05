from __future__ import annotations

import numpy as np
import pytest

from regime.models.probabilistic import (
    ARHMM,
    GMMHMM,
    HSMM,
    GaussianHMM,
    HDPHMMAdapter,
    ProbabilisticHMMConfig,
    StudentTHMM,
)


def _sample() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.r_[rng.normal(-2.0, 0.3, size=25), rng.normal(2.0, 0.3, size=25)][:, None]


def _config(**kwargs: object) -> ProbabilisticHMMConfig:
    return ProbabilisticHMMConfig(
        n_states=2,
        random_seed=11,
        n_init=2,
        max_iter=8,
        model_name="test_hmm",
        **kwargs,
    )


def test_gaussian_hmm_fit_filter_smooth_and_serialize(tmp_path) -> None:
    model = GaussianHMM(_config()).fit(_sample())

    proba = np.asarray(model.predict_proba(_sample()))
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.asarray(model.transition_matrix()).shape == (2, 2)
    assert len(model.expected_durations()) == 2
    assert "state_0" in model.state_statistics()

    result = model.filter([[1.5]])
    assert result.confidence <= 1.0
    assert result.entropy >= 0.0
    smoothed = model.smooth(_sample()[:5])
    assert len(smoothed) == 5
    assert smoothed[0].smoothed_probabilities is not None

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = GaussianHMM.load(path)
    assert loaded.metadata.n_states == 2


def test_student_t_ar_gmm_and_hsmm_variants() -> None:
    x = _sample()
    assert len(StudentTHMM(_config(student_t_dof=5.0)).fit(x).predict(x)) == len(x)
    assert len(GMMHMM(_config()).fit(x).predict(x)) == len(x)

    ar = ARHMM(_config(ar_order=2)).fit(x)
    assert np.asarray(ar.predict_proba(x)).shape == (len(x) - 2, 2)

    hsmm = HSMM(_config(duration_mean=4.0, max_duration=6)).fit(x)
    assert np.asarray(hsmm.duration_pmf()).shape == (2, 6)


def test_hdp_hmm_placeholder_is_explicit() -> None:
    with pytest.raises(NotImplementedError):
        HDPHMMAdapter(_config()).fit(_sample())
