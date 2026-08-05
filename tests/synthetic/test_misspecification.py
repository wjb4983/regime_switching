"""Synthetic robustness checks under deliberate model misspecification."""

from __future__ import annotations

import numpy as np
import pytest

from regime.models.probabilistic import GaussianHMM, ProbabilisticHMMConfig, StudentTHMM
from regime.synthetic import outliers, student_t_hmm

pytestmark = [pytest.mark.synthetic, pytest.mark.timeout(30)]


@pytest.mark.parametrize("model_type", [GaussianHMM, StudentTHMM])
def test_heavy_tails_and_outliers_still_produce_valid_probabilities(model_type) -> None:
    data = outliers(student_t_hmm(n_steps=180, n_states=2, df=4.0, seed=44), seed=45)
    config = ProbabilisticHMMConfig(
        model_name="misspecified_emissions",
        n_states=2,
        random_seed=9,
        max_iter=20,
        student_t_dof=4.0,
    )
    probabilities = np.asarray(
        model_type(config).fit(data.observations).predict_proba(data.observations)
    )

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    assert np.mean(np.max(probabilities, axis=1)) > 0.6


def test_wrong_state_count_fails_gracefully_without_invalid_output() -> None:
    data = student_t_hmm(n_steps=160, n_states=3, df=5.0, seed=73)
    model = GaussianHMM(
        ProbabilisticHMMConfig(
            model_name="wrong_state_count", n_states=2, random_seed=5, max_iter=20
        )
    ).fit(data.observations)
    probabilities = np.asarray(model.predict_proba(data.observations))

    assert probabilities.shape == (160, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(np.asarray(model.transition_matrix()).sum(axis=1), 1.0)
