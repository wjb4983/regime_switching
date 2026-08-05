"""Inference, persistence, alignment, and numerical contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from regime.evaluation.alignment import align_states
from regime.models.probabilistic import GaussianHMM, ProbabilisticHMMConfig
from regime.synthetic import SyntheticDataset

pytestmark = [pytest.mark.synthetic, pytest.mark.timeout(30)]


def test_label_alignment_undoes_a_known_permutation() -> None:
    reference = {"labels": [0, 1, 2], "state_means": [[-3.0], [0.0], [4.0]]}
    candidate = {"labels": [0, 1, 2], "state_means": [[4.0], [-3.0], [0.0]]}

    alignment = align_states(reference, candidate, "state_means")

    assert alignment.candidate_to_reference == {0: 2, 1: 0, 2: 1}
    assert alignment.aligned_labels([1, 2, 0]) == (0, 1, 2)


def test_online_filtering_is_normalized_and_reacts_to_new_regime(
    known_hmm: SyntheticDataset, fitted_hmm: GaussianHMM
) -> None:
    assert fitted_hmm.means_ is not None
    low_state, high_state = np.argsort(fitted_hmm.means_.mean(axis=1))

    low = fitted_hmm.filter(np.repeat(fitted_hmm.means_[low_state][None, :], 3, axis=0))
    high = fitted_hmm.filter(np.repeat(fitted_hmm.means_[high_state][None, :], 3, axis=0))

    assert sum(low.filtered_probabilities) == pytest.approx(1.0)
    assert sum(high.filtered_probabilities) == pytest.approx(1.0)
    assert high.filtered_probabilities[high_state] > 0.95
    assert high.state == high_state


def test_model_serialization_preserves_inference(
    tmp_path, known_hmm: SyntheticDataset, fitted_hmm: GaussianHMM
) -> None:
    path = tmp_path / "known-state-hmm.pkl"
    before = np.asarray(fitted_hmm.predict_proba(known_hmm.observations[:20]))

    fitted_hmm.save(path)
    restored = GaussianHMM.load(path)

    np.testing.assert_allclose(restored.predict_proba(known_hmm.observations[:20]), before)
    np.testing.assert_allclose(restored.transition_matrix(), fitted_hmm.transition_matrix())
    assert restored.metadata == fitted_hmm.metadata


def test_numerical_stability_for_large_offset_and_near_constant_features() -> None:
    rng = np.random.default_rng(104)
    observations = np.column_stack(
        [
            1e9 + np.r_[rng.normal(-3.0, 0.02, 60), rng.normal(3.0, 0.02, 60)],
            1e-10 + rng.normal(0.0, 1e-12, 120),
        ]
    )
    model = GaussianHMM(
        ProbabilisticHMMConfig(
            model_name="stability",
            n_states=2,
            random_seed=3,
            max_iter=15,
            covariance_regularization=1e-5,
        )
    ).fit(observations)
    probabilities = np.asarray(model.predict_proba(observations))

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    assert np.isfinite(model.log_likelihood_)
