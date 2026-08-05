"""Known-truth recovery checks for latent-state and boundary estimators."""

from __future__ import annotations

import numpy as np
import pytest

from regime.models.changepoint.detectors import (
    BinarySegmentationConfig,
    BinarySegmentationDetector,
)
from regime.models.probabilistic import GaussianHMM, ProbabilisticHMMConfig
from regime.synthetic import SyntheticDataset, abrupt_change_points, gaussian_hmm

from .conftest import align_model_labels, mapped_predictions, run_lengths

pytestmark = [pytest.mark.synthetic, pytest.mark.timeout(30)]


def test_recovers_known_latent_states(known_hmm: SyntheticDataset, fitted_hmm: GaussianHMM) -> None:
    assert known_hmm.latent_states is not None
    recovered = mapped_predictions(fitted_hmm, known_hmm)

    assert np.mean(recovered == known_hmm.latent_states) > 0.86


def test_recovers_abrupt_change_points_within_tolerance() -> None:
    data = abrupt_change_points(n_steps=180, change_points=(55, 120), seed=91)
    detector = BinarySegmentationDetector(
        BinarySegmentationConfig(min_size=15, max_breakpoints=2, threshold=8.0, tolerance=4)
    )
    result = detector.detect(data.observations, ground_truth=data.true_change_points)

    assert len(result.boundary_indices) == 2
    assert result.detection_delays is not None
    assert max(abs(delay) for delay in result.detection_delays) <= 4


def test_filtered_probabilities_are_calibrated(
    known_hmm: SyntheticDataset, fitted_hmm: GaussianHMM
) -> None:
    assert known_hmm.latent_states is not None
    mapping = align_model_labels(fitted_hmm, known_hmm)
    raw = np.asarray(fitted_hmm.predict_proba(known_hmm.observations))
    probability_state_one = raw[:, next(k for k, value in mapping.items() if value == 1)]
    truth = (known_hmm.latent_states == 1).astype(float)
    brier = np.mean((probability_state_one - truth) ** 2)

    assert brier < 0.12
    assert np.allclose(raw.sum(axis=1), 1.0)


def test_recovers_state_durations_and_transition_matrix(
    known_hmm: SyntheticDataset, fitted_hmm: GaussianHMM
) -> None:
    assert known_hmm.latent_states is not None
    assert known_hmm.transition_matrix is not None
    mapping = align_model_labels(fitted_hmm, known_hmm)
    estimated_transition = np.asarray(fitted_hmm.transition_matrix())
    estimated_duration = np.asarray(fitted_hmm.expected_durations())

    for fitted_state, true_state in mapping.items():
        empirical_duration = run_lengths(known_hmm.latent_states, true_state).mean()
        assert estimated_duration[fitted_state] == pytest.approx(empirical_duration, rel=0.45)
        for fitted_target, true_target in mapping.items():
            assert estimated_transition[fitted_state, fitted_target] == pytest.approx(
                known_hmm.transition_matrix[true_state, true_target], abs=0.09
            )


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_research_scale_transition_recovery() -> None:
    """Lower-variance recovery check, intentionally excluded from normal CI."""
    truth = np.array([[0.97, 0.03], [0.04, 0.96]])
    data = gaussian_hmm(n_steps=5_000, n_states=2, transition_matrix=truth, seed=812)
    # More restarts make this useful as a research regression rather than a smoke test.
    config = ProbabilisticHMMConfig(
        model_name="slow_recovery", n_states=2, random_seed=8, n_init=5, max_iter=75
    )
    model = GaussianHMM(config).fit(data.observations)
    mapping = align_model_labels(model, data)
    estimate = np.asarray(model.transition_matrix())
    order = [next(k for k, value in mapping.items() if value == i) for i in range(2)]
    reordered = estimate[np.ix_(order, order)]
    np.testing.assert_allclose(reordered, truth, atol=0.035)
