from collections.abc import Callable

import numpy as np
import pytest

from regime.synthetic import (
    SyntheticDataset,
    abrupt_change_points,
    gaussian_hmm,
    gradual_transitions,
    hsmm_non_geometric,
    markov_switching_ar,
    missing_observations,
    misspecified_state_count_scenarios,
    non_recurring_regimes,
    outliers,
    overlapping_emissions,
    rare_crisis_states,
    recurring_regimes,
    structural_drift,
    student_t_hmm,
    switching_covariance_matrices,
    switching_stochastic_volatility,
)


@pytest.mark.timeout(5)
@pytest.mark.synthetic
@pytest.mark.parametrize(
    "factory",
    [
        gaussian_hmm,
        student_t_hmm,
        hsmm_non_geometric,
        markov_switching_ar,
        switching_stochastic_volatility,
        switching_covariance_matrices,
        abrupt_change_points,
        gradual_transitions,
        recurring_regimes,
        non_recurring_regimes,
        rare_crisis_states,
        overlapping_emissions,
        structural_drift,
        misspecified_state_count_scenarios,
    ],
)
def test_generators_are_deterministic(factory: Callable[..., SyntheticDataset]) -> None:
    first = factory(seed=123)
    second = factory(seed=123)

    assert isinstance(first, SyntheticDataset)
    np.testing.assert_equal(first.observations, second.observations)
    if first.latent_states is not None:
        np.testing.assert_array_equal(first.latent_states, second.latent_states)
    assert first.seed == 123
    assert first.metadata["kind"]


@pytest.mark.timeout(5)
@pytest.mark.synthetic
def test_hmm_shapes_and_transition_matrix() -> None:
    data = gaussian_hmm(n_steps=50, n_states=4, n_features=2, seed=7)

    assert data.observations.shape == (50, 2)
    assert data.latent_states is not None
    assert data.latent_states.shape == (50,)
    assert data.transition_matrix is not None
    np.testing.assert_allclose(data.transition_matrix.sum(axis=1), np.ones(4))


@pytest.mark.timeout(5)
@pytest.mark.synthetic
def test_change_point_generators_report_true_boundaries() -> None:
    abrupt = abrupt_change_points(n_steps=20, change_points=(5, 12), seed=9)
    gradual = gradual_transitions(n_steps=20, center=11, seed=9)

    np.testing.assert_array_equal(abrupt.true_change_points, np.array([5, 12]))
    np.testing.assert_array_equal(gradual.true_change_points, np.array([11]))


@pytest.mark.timeout(5)
@pytest.mark.synthetic
def test_missing_observations_and_outliers_are_injected_deterministically() -> None:
    base = gaussian_hmm(n_steps=30, n_features=2, seed=1)
    missing = missing_observations(base, missing_probability=0.5, seed=2)
    spiked = outliers(base, outlier_probability=0.5, magnitude=10.0, seed=2)

    assert np.isnan(missing.observations).any()
    assert missing.metadata["missing_mask"].shape == base.observations.shape
    assert spiked.metadata["outlier_mask"].shape == (30,)
    assert np.max(np.abs(spiked.observations - base.observations)) >= 10.0


@pytest.mark.timeout(5)
@pytest.mark.synthetic
def test_misspecified_state_count_metadata() -> None:
    data = misspecified_state_count_scenarios(true_states=4, assumed_states=2, seed=4)

    assert data.metadata["true_states"] == 4
    assert data.metadata["assumed_states"] == 2
    assert data.latent_states is not None
    assert len(np.unique(data.latent_states)) <= 4
