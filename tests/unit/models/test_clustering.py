"""Tests for clustering regime models and label utilities."""

import numpy as np
import pandas as pd

from regime.models.clustering import (
    ClusteringConfig,
    GaussianMixtureRegimeModel,
    JumpPenalizedKMeansRegimeModel,
    KMeansRegimeModel,
    align_labels,
    smooth_assignments,
    state_occupancy,
    transition_summary,
)


def _sample() -> pd.DataFrame:
    left = np.column_stack([np.linspace(-2.0, -1.0, 12), np.linspace(0.0, 1.0, 12)])
    right = np.column_stack([np.linspace(1.0, 2.0, 12), np.linspace(3.0, 4.0, 12)])
    return pd.DataFrame(np.vstack([left, right]), columns=["ret", "vol"])


def test_kmeans_summaries_are_in_original_scale() -> None:
    model = KMeansRegimeModel(ClusteringConfig(n_states=2, random_seed=7)).fit(_sample())

    assert model.result is not None
    assert len(model.result.assignments) == 24
    assert set(model.result.occupancy) == set(model.result.centroids)
    assert set(model.result.centroids[model.result.assignments[0]]) == {"ret", "vol"}
    assert len(model.transition_matrix()) == 2


def test_gaussian_mixture_returns_probabilities_and_entropy() -> None:
    model = GaussianMixtureRegimeModel(ClusteringConfig(n_states=2, random_seed=11)).fit(_sample())
    probabilities = model.predict_proba(_sample().iloc[:3])

    assert len(probabilities) == 3
    assert all(abs(sum(row) - 1.0) < 1e-8 for row in probabilities)
    assert model.result is not None
    assert model.result.entropy is not None
    assert len(model.result.entropy) == len(model.result.assignments)


def test_temporal_smoothing_occupancy_transitions_and_label_alignment() -> None:
    assert smooth_assignments([0, 1, 0, 0, 1], window=3) == (0, 0, 0, 0, 0)
    assert state_occupancy([2, 2, 3, 3]) == {2: 0.5, 3: 0.5}
    assert transition_summary([2, 3, 3])["counts"] == [[0, 1], [0, 1]]
    assert align_labels([0, 0, 1, 1], [4, 4, 9, 9]) == (0, 0, 1, 1)


def test_jump_penalized_kmeans_fits() -> None:
    model = JumpPenalizedKMeansRegimeModel(
        ClusteringConfig(n_states=2, random_seed=3, jump_penalty=2.0)
    ).fit(_sample())

    assert model.result is not None
    assert len(model.result.assignments) == 24
