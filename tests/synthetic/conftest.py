"""Shared deterministic fixtures and metrics for synthetic recovery tests."""

from __future__ import annotations

import numpy as np
import pytest

from regime.evaluation.alignment import align_states
from regime.models.probabilistic import GaussianHMM, ProbabilisticHMMConfig
from regime.synthetic import SyntheticDataset, gaussian_hmm


@pytest.fixture
def known_hmm() -> SyntheticDataset:
    """A small, identifiable two-state process suitable for pull-request CI."""
    transition = np.array([[0.94, 0.06], [0.08, 0.92]], dtype=float)
    return gaussian_hmm(
        n_steps=240, n_states=2, n_features=2, transition_matrix=transition, seed=1729
    )


@pytest.fixture
def fitted_hmm(known_hmm: SyntheticDataset) -> GaussianHMM:
    config = ProbabilisticHMMConfig(
        model_name="synthetic_gaussian_hmm",
        n_states=2,
        random_seed=37,
        n_init=2,
        max_iter=25,
        tol=1e-5,
    )
    return GaussianHMM(config).fit(known_hmm.observations)


def align_model_labels(model: GaussianHMM, data: SyntheticDataset) -> dict[int, int]:
    """Map fitted nominal labels to the generator's labels using emission means."""
    assert model.means_ is not None
    true_means = np.asarray(data.metadata["means"], dtype=float)
    result = align_states(
        {"labels": range(len(true_means)), "state_means": true_means},
        {"labels": range(len(model.means_)), "state_means": model.means_},
        "state_means",
    )
    return dict(result.candidate_to_reference)


def mapped_predictions(model: GaussianHMM, data: SyntheticDataset) -> np.ndarray:
    mapping = align_model_labels(model, data)
    return np.asarray([mapping[state] for state in model.predict(data.observations)])


def run_lengths(states: np.ndarray, state: int) -> np.ndarray:
    """Return lengths of completed and edge runs for one latent state."""
    padded = np.r_[False, states == state, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]
