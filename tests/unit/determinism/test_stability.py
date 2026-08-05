"""Determinism tests for hashes, seeds, and synthetic data generation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from regime.experiments.hashes import config_hash, dataset_hash, feature_hash
from regime.models.probabilistic import GaussianHMM, ProbabilisticHMMConfig
from regime.synthetic import SyntheticDataset, abrupt_change_points, gaussian_hmm, student_t_hmm


@pytest.mark.timeout(5)
@pytest.mark.unit
def test_config_hash_is_stable_across_mapping_order_and_copies() -> None:
    first = {"model": {"states": 3, "sticky": True}, "features": ["return", "volume"]}
    reordered = {"features": ["return", "volume"], "model": {"sticky": True, "states": 3}}

    assert config_hash(first) == config_hash(reordered)
    assert config_hash(first) == config_hash(dict(first))
    assert config_hash(first) != config_hash({**first, "features": ["return"]})


@pytest.mark.timeout(5)
@pytest.mark.unit
def test_dataset_hash_is_stable_for_equivalent_tabular_content() -> None:
    frame = pd.DataFrame(
        {"return": [0.01, -0.02, 0.03], "volume": [100, 120, 90]},
        index=pd.date_range("2026-01-01", periods=3, tz="UTC"),
    )

    assert dataset_hash(frame) == dataset_hash(frame.copy(deep=True))
    changed = frame.copy(deep=True)
    changed.loc[changed.index[0], "return"] = 0.02
    assert dataset_hash(frame) != dataset_hash(changed)


@pytest.mark.timeout(5)
@pytest.mark.unit
def test_feature_hash_is_stable_and_order_sensitive() -> None:
    definitions = {"return_5d": {"window": 5}, "volatility_20d": {"window": 20}}
    reordered = {"volatility_20d": {"window": 20}, "return_5d": {"window": 5}}

    assert feature_hash(definitions) == feature_hash(reordered)
    assert feature_hash(["return_5d", "volatility_20d"]) != feature_hash(
        ["volatility_20d", "return_5d"]
    )


@pytest.mark.timeout(10)
@pytest.mark.unit
def test_random_seed_reproduces_model_fit() -> None:
    observations = gaussian_hmm(n_steps=100, n_features=2, seed=94).observations
    config = ProbabilisticHMMConfig(
        model_name="seeded_hmm", n_states=2, random_seed=27, n_init=2, max_iter=8
    )

    first = GaussianHMM(config).fit(observations)
    second = GaussianHMM(config).fit(observations.copy())

    np.testing.assert_array_equal(first.predict(observations), second.predict(observations))
    np.testing.assert_allclose(
        first.predict_proba(observations), second.predict_proba(observations)
    )


@pytest.mark.timeout(5)
@pytest.mark.unit
@pytest.mark.synthetic
@pytest.mark.parametrize("generator", [gaussian_hmm, student_t_hmm, abrupt_change_points])
def test_synthetic_generators_are_deterministic(
    generator: Callable[..., SyntheticDataset],
) -> None:
    # The abrupt generator's documented default boundaries extend to step 130.
    first = generator(n_steps=160, n_features=2, seed=311)
    second = generator(n_steps=160, n_features=2, seed=311)

    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.latent_states, second.latent_states)
    np.testing.assert_array_equal(first.true_change_points, second.true_change_points)
    assert first.metadata.keys() == second.metadata.keys()
    assert first.seed == second.seed == 311
