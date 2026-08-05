"""Contract tests for experimental switching state-space prototypes."""

from pathlib import Path

import numpy as np
import pytest

from regime.models.state_space import (
    ExplicitDurationSwitchingLinearDynamicalSystem,
    NumPyBackend,
    RecurrentSwitchingLinearDynamicalSystem,
    StateSpaceConfig,
    SwitchingDynamicFactorModel,
    SwitchingLinearDynamicalSystem,
    get_backend,
)


@pytest.fixture
def observations() -> np.ndarray:
    rng = np.random.default_rng(4)
    return np.r_[rng.normal(-2, 0.25, (35, 2)), rng.normal(2, 0.25, (35, 2))]


def _config(name: str) -> StateSpaceConfig:
    return StateSpaceConfig(model_name=name, n_states=2, state_dim=2, random_seed=7)


def test_slds_exposes_inference_parameters_uncertainty_and_diagnostics(
    observations: np.ndarray,
) -> None:
    model = SwitchingLinearDynamicalSystem(_config("slds")).fit(observations)
    result = model.infer(observations)

    assert result.filtered_probabilities.shape == (70, 2)
    assert result.smoothed_probabilities.shape == (70, 2)
    assert result.filtered_state_covariances.shape == (70, 2, 2, 2)
    assert np.allclose(result.filtered_probabilities.sum(axis=1), 1)
    assert np.allclose(result.smoothed_probabilities.sum(axis=1), 1)
    assert model.state_space_parameters.transition_matrix.shape == (2, 2)
    assert result.numerical_diagnostics["probabilities_normalized"] is True
    assert model.metadata.attributes["experimental"] is True


def test_json_serialization_round_trip(tmp_path: Path, observations: np.ndarray) -> None:
    model = SwitchingLinearDynamicalSystem(_config("serial")).fit(observations)
    destination = tmp_path / "model.json"
    model.save(destination)
    restored = SwitchingLinearDynamicalSystem.load(destination)

    assert np.allclose(
        restored.state_space_parameters.transition_matrix,
        model.state_space_parameters.transition_matrix,
    )
    assert restored.config == model.config


@pytest.mark.parametrize(
    "model_type",
    [
        SwitchingDynamicFactorModel,
        RecurrentSwitchingLinearDynamicalSystem,
        ExplicitDurationSwitchingLinearDynamicalSystem,
    ],
)
def test_experimental_variants_fit(model_type: type, observations: np.ndarray) -> None:
    model = model_type(_config(model_type.__name__)).fit(observations)
    assert np.asarray(model.predict_proba(observations)).shape == (70, 2)
    if isinstance(model, ExplicitDurationSwitchingLinearDynamicalSystem):
        assert np.all(model.expected_durations >= 1)


def test_cpu_backend_and_optional_backend_selection() -> None:
    backend = get_backend()
    assert isinstance(backend, NumPyBackend)
    assert backend.asarray([1, 2]).dtype == np.float64
    with pytest.raises(ValueError, match="unknown array backend"):
        get_backend("magic")
