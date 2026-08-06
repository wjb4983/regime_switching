"""Focused model-registry contract tests."""

from __future__ import annotations

import pytest

from regime.models import ModelRegistryError, available_models, create_model, model_spec
from regime.models.registry import model_configuration


@pytest.mark.unit
def test_every_registered_model_has_a_small_valid_configuration() -> None:
    for spec in available_models():
        parameters = (
            {"feature": "volatility", "threshold": 0.2}
            if spec.name == "volatility-threshold"
            else {}
        )
        try:
            model = create_model(spec.name, parameters)
        except ModelRegistryError as error:
            if spec.optional_dependency_group and "requires optional extra" in str(error):
                continue
            raise
        assert model is not None, spec.name


@pytest.mark.unit
def test_unknown_parameters_are_actionable() -> None:
    with pytest.raises(ModelRegistryError, match=r"Invalid parameters.*unexpected"):
        create_model("gaussian-hmm", {"unexpected": 1})


@pytest.mark.unit
def test_alias_and_standard_fields_are_normalized() -> None:
    model = create_model(
        "gaussian_hmm",
        {"state_count": 3, "seed": 7, "fit_parameters": {"max_iter": 1}},
    )
    assert model_spec("gaussian_hmm").name == "gaussian-hmm"
    assert model.config.n_states == 3
    assert model.config.random_seed == 7
    assert model.config.max_iter == 1


@pytest.mark.unit
def test_catalog_alias_fields_are_normalized_for_deep_and_transformer_models() -> None:
    deep = model_configuration(
        "gru", {"fit_parameters": {"input_dim": 2, "hidden_dim": 16, "epochs": 5}}
    )
    transformer = model_configuration(
        "transformer-hmm",
        {"fit_parameters": {"input_dim": 2, "d_model": 16, "n_heads": 2, "n_layers": 1}},
    )

    assert deep.input_dim == 2
    assert deep.hidden_size == 16
    assert deep.max_epochs == 5
    assert transformer.input_dim == 2
    assert transformer.embedding_dim == 16
    assert transformer.num_heads == 2
    assert transformer.num_layers == 1
