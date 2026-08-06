"""Focused model-registry contract tests."""

from __future__ import annotations

import importlib

import pytest

from regime.models import ModelRegistryError, available_models, create_model, model_spec


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
def test_missing_optional_dependency_names_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def missing(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name == "hdbscan":
            raise ModuleNotFoundError("No module named hdbscan", name="hdbscan")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModelRegistryError, match=r"regime-switching\[clustering\]"):
        create_model("hdbscan")
