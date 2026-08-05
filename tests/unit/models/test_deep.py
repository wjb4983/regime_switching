"""Dependency-light tests for the optional deep model surface."""

from __future__ import annotations

import importlib.util

import pytest
from pydantic import ValidationError

from regime.models.deep import DeepModelConfig


@pytest.mark.unit
def test_deep_config_exposes_reproducible_runtime_controls() -> None:
    config = DeepModelConfig(random_seed=42, device="cpu", precision="float64")
    assert config.deterministic
    assert config.random_seed == 42
    assert config.device == "cpu"
    assert config.precision == "float64"
    assert config.calibration == "temperature"


@pytest.mark.unit
def test_validation_fraction_must_be_strictly_between_zero_and_one() -> None:
    with pytest.raises(ValidationError, match="fractional validation_window"):
        DeepModelConfig(validation_window=1.0)


@pytest.mark.unit
def test_deep_import_has_actionable_error_without_extra() -> None:
    if importlib.util.find_spec("torch") is not None:
        from regime.models.deep import LSTM

        assert LSTM.__name__ == "LSTM"
    else:
        with pytest.raises(ImportError, match=r"regime-switching\[deep\]"):
            from regime.models.deep import LSTM
