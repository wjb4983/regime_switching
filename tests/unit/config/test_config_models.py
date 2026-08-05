"""Unit tests for typed configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from regime.config import (
    ConfigLoadError,
    ExperimentConfig,
    ModelConfig,
    ValidationConfig,
    load_config,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(5)]


def test_experiment_yaml_loading_supports_inheritance_env_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    base = tmp_path / "base.yml"
    child = tmp_path / "child.yml"
    base.write_text(
        """
name: base-experiment
work_dir: ${DATA_ROOT}/work
dataset:
  name: base-dataset
  target: return
  data:
    name: prices
    source: ${DATA_ROOT}/prices.csv
    source_type: csv
  features:
    - name: returns
      inputs: [close]
      transforms: [pct_change]
model:
  name: baseline
  model_type: hmm
  n_regimes: 2
""",
        encoding="utf-8",
    )
    child.write_text(
        """
extends: base.yml
name: child-experiment
model:
  n_regimes: 3
evaluation:
  metrics: [accuracy, f1]
""",
        encoding="utf-8",
    )

    config = load_config(child, ExperimentConfig)

    assert config.name == "child-experiment"
    assert config.model.n_regimes == 3
    assert config.dataset.data.source == (tmp_path / "data" / "prices.csv").resolve(strict=False)
    assert config.work_dir == (tmp_path / "data" / "work").resolve(strict=False)
    assert config.evaluation.metrics == ["accuracy", "f1"]


def test_config_hash_is_stable_for_equal_configs() -> None:
    first = ModelConfig(name="m", model_type="hmm", n_regimes=2)
    second = ModelConfig(name="m", model_type="hmm", n_regimes=2)

    assert first.config_hash() == second.config_hash()
    assert len(first.config_hash()) == 64


def test_validation_errors_are_actionable(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yml"
    config_path.write_text("strategy: walk_forward\nn_splits: 1\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="n_splits >= 2") as exc_info:
        load_config(config_path, ValidationConfig)

    assert "Review the field name" in str(exc_info.value)


def test_missing_environment_variable_message_is_actionable(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_env.yml"
    config_path.write_text(
        "name: m\nmodel_type: hmm\nartifact_dir: ${MISSING_ARTIFACT_DIR}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="Set it in the environment"):
        load_config(config_path, ModelConfig)
