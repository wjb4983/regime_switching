from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pytest

from regime.evaluation.runner import (
    ComparisonContractError,
    DatasetConfig,
    EvaluationConfig,
    EvaluationRunner,
    ModelConfig,
    ValidationConfig,
)
from regime.models.base import ModelMetadata, RegimeModel, RegimeModelConfig
from regime.validation.splitters import ExpandingWindowSplitter


class MeanThresholdModel(RegimeModel):
    def __init__(self) -> None:
        self.threshold = 0.0
        self._metadata = ModelMetadata(model_name="mean_threshold", model_version="0.1", n_states=2)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def fit(self, dataset: Any, config: RegimeModelConfig) -> Self:
        self.threshold = float(pd.DataFrame(dataset)["feature"].mean())
        return self

    def predict(self, dataset: Any) -> list[int]:
        return [int(value >= self.threshold) for value in pd.DataFrame(dataset)["feature"]]

    def predict_proba(self, dataset: Any) -> list[list[float]]:
        values = pd.DataFrame(dataset)["feature"].to_numpy(dtype=float)
        p1 = 1.0 / (1.0 + np.exp(-(values - self.threshold)))
        return np.column_stack([1.0 - p1, p1]).tolist()

    def state_statistics(self) -> dict[str, dict[str, float]]:
        return {
            "0": {"mean": self.threshold - 1.0},
            "1": {"mean": self.threshold + 1.0},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(str(self.threshold), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        model = cls()
        model.threshold = float(Path(path).read_text(encoding="utf-8"))
        return model


def test_runner_persists_predictions_metrics_diagnostics_and_resumes(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "feature": np.linspace(-2.0, 2.0, 20),
            "state": [0] * 10 + [1] * 10,
        }
    )
    runner = EvaluationRunner()
    result = runner.run(
        DatasetConfig(data=data, dataset_id="synthetic"),
        ModelConfig(model_factory=MeanThresholdModel, save_models=False),
        ValidationConfig(
            splitter=ExpandingWindowSplitter(
                initial_train_size=8, validation_size=2, test_size=3, step=3
            ),
            execution_delay=1,
        ),
        EvaluationConfig(output_dir=tmp_path, run_id="smoke", produce_smoothed=True),
    )

    assert Path(result.predictions_path).exists()
    assert Path(result.metrics_path).exists()
    assert Path(result.diagnostics_path).exists()
    assert Path(result.provenance_path).exists()
    assert Path(result.checkpoint_path).exists()
    assert result.metrics["n_windows"] == 3.0
    assert result.metrics["n_predictions"] == 9.0
    assert "brier_score" in result.metrics

    resumed = runner.run(
        DatasetConfig(data=data, dataset_id="synthetic"),
        ModelConfig(model_factory=MeanThresholdModel, save_models=False),
        ValidationConfig(
            splitter=ExpandingWindowSplitter(
                initial_train_size=8, validation_size=2, test_size=3, step=3
            ),
            execution_delay=1,
        ),
        EvaluationConfig(output_dir=tmp_path, run_id="smoke", resume=True),
    )

    assert resumed.metrics["n_predictions"] == 9.0


def test_runner_rejects_non_matching_comparison_contract() -> None:
    left = {
        "information_set": "close_t",
        "validation_period": (0, 10),
        "retraining_schedule": "each_window",
        "execution_delay": 1,
        "cost_assumptions": {"bps": 1},
        "downstream_decision_rules": {"threshold": 0.6},
    }
    right = {**left, "execution_delay": 0}

    with pytest.raises(ComparisonContractError, match="execution_delay"):
        EvaluationRunner.assert_comparable(left, right)
