"""CLI contracts used by the quick-start smoke workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from regime.cli import app, common


@pytest.mark.parametrize(
    ("arguments", "operation"),
    [
        (("synthetic", "generate"), "synthetic.generate"),
        (("features", "build"), "features.build"),
        (("evaluate",), "evaluate"),
    ],
)
def test_config_driven_quick_start_commands_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
    operation: str,
) -> None:
    """Each reusable quick-start command accepts a small local configuration."""
    config = tmp_path / "small.yaml"
    config.write_text("seed: 42\nobservations: 32\n", encoding="utf-8")
    monkeypatch.setattr(common, "EXPERIMENTS_DIR", tmp_path / "experiments")

    result = CliRunner().invoke(app, [*arguments, "--config", str(config), "--no-resume"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["operation"] == operation
    assert payload["run_id"]


@pytest.mark.parametrize(
    ("model", "features", "fit_parameters", "model_file"),
    [
        (
            "volatility-threshold",
            ["realized_volatility"],
            {"feature": "realized_volatility", "threshold": 0.2},
            "model.json",
        ),
        ("kmeans", ["return_1d", "realized_volatility"], {"n_init": 2}, "model.pkl"),
        ("gaussian-hmm", ["return_1d"], {"max_iter": 2, "n_init": 1}, "model.pkl"),
    ],
)
def test_train_creates_registered_static_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    features: list[str],
    fit_parameters: dict[str, object],
    model_file: str,
) -> None:
    """Representative rule, clustering, and HMM runs produce loadable artifacts."""
    import pandas as pd
    import yaml

    dataset = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, tz="UTC"),
            "return_1d": [(-1) ** index * index / 100 for index in range(20)],
            "realized_volatility": [index / 50 for index in range(20)],
        }
    ).to_parquet(dataset, index=False)
    output = tmp_path / model
    config = tmp_path / f"{model}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": model,
                "input": str(dataset),
                "output": str(output),
                "features": features,
                "n_states": 2,
                "random_seed": 7,
                "minimum_observations": 10,
                "fit_parameters": fit_parameters,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(common, "EXPERIMENTS_DIR", tmp_path / "experiments")

    result = CliRunner().invoke(app, ["train", "--config", str(config), "--no-resume"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert (output / model_file).is_file()
    assert (output / "resolved_configuration.json").is_file()
    assert (output / "metadata.json").is_file()
    assert (output / "training_diagnostics.json").is_file()
    assert (output / "state_statistics.json").is_file()
    assert (output / "in_sample_predictions.parquet").is_file()
    with common.ExperimentStore(tmp_path / "experiments").connect() as connection:
        run = connection.execute(
            "SELECT status, dataset_hash, model_hash FROM runs WHERE run_id=?",
            (payload["run_id"],),
        ).fetchone()
        artifact_count = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id=?", (payload["run_id"],)
        ).fetchone()[0]
    assert dict(run)["status"] == "completed"
    assert dict(run)["dataset_hash"]
    assert dict(run)["model_hash"]
    assert artifact_count >= 9


def test_quick_start_model_commands_are_in_recommended_order() -> None:
    """Keep the transparent baseline ahead of clustering and the Gaussian HMM."""
    model_configs = [
        "configs/models/rule_volatility_threshold.yaml",
        "configs/models/kmeans_regime.yaml",
        "configs/models/gaussian_hmm.yaml",
    ]

    assert [Path(config).stem for config in model_configs] == [
        "rule_volatility_threshold",
        "kmeans_regime",
        "gaussian_hmm",
    ]
