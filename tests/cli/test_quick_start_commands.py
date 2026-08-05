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
        (("train",), "train"),
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
