"""Small synthetic, ordered smoke test for the documented quick start."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from regime.cli import app, common, report


@pytest.mark.integration
@pytest.mark.synthetic
def test_quick_start_workflow_in_recommended_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run every quick-start stage using only a tiny synthetic configuration."""
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(common, "EXPERIMENTS_DIR", experiments)
    monkeypatch.setattr(report, "EXPERIMENTS_DIR", experiments)
    runner = CliRunner()

    configs = tmp_path / "configs"
    configs.mkdir()
    feature_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=32, tz="UTC"),
            "return_1d": [(-1) ** index * index / 100 for index in range(32)],
            "realized_volatility": [index / 100 for index in range(32)],
        }
    ).to_parquet(feature_path, index=False)
    training_common = (
        f"input: {feature_path}\nfeatures: [return_1d, realized_volatility]\n"
        "minimum_observations: 16\nn_states: 2\nrandom_seed: 42\n"
    )
    workflow = [
        (("synthetic", "generate"), "synthetic.generate", "observations: 64\nseed: 42\n"),
        (("features", "build"), "features.build", "window: 5\ndrop_warmup: true\n"),
        (
            ("train",),
            "train",
            training_common + f"output: {tmp_path / 'rule'}\nmodel: volatility_threshold\n"
            "fit_parameters:\n  feature: realized_volatility\n  threshold: 0.2\n",
        ),
        (
            ("train",),
            "train",
            training_common + f"output: {tmp_path / 'kmeans'}\nmodel: kmeans\n",
        ),
        (
            ("train",),
            "train",
            training_common + f"output: {tmp_path / 'hmm'}\nmodel: gaussian_hmm\n"
            "fit_parameters:\n  max_iter: 3\n",
        ),
        (("evaluate",), "evaluate", "validation: walk_forward\nfolds: 2\n"),
        (("evaluate",), "evaluate", "strategy: volatility_targeting\nlookback: 5\n"),
    ]

    completed: list[dict[str, object]] = []
    for index, (arguments, operation, contents) in enumerate(workflow, start=1):
        config = configs / f"{index:02d}-{operation.replace('.', '-')}.yaml"
        config.write_text(contents, encoding="utf-8")
        result = runner.invoke(app, [*arguments, "--config", str(config), "--no-resume"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["operation"] == operation
        completed.append(payload)

    report_path = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["report", "--run-id", str(completed[-1]["run_id"]), "--output", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    assert report_path.is_file()
    assert "Regime run" in report_path.read_text(encoding="utf-8")
    assert [payload["operation"] for payload in completed] == [
        "synthetic.generate",
        "features.build",
        "train",
        "train",
        "train",
        "evaluate",
        "evaluate",
    ]
