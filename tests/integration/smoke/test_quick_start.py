"""Small synthetic, ordered smoke test for the documented quick start."""

from __future__ import annotations

import json
from pathlib import Path

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
    workflow = [
        (("synthetic", "generate"), "synthetic.generate", "observations: 64\nseed: 42\n"),
        (("features", "build"), "features.build", "window: 5\ndrop_warmup: true\n"),
        (("train",), "train", "model: volatility_threshold\n"),
        (("train",), "train", "model: kmeans\nn_states: 2\nrandom_seed: 42\n"),
        (("train",), "train", "model: gaussian_hmm\nn_states: 2\nmax_iter: 5\n"),
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
