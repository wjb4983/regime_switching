"""Command-line interface contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from regime.cli import app, common


def test_commands_are_registered_in_workflow_order() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    commands = ("data", "features", "train", "evaluate", "compare", "report", "tune", "synthetic")
    for command in commands:
        assert command in result.stdout


def test_missing_config_returns_structured_input_error() -> None:
    result = CliRunner().invoke(app, ["train", "--config", "missing.yaml"])

    assert result.exit_code == 2
    error = json.loads(result.stderr)
    assert error["status"] == "error"
    assert error["error"]["code"] == "invalid_input"


def test_config_workflow_redacts_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "model.yaml"
    config.write_text("name: example\napi_key: sk-verysecret123\n", encoding="utf-8")
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(common, "EXPERIMENTS_DIR", experiments)

    result = CliRunner().invoke(app, ["train", "--config", str(config)])

    assert result.exit_code == 0
    assert "verysecret" not in result.stdout
    artifacts = "".join(path.read_text(encoding="utf-8") for path in experiments.rglob("*.json"))
    assert "verysecret" not in artifacts
    assert "[REDACTED]" in artifacts
