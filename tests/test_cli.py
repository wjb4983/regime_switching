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

    result = CliRunner().invoke(app, ["evaluate", "--config", str(config)])

    assert result.exit_code == 0
    assert "verysecret" not in result.stdout
    artifacts = "".join(path.read_text(encoding="utf-8") for path in experiments.rglob("*.json"))
    assert "verysecret" not in artifacts
    assert "[REDACTED]" in artifacts


def test_config_workflow_interpolates_environment_before_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("provider: massive\napi_key: ${MASSIVE_API_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY", "resolved-test-key")
    monkeypatch.setattr(common, "EXPERIMENTS_DIR", tmp_path / "experiments")
    received: dict[str, object] = {}

    common.config_workflow(
        "data.ingest",
        config,
        resume=False,
        worker=lambda _run, loaded: received.update(loaded) or {},
    )

    assert received["api_key"] == "resolved-test-key"


def test_cli_reports_missing_environment_variable_before_running_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("provider: massive\napi_key: ${MISSING_MASSIVE_KEY}\n", encoding="utf-8")
    monkeypatch.delenv("MISSING_MASSIVE_KEY", raising=False)
    monkeypatch.setattr(common, "EXPERIMENTS_DIR", tmp_path / "experiments")

    result = CliRunner().invoke(app, ["data", "ingest", "--config", str(config)])

    assert result.exit_code == 2
    error = json.loads(result.stderr)
    assert error["error"]["code"] == "invalid_input"
    assert "MISSING_MASSIVE_KEY" in error["error"]["message"]
