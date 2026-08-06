"""CLI discovery tests for registered models."""

from typer.testing import CliRunner

from regime.cli.app import app


def test_models_list_and_describe() -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["models", "list"])
    assert listed.exit_code == 0
    assert "gaussian-hmm" in listed.stdout

    described = runner.invoke(app, ["models", "describe", "gaussian_hmm"])
    assert described.exit_code == 0
    assert '"configuration": "ProbabilisticHMMConfig"' in described.stdout
    assert '"max_iter"' in described.stdout
