"""Typer application composition."""

from __future__ import annotations

import typer

from regime.cli import compare, data, evaluate, features, models, report, synthetic, train, tune

app = typer.Typer(name="regime", no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(data.app, name="data")
app.add_typer(features.app, name="features")
app.add_typer(synthetic.app, name="synthetic")
app.add_typer(models.app, name="models")
app.command("train")(train.train)
app.command("evaluate")(evaluate.evaluate)
app.command("compare")(compare.compare)
app.command("report")(report.report)
app.command("tune")(tune.tune)


def main() -> None:
    """Run the CLI using argv without invoking a platform shell."""
    app()
