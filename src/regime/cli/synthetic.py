"""Synthetic data commands."""

from pathlib import Path

import typer

from regime.cli.common import command_errors, config_option, config_workflow, resume_option

app = typer.Typer(no_args_is_help=True)


@app.command("generate")
@command_errors
def generate(
    config: Path = config_option("Synthetic generator YAML."), resume: bool = resume_option()
) -> None:
    """Generate data from a configured synthetic process."""
    config_workflow("synthetic.generate", config, resume=resume)
