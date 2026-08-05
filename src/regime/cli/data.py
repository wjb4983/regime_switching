"""Data commands."""

from pathlib import Path

import typer

from regime.cli.common import command_errors, config_option, config_workflow, resume_option

app = typer.Typer(no_args_is_help=True)


@app.command("ingest")
@command_errors
def ingest(
    config: Path = config_option("Data ingestion YAML."), resume: bool = resume_option()
) -> None:
    """Ingest data described by CONFIG."""
    config_workflow("data.ingest", config, resume=resume)
