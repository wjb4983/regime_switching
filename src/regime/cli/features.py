"""Feature commands."""

from pathlib import Path

import typer

from regime.cli.common import command_errors, config_option, config_workflow, resume_option

app = typer.Typer(no_args_is_help=True)


@app.command("build")
@command_errors
def build(
    config: Path = config_option("Feature build YAML."), resume: bool = resume_option()
) -> None:
    """Build configured features."""
    config_workflow("features.build", config, resume=resume)
