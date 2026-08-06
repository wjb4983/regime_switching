"""Complete experiment pipeline command."""

from pathlib import Path

import typer

from regime.cli import common
from regime.cli.common import command_errors, config_option, emit, resume_option
from regime.experiments.pipeline import ExperimentPipeline

app = typer.Typer(no_args_is_help=True)


@app.command("run")
@command_errors
def run_experiment(
    config: Path = config_option("Experiment graph YAML."),
    resume: bool = resume_option(),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without fitting."),
) -> None:
    """Validate or execute a dependency-ordered experiment graph."""
    pipeline = ExperimentPipeline.from_root(common.EXPERIMENTS_DIR)
    result = pipeline.dry_run(config) if dry_run else pipeline.run(config, resume=resume)
    emit(result)
