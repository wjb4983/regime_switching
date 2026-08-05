"""Evaluation command."""

from pathlib import Path

from regime.cli.common import command_errors, config_option, config_workflow, resume_option


@command_errors
def evaluate(
    config: Path = config_option("Evaluation YAML."), resume: bool = resume_option()
) -> None:
    """Run a configured evaluation."""
    config_workflow("evaluate", config, resume=resume)
