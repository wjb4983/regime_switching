"""Hyperparameter tuning command."""

from pathlib import Path

from regime.cli.common import command_errors, config_option, config_workflow, resume_option
from regime.tuning.config import SearchSpace


@command_errors
def tune(
    config: Path = config_option("Tuning search-space YAML."), resume: bool = resume_option()
) -> None:
    """Validate and run a resumable tuning search."""
    SearchSpace.from_yaml(config)
    config_workflow("tune", config, resume=resume)
