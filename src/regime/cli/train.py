"""Model training command."""

from pathlib import Path

from regime.cli.common import command_errors, config_option, config_workflow, resume_option
from regime.training.runner import train_model


@command_errors
def train(
    config: Path = config_option("Model training YAML."), resume: bool = resume_option()
) -> None:
    """Train a configured model."""
    config_workflow("train", config, resume=resume, worker=train_model)
