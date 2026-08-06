"""Evaluation command."""

from pathlib import Path

from regime.cli.common import command_errors, config_option, config_workflow, resume_option
from regime.evaluation.service import evaluate_config


@command_errors
def evaluate(
    config: Path = config_option("Evaluation YAML."), resume: bool = resume_option()
) -> dict[str, object]:
    """Run a configured evaluation."""
    return config_workflow("evaluate", config, resume=resume, worker=evaluate_config)
