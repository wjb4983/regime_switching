"""Run report command."""

from __future__ import annotations

from pathlib import Path

import typer

from regime.cli.common import EXPERIMENTS_DIR, command_errors, emit
from regime.experiments.store import ExperimentStore
from regime.reporting.experiment_report import ExperimentReportAssembler


@command_errors
def report(
    run_id: str | None = typer.Option(None, "--run-id", help="Registered run identifier."),
    experiment_group: str | None = typer.Option(
        None, "--experiment-group", help="Experiment group name or identifier."
    ),
    config: Path | None = typer.Option(
        None, "--config", help="Optional report YAML configuration."
    ),
    output: Path | None = typer.Option(None, "--output", help="Destination HTML path."),
) -> None:
    """Create a portable HTML summary for a registered run."""
    store = ExperimentStore(EXPERIMENTS_DIR)
    if bool(run_id) == bool(experiment_group):
        raise ValueError("provide exactly one of --run-id or --experiment-group")
    identifier = run_id or experiment_group or "report"
    destination = (output or Path("reports") / f"{identifier}.html").expanduser().resolve()
    builder = ExperimentReportAssembler(store).assemble(
        run_id=run_id, experiment_group=experiment_group, config=config
    )
    # Retain the familiar single-run heading unless a report configuration supplies one.
    if run_id and config is None:
        builder.title = f"Regime run {run_id}"
    builder.write(destination)
    if run_id:
        store.add_artifact(run_id, "report", destination)
    emit(
        {
            "status": "completed",
            "run_id": run_id,
            "experiment_group": experiment_group,
            "report": str(destination),
        }
    )
