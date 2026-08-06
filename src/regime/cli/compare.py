"""Experiment comparison command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from regime.cli.common import EXPERIMENTS_DIR, command_errors, emit
from regime.evaluation.comparison_service import ComparisonConfiguration, ComparisonService
from regime.experiments.store import ExperimentStore


@command_errors
def compare(
    experiment_group: str = typer.Option(
        ..., "--experiment-group", help="Name or ID of the experiment group to compare."
    ),
    metric: str = typer.Option("loss", "--metric", help="Primary metric used for ranking."),
    benchmark: str | None = typer.Option(
        None, "--benchmark", help="Benchmark model name or run ID (default: first completed run)."
    ),
    format_: str = typer.Option("json", "--format", help="Output format: json, csv, or html."),
    output: Path | None = typer.Option(
        None, "--output", help="Write rendered output to this path."
    ),
    bootstrap_method: str = typer.Option(
        "stationary", "--bootstrap-method", help="Bootstrap method: stationary or block."
    ),
    block_length: float = typer.Option(10.0, "--block-length", min=1.0),
    seed: int = typer.Option(0, "--seed"),
    alpha: float = typer.Option(0.05, "--alpha", min=0.0, max=1.0),
    correction: str = typer.Option("holm", "--multiple-testing-correction"),
) -> None:
    """Compare completed runs without crossing incompatible evaluation contracts."""
    if format_ not in {"json", "csv", "html"}:
        raise ValueError("format must be one of: json, csv, html")
    if bootstrap_method not in {"stationary", "block"}:
        raise ValueError("bootstrap method must be one of: stationary, block")
    config = ComparisonConfiguration(
        primary_metric=metric,
        benchmark=benchmark,
        primary_loss=metric,
        bootstrap_method=bootstrap_method,  # type: ignore[arg-type]
        block_length=block_length,
        seed=seed,
        alpha=alpha,
        multiple_testing_correction=correction,
    )
    result = ComparisonService(ExperimentStore(EXPERIMENTS_DIR)).compare(experiment_group, config)
    payload = {"status": "completed", **result.to_dict()}
    if format_ == "json":
        rendered = json.dumps(payload, sort_keys=True, default=str)
    else:
        frame = pd.DataFrame(result.table)
        if "downstream_metrics" in frame:
            frame["downstream_metrics"] = frame["downstream_metrics"].map(
                lambda value: json.dumps(value, sort_keys=True)
            )
        rendered = frame.to_csv(index=False) if format_ == "csv" else frame.to_html(index=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        emit(
            {
                "status": "completed",
                "experiment_group": experiment_group,
                "format": format_,
                "output": str(output),
                "comparison_artifact": result.artifact_path,
            }
        )
    else:
        typer.echo(rendered)
