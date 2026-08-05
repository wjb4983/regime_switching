"""Experiment comparison command."""

from __future__ import annotations

import json

import typer

from regime.cli.common import EXPERIMENTS_DIR, command_errors, emit
from regime.experiments.store import ExperimentStore


@command_errors
def compare(
    experiment_group: str = typer.Option(
        ..., "--experiment-group", help="Name of the experiment group to compare."
    ),
) -> None:
    """List comparable runs and their recorded results."""
    store = ExperimentStore(EXPERIMENTS_DIR)
    with store.connect() as connection:
        group = connection.execute(
            "SELECT group_id FROM experiment_groups WHERE name=?", (experiment_group,)
        ).fetchone()
        if group is None:
            raise FileNotFoundError(f"Experiment group not found: {experiment_group}")
        rows = connection.execute(
            "SELECT run_id, name, status, config_hash, started_at FROM runs "
            "WHERE group_id=? ORDER BY started_at",
            (group["group_id"],),
        ).fetchall()
        runs = []
        for row in rows:
            results = connection.execute(
                "SELECT kind, value_json FROM results WHERE run_id=? ORDER BY created_at",
                (row["run_id"],),
            ).fetchall()
            record = dict(row)
            record["results"] = [
                {"kind": item["kind"], "value": json.loads(item["value_json"])} for item in results
            ]
            runs.append(record)
    emit({"status": "completed", "experiment_group": experiment_group, "runs": runs})
