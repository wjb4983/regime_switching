"""Run report command."""

from __future__ import annotations

import html
from pathlib import Path

import typer

from regime.cli.common import EXPERIMENTS_DIR, command_errors, emit
from regime.experiments.store import ExperimentStore
from regime.reporting.report import ReportBuilder


@command_errors
def report(
    run_id: str = typer.Option(..., "--run-id", help="Registered run identifier."),
    output: Path | None = typer.Option(None, "--output", help="Destination HTML path."),
) -> None:
    """Create a portable HTML summary for a registered run."""
    store = ExperimentStore(EXPERIMENTS_DIR)
    with store.connect() as connection:
        run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise FileNotFoundError(f"Run not found: {run_id}")
        artifacts = connection.execute(
            "SELECT kind, path, hash FROM artifacts WHERE run_id=? ORDER BY created_at", (run_id,)
        ).fetchall()
    destination = (output or Path("reports") / f"{run_id}.html").expanduser().resolve()
    builder = ReportBuilder(f"Regime run {run_id}", subtitle=f"Status: {run['status']}")
    # Keep the report self-contained and avoid placing configuration/secrets in its title.
    summary = (
        "<h2>Artifacts</h2><ul>"
        + "".join(
            f"<li>{html.escape(item['kind'])}: {html.escape(item['path'])}</li>"
            for item in artifacts
        )
        + "</ul>"
    )
    # ReportBuilder figures require research metadata, so append a small safe summary directly.
    rendered = builder.render().replace("</main>", summary + "</main>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(destination)
    store.add_artifact(run_id, "report", destination)
    emit({"status": "completed", "run_id": run_id, "report": str(destination)})
