"""Artifact-driven assembly of compact, portable experiment reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from regime.experiments.store import ExperimentStore

from .figures import (
    bar_chart,
    distribution_by_state,
    heatmap,
    line_chart,
    probability_area_chart,
    regime_time_series,
    table,
)
from .report import ReportBuilder, ReportFigure, VisualizationMetadata

DEFAULT_SECTIONS = (
    "model_ranking",
    "metric_by_model",
    "regime_timeline",
    "filtered_probabilities",
    "occupancy_duration",
    "transition_heatmap",
    "fold_seed_stability",
    "equity_drawdown",
    "cost_comparison",
    "runtime_resources",
    "comparison_warnings",
)


@dataclass(frozen=True)
class ReportConfiguration:
    """Small configuration surface; omitted sections receive research defaults."""

    title: str = "Regime-switching experiment report"
    sections: tuple[str, ...] = DEFAULT_SECTIONS

    @classmethod
    def load(cls, path: str | Path | None) -> ReportConfiguration:
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        report = raw.get("report", raw) if isinstance(raw, Mapping) else {}
        sections = report.get("sections")
        return cls(
            title=str(report.get("title", cls.title)),
            sections=tuple(sections) if sections is not None else DEFAULT_SECTIONS,
        )


@dataclass
class _RunData:
    row: Mapping[str, Any]
    artifacts: dict[str, list[Path]]
    frames: dict[str, list[pd.DataFrame]]
    json: dict[str, list[dict[str, Any]]]

    @property
    def model(self) -> str:
        metadata = json.loads(str(self.row["metadata_json"]))
        return str(metadata.get("model") or self.row["name"] or self.row["run_id"])


class ExperimentReportAssembler:
    """Read registered artifacts and turn them into a deterministic research report."""

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store

    def assemble(
        self,
        *,
        run_id: str | None = None,
        experiment_group: str | None = None,
        config: str | Path | None = None,
    ) -> ReportBuilder:
        if bool(run_id) == bool(experiment_group):
            raise ValueError("exactly one of run_id or experiment_group is required")
        configuration = ReportConfiguration.load(config)
        group_name, runs, comparison = self._load(run_id, experiment_group)
        subject = experiment_group or run_id or ""
        builder = ReportBuilder(
            configuration.title, subtitle=f"Experiment: {group_name or subject}"
        )
        figures = self._figures(runs, comparison)
        for section in configuration.sections:
            for figure in figures.get(section, ()):
                builder.add(figure)
        return builder

    def write(self, output: str | Path, **selection: Any) -> Path:
        """Assemble and atomically write the self-contained HTML document."""
        return self.assemble(**selection).write(output)

    def _load(
        self, run_id: str | None, group: str | None
    ) -> tuple[str, list[_RunData], dict[str, Any]]:
        with self.store.connect() as con:
            if run_id:
                parent = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if parent is None:
                    raise FileNotFoundError(f"Run not found: {run_id}")
                group_id = parent["group_id"]
                rows = (
                    con.execute(
                        "SELECT * FROM runs WHERE group_id=? AND status='completed' "
                        "ORDER BY started_at",
                        (group_id,),
                    ).fetchall()
                    if group_id
                    else [parent]
                )
                group_row = (
                    con.execute(
                        "SELECT name FROM experiment_groups WHERE group_id=?", (group_id,)
                    ).fetchone()
                    if group_id
                    else None
                )
                group_name = str(group_row["name"] if group_row else parent["name"] or run_id)
            else:
                group_row = con.execute(
                    "SELECT * FROM experiment_groups WHERE name=? OR group_id=?", (group, group)
                ).fetchone()
                if group_row is None:
                    raise FileNotFoundError(f"Experiment group not found: {group}")
                group_id, group_name = group_row["group_id"], str(group_row["name"])
                rows = con.execute(
                    "SELECT * FROM runs WHERE group_id=? AND status='completed' "
                    "ORDER BY started_at",
                    (group_id,),
                ).fetchall()
            loaded = [self._load_run(con, dict(row)) for row in rows]
        comparison: dict[str, Any] = {}
        for run in loaded:
            for value in run.json.get("comparison", []):
                if "table" in value:
                    comparison = value
        return group_name, loaded, comparison

    def _load_run(self, con: Any, row: Mapping[str, Any]) -> _RunData:
        artifacts: dict[str, list[Path]] = {}
        frames: dict[str, list[pd.DataFrame]] = {}
        mappings: dict[str, list[dict[str, Any]]] = {}
        records = con.execute(
            "SELECT kind,path FROM artifacts WHERE run_id=? ORDER BY created_at", (row["run_id"],)
        ).fetchall()
        for record in records:
            kind, path = str(record["kind"]), Path(record["path"])
            if not path.is_absolute() and not path.exists():
                path = Path(self.store.root) / path
            artifacts.setdefault(kind, []).append(path)
            if not path.is_file():
                continue
            try:
                if path.suffix.lower() in {".csv", ".parquet", ".pq"}:
                    frame = (
                        pd.read_csv(path)
                        if path.suffix.lower() == ".csv"
                        else pd.read_parquet(path)
                    )
                    frames.setdefault(kind, []).append(frame)
                elif path.suffix.lower() == ".json":
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        mappings.setdefault(kind, []).append(value)
            except (OSError, ValueError, TypeError):
                continue
        return _RunData(row, artifacts, frames, mappings)

    def _metadata(
        self,
        run: _RunData,
        question: str,
        interpretation: str,
        *,
        probability: str = "not applicable",
    ) -> VisualizationMetadata:
        frames = [frame for values in run.frames.values() for frame in values]
        dates = []
        for frame in frames:
            for column in ("timestamp", "date"):
                if column in frame:
                    values = pd.to_datetime(frame[column], errors="coerce").dropna()
                    if len(values):
                        dates.extend((values.min(), values.max()))
                    break
        period = f"{min(dates).date()} to {max(dates).date()}" if dates else "Not recorded"
        identity = str(run.row.get("model_hash") or run.row.get("config_hash") or "not-recorded")
        return VisualizationMetadata(
            question,
            interpretation,
            period,
            run.model,
            identity,
            probability,  # type: ignore[arg-type]
        )

    @staticmethod
    def _time_index(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in ("timestamp", "date"):
            if column in result:
                result.index = pd.to_datetime(result.pop(column), errors="coerce")
                break
        return result

    def _figures(
        self, runs: list[_RunData], comparison: dict[str, Any]
    ) -> dict[str, list[ReportFigure]]:
        output: dict[str, list[ReportFigure]] = {name: [] for name in DEFAULT_SECTIONS}
        if not runs:
            return output
        representative = next(
            (run for run in runs if not run.model.startswith("comparison:")), runs[0]
        )
        rows = comparison.get("table", [])
        if rows:
            ranking = pd.DataFrame(rows).set_index("model")
            keep = [
                c
                for c in ("rank", "primary_metric", "uncertainty", "stability", "status")
                if c in ranking
            ]
            ranking = ranking[keep].rename(
                columns={
                    "primary_metric": "statistical fit",
                    "uncertainty": "uncertainty interval / SE",
                    "stability": "fold/seed dispersion",
                }
            )
            meta = self._metadata(
                representative,
                "Which model performs best with uncertainty?",
                "Statistical fit and its uncertainty are shown separately from regime "
                "quality and economic usefulness; incompatible models are not ranked.",
            )
            output["model_ranking"].append(
                table(ranking, meta, title="Statistical fit — model ranking with uncertainty")
            )
            metric = pd.DataFrame(
                {str(row.get("model")): [row.get("primary_metric")] for row in rows},
                index=["primary metric"],
            ).T
            output["metric_by_model"].append(
                bar_chart(metric, meta, title="Statistical fit — metric by model")
            )
            stability = pd.Series(
                {str(row.get("model")): row.get("stability") for row in rows},
                name="dispersion",
                dtype=float,
            ).dropna()
            if len(stability):
                output["fold_seed_stability"].append(
                    bar_chart(stability, meta, title="Regime quality — fold/seed stability")
                )
            runtime = pd.Series(
                {str(row.get("model")): row.get("runtime_seconds") for row in rows},
                name="seconds",
                dtype=float,
            ).dropna()
            if len(runtime):
                output["runtime_resources"].append(
                    bar_chart(runtime, meta, title="Runtime/resource comparison", y_label="Seconds")
                )
            diagnostics = [
                item
                for item in comparison.get("diagnostics", [])
                if item.get("status") != "compatible"
            ]
            if diagnostics:
                warnings = pd.DataFrame(diagnostics).set_index("model")
                warning_meta = self._metadata(
                    representative,
                    "Are model comparisons contract-compatible?",
                    "Warnings identify missing metrics or contract differences; affected "
                    "models must not be interpreted as ranked evidence.",
                )
                output["comparison_warnings"].append(
                    table(warnings, warning_meta, title="Comparison warnings")
                )

        for run in runs:
            all_frames = [frame for values in run.frames.values() for frame in values]
            for raw in all_frames:
                frame = self._time_index(raw)
                state_col = next(
                    (c for c in ("regime", "state", "predicted_state") if c in frame), None
                )
                value_col = next(
                    (c for c in ("price", "close", "return", "returns", "loss") if c in frame), None
                )
                if state_col and value_col:
                    meta = self._metadata(
                        run,
                        "When did inferred regimes occur?",
                        "Timeline shows model states, not proof of economically distinct regimes.",
                    )
                    output["regime_timeline"].append(
                        regime_time_series(
                            frame[value_col],
                            frame[state_col],
                            meta,
                            title=f"Regime quality — timeline ({run.model})",
                        )
                    )
                    output["occupancy_duration"].append(
                        bar_chart(
                            frame[state_col]
                            .value_counts(normalize=True)
                            .sort_index()
                            .rename("occupancy"),
                            meta,
                            title=f"Regime quality — occupancy ({run.model})",
                            y_label="Share",
                        )
                    )
                    groups = frame[state_col].ne(frame[state_col].shift()).cumsum()
                    durations = frame.groupby(groups)[state_col].agg(["first", "size"])
                    output["occupancy_duration"].append(
                        distribution_by_state(
                            durations["size"],
                            durations["first"],
                            meta,
                            title=f"Regime quality — duration distribution ({run.model})",
                            x_label="Observations",
                        )
                    )
                    transition = pd.crosstab(
                        frame[state_col].shift(), frame[state_col], normalize="index"
                    )
                    output["transition_heatmap"].append(
                        heatmap(
                            transition,
                            meta,
                            title=f"Regime quality — transition heatmap ({run.model})",
                        )
                    )
                probability_cols = [
                    c
                    for c in frame
                    if str(c).lower().startswith(("prob_", "probability_", "state_probability"))
                ]
                if probability_cols:
                    meta = self._metadata(
                        run,
                        "How uncertain are real-time regime assignments?",
                        "Filtered probabilities use only information available at each timestamp.",
                        probability="filtered",
                    )
                    output["filtered_probabilities"].append(
                        probability_area_chart(
                            frame[probability_cols],
                            meta,
                            title=f"Filtered regime probabilities ({run.model})",
                        )
                    )
                equity_cols = [
                    c
                    for c in frame
                    if str(c).lower() in {"equity", "equity_curve", "cumulative_return", "drawdown"}
                ]
                if equity_cols:
                    meta = self._metadata(
                        run,
                        "Is the regime signal economically useful?",
                        "Equity and drawdown are economic outcomes and do not establish "
                        "statistical fit.",
                    )
                    output["equity_drawdown"].append(
                        line_chart(
                            frame[equity_cols],
                            meta,
                            title=f"Economic usefulness — equity and drawdown ({run.model})",
                        )
                    )
        metrics: dict[str, dict[str, float]] = {}
        for run in runs:
            merged = {k: v for value in run.json.get("metrics", []) for k, v in value.items()}
            costs = {
                k: float(v)
                for k, v in merged.items()
                if "cost" in k.lower() and isinstance(v, (int, float))
            }
            if costs:
                metrics[run.model] = costs
        if metrics:
            meta = self._metadata(
                representative,
                "How sensitive is economic usefulness to costs?",
                "Costs are reported separately from gross statistical or regime-quality metrics.",
            )
            output["cost_comparison"].append(
                bar_chart(
                    pd.DataFrame(metrics).T, meta, title="Economic usefulness — cost comparison"
                )
            )
        return output


__all__ = ["DEFAULT_SECTIONS", "ExperimentReportAssembler", "ReportConfiguration"]
