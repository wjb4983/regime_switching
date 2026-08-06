"""Artifact-driven, contract-safe comparisons for completed experiment runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime.evaluation.comparison import (
    block_bootstrap_indices,
    diebold_mariano_test,
    false_discovery_control,
    stationary_bootstrap_indices,
)
from regime.evaluation.economic import METRICS as ECONOMIC_METRICS
from regime.evaluation.statistical import METRICS as STATISTICAL_METRICS
from regime.experiments.store import ExperimentStore

Direction = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class ComparisonConfiguration:
    """All choices that affect ranking or statistical inference."""

    primary_metric: str = "loss"
    benchmark: str | None = None
    primary_loss: str = "loss"
    direction: Direction | None = None
    bootstrap_method: Literal["block", "stationary"] = "stationary"
    block_length: float = 10.0
    seed: int = 0
    alpha: float = 0.05
    multiple_testing_correction: str = "holm"
    n_bootstrap: int = 1000


@dataclass(frozen=True)
class LoadedRun:
    run_id: str
    model: str
    parameter_hash: str | None
    runtime: float | None
    metrics: dict[str, Any]
    contract: dict[str, Any]
    observations: pd.DataFrame


@dataclass(frozen=True)
class ComparisonResult:
    experiment_group: str
    configuration: ComparisonConfiguration
    table: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    statistical_tests: tuple[dict[str, Any], ...]
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        return {
            "experiment_group": self.experiment_group,
            "configuration": asdict(self.configuration),
            "table": list(self.table),
            "diagnostics": list(self.diagnostics),
            "statistical_tests": list(self.statistical_tests),
            "artifact_path": self.artifact_path,
        }


class ComparisonService:
    """Load, validate, rank, test, and persist an experiment-group comparison."""

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store

    def compare(
        self,
        experiment_group: str,
        configuration: ComparisonConfiguration | None = None,
        *,
        persist: bool = True,
    ) -> ComparisonResult:
        config = configuration or ComparisonConfiguration()
        group_id, runs = self._load(experiment_group)
        if not runs:
            raise ValueError(f"experiment group has no completed runs: {experiment_group}")
        benchmark = self._benchmark(runs, config.benchmark)
        direction = config.direction or self._direction(config.primary_metric)
        reference_contract = benchmark.contract
        rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        compatible: list[LoadedRun] = []
        for run in runs:
            differences = self.contract_differences(reference_contract, run.contract)
            missing = config.primary_metric not in run.metrics
            status = "compatible"
            if differences:
                status = "incompatible"
            elif missing:
                status = "missing_metric"
            else:
                compatible.append(run)
            diagnostics.append(
                {
                    "run_id": run.run_id,
                    "model": run.model,
                    "status": status,
                    "contract_differences": differences,
                    "missing_metrics": [config.primary_metric] if missing else [],
                }
            )
            rows.append(self._row(run, config, status))

        ordered = sorted(
            compatible,
            key=lambda item: float(item.metrics[config.primary_metric]),
            reverse=direction == "maximize",
        )
        ranks = {run.run_id: rank for rank, run in enumerate(ordered, 1)}
        for row in rows:
            row["rank"] = ranks.get(str(row["run_id"]))

        tests = self._tests(compatible, benchmark, config, direction)
        result = ComparisonResult(
            experiment_group,
            config,
            tuple(rows),
            tuple(diagnostics),
            tuple(tests),
        )
        if persist:
            path = self._persist(group_id, experiment_group, result)
            result = ComparisonResult(
                result.experiment_group,
                result.configuration,
                result.table,
                result.diagnostics,
                result.statistical_tests,
                str(path),
            )
        return result

    @staticmethod
    def contract_differences(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
        """Return precise leaf-level contract differences, including missing fields."""
        differences: list[dict[str, Any]] = []

        def visit(a: Any, b: Any, path: str) -> None:
            if isinstance(a, dict) and isinstance(b, dict):
                for key in sorted(set(a) | set(b)):
                    visit(a.get(key, _MISSING), b.get(key, _MISSING), f"{path}.{key}".lstrip("."))
            elif a != b:
                differences.append(
                    {
                        "field": path,
                        "reference": None if a is _MISSING else a,
                        "candidate": None if b is _MISSING else b,
                        "reference_missing": a is _MISSING,
                        "candidate_missing": b is _MISSING,
                    }
                )

        visit(left, right, "")
        return differences

    def _load(self, group: str) -> tuple[str, list[LoadedRun]]:
        with self.store.connect() as con:
            group_row = con.execute(
                "SELECT group_id FROM experiment_groups WHERE name=? OR group_id=?", (group, group)
            ).fetchone()
            if group_row is None:
                raise FileNotFoundError(f"Experiment group not found: {group}")
            group_id = str(group_row["group_id"])
            rows = con.execute(
                "SELECT * FROM runs WHERE group_id=? AND status='completed' "
                "AND (name IS NULL OR name NOT LIKE 'comparison:%') ORDER BY started_at",
                (group_id,),
            ).fetchall()
            loaded: list[LoadedRun] = []
            for row in rows:
                artifacts = con.execute(
                    "SELECT kind, path, metadata_json FROM artifacts WHERE run_id=? "
                    "ORDER BY created_at",
                    (row["run_id"],),
                ).fetchall()
                results = con.execute(
                    "SELECT kind, value_json FROM results WHERE run_id=? ORDER BY created_at",
                    (row["run_id"],),
                ).fetchall()
                loaded.append(self._load_run(dict(row), artifacts, results))
        return group_id, loaded

    def _load_run(self, row: dict[str, Any], artifacts: Any, results: Any) -> LoadedRun:
        metadata = json.loads(row["metadata_json"])
        metrics: dict[str, Any] = {}
        contract: dict[str, Any] = {}
        frames: list[pd.DataFrame] = []
        for artifact in artifacts:
            path = Path(artifact["path"])
            if not path.is_absolute() and not path.exists():
                path = Path(self.store.root) / path
            if not path.exists():
                continue
            kind = str(artifact["kind"])
            name = path.name.lower()
            if kind == "metrics":
                metrics.update(self._json_mapping(path))
            elif kind == "comparison" and "contract" in name:
                contract.update(self._json_mapping(path))
            elif kind in {"predictions", "data"}:
                frames.append(self._table(path))
        for item in results:
            value = json.loads(item["value_json"])
            if isinstance(value, dict) and item["kind"] in {"metrics", "evaluation", "result"}:
                metrics.update(value.get("metrics", value))
        observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        model = str(metadata.get("model") or row.get("name") or row["run_id"])
        runtime = None
        if row.get("completed_at") is not None:
            runtime = float(row["completed_at"] - row["started_at"])
        return LoadedRun(
            str(row["run_id"]),
            model,
            row.get("model_hash") or row.get("config_hash"),
            runtime,
            metrics,
            contract,
            observations,
        )

    @staticmethod
    def _json_mapping(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _table(path: Path) -> pd.DataFrame:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(value if isinstance(value, list) else value.get("records", value))

    @staticmethod
    def _benchmark(runs: list[LoadedRun], requested: str | None) -> LoadedRun:
        if requested is None:
            return runs[0]
        matches = [run for run in runs if requested in {run.run_id, run.model}]
        if not matches:
            raise ValueError(f"benchmark run or model not found: {requested}")
        return matches[0]

    @staticmethod
    def _direction(metric: str) -> Direction:
        descriptor = {**STATISTICAL_METRICS, **ECONOMIC_METRICS}.get(metric)
        if descriptor is not None and descriptor.direction != "diagnostic":
            return descriptor.direction
        return (
            "minimize"
            if any(word in metric.lower() for word in ("loss", "error", "rmse", "mae"))
            else "maximize"
        )

    @staticmethod
    def _row(run: LoadedRun, config: ComparisonConfiguration, status: str) -> dict[str, Any]:
        metric = config.primary_metric
        downstream = {
            key: value
            for key, value in run.metrics.items()
            if key != metric and isinstance(value, (int, float))
        }
        uncertainty = run.metrics.get(f"{metric}_uncertainty", run.metrics.get(f"{metric}_std"))
        if uncertainty is None and metric in run.observations:
            values = pd.to_numeric(run.observations[metric], errors="coerce").dropna().to_numpy()
            if values.size > 1:
                if config.bootstrap_method == "block":
                    indexes = block_bootstrap_indices(
                        len(values),
                        block_length=round(config.block_length),
                        n_bootstrap=config.n_bootstrap,
                        random_state=config.seed,
                    )
                else:
                    indexes = stationary_bootstrap_indices(
                        len(values),
                        average_block_length=config.block_length,
                        n_bootstrap=config.n_bootstrap,
                        random_state=config.seed,
                    )
                uncertainty = float(np.std(values[indexes].mean(axis=1), ddof=1))
        stability_values = [
            float(value)
            for key, value in run.metrics.items()
            if ("fold" in key or "seed" in key)
            and metric in key
            and isinstance(value, (int, float))
        ]
        if not stability_values and metric in run.observations:
            groups = [name for name in ("fold", "seed") if name in run.observations]
            if groups:
                stability_values = list(
                    run.observations.groupby(groups, dropna=False)[metric].mean().astype(float)
                )
        stability = float(np.std(stability_values, ddof=0)) if stability_values else None
        return {
            "run_id": run.run_id,
            "model": run.model,
            "parameter_hash": run.parameter_hash,
            "primary_metric": run.metrics.get(metric),
            "uncertainty": uncertainty,
            "rank": None,
            "runtime_seconds": run.runtime,
            "stability": stability,
            "downstream_metrics": downstream,
            "status": status,
        }

    def _tests(
        self,
        runs: list[LoadedRun],
        benchmark: LoadedRun,
        config: ComparisonConfiguration,
        direction: Direction,
    ) -> list[dict[str, Any]]:
        # DM is documented for lower-is-better, aligned loss series. Do not reinterpret
        # arbitrary aggregate metrics or manufacture alignment when timestamps differ.
        if direction != "minimize" or benchmark not in runs:
            return []
        tests: list[dict[str, Any]] = []
        for run in runs:
            if run is benchmark:
                continue
            aligned = self._aligned_loss(run, benchmark, config.primary_loss)
            if aligned is None or len(aligned) < 3:
                continue
            try:
                tested = diebold_mariano_test(aligned.iloc[:, 0], aligned.iloc[:, 1])
            except ValueError:
                continue
            tests.append(
                {
                    "run_id": run.run_id,
                    "benchmark_run_id": benchmark.run_id,
                    "method": tested.method,
                    "statistic": tested.statistic,
                    "p_value": tested.p_value,
                    "adjusted_p_value": None,
                    "rejected": None,
                    "n_obs": tested.n_obs,
                }
            )
        if tests:
            adjusted = false_discovery_control(
                [float(test["p_value"]) for test in tests],
                alpha=config.alpha,
                method=config.multiple_testing_correction,  # type: ignore[arg-type]
            )
            for test, p_value, rejected in zip(
                tests, adjusted.adjusted_p_values, adjusted.rejected, strict=True
            ):
                test["adjusted_p_value"] = p_value
                test["rejected"] = rejected
        return tests

    @staticmethod
    def _aligned_loss(left: LoadedRun, right: LoadedRun, column: str) -> pd.DataFrame | None:
        if column not in left.observations or column not in right.observations:
            return None
        index_columns = [
            name
            for name in ("timestamp", "date", "forecast_origin")
            if name in left.observations and name in right.observations
        ]
        if not index_columns:
            return None
        key = index_columns[0]
        a = left.observations[[key, column]].rename(columns={column: "candidate"})
        b = right.observations[[key, column]].rename(columns={column: "benchmark"})
        merged = a.merge(b, on=key, how="inner", validate="one_to_one").dropna()
        if len(merged) != len(a.dropna()) or len(merged) != len(b.dropna()):
            return None
        return merged[["candidate", "benchmark"]]

    def _persist(self, group_id: str, name: str, result: ComparisonResult) -> Path:
        digest = hashlib.sha256(
            json.dumps(asdict(result.configuration), sort_keys=True).encode()
        ).hexdigest()[:12]
        directory = Path(self.store.root) / "comparisons" / group_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"comparison-{digest}.json"
        path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        run_id = self.store.create_run(
            group_id=group_id,
            name=f"comparison:{name}",
            metadata={"parent_experiment_group": group_id},
        )
        self.store.add_artifact(
            run_id,
            "comparison",
            path,
            metadata={"view": "experiment_comparison", "parent_experiment_group": group_id},
        )
        self.store.add_result(run_id, "comparison", result.to_dict())
        self.store.update_run(run_id, "completed")
        return path


_MISSING = object()

__all__ = ["ComparisonConfiguration", "ComparisonResult", "ComparisonService", "LoadedRun"]
