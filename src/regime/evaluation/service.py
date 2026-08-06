"""Configuration-to-runner orchestration for evaluation commands."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from regime.evaluation.config import EvaluationWorkflowConfig, parse_evaluation_config
from regime.evaluation.runner import (
    DatasetConfig,
    EvaluationConfig,
    EvaluationRunner,
    ModelConfig,
    ValidationConfig,
)
from regime.experiments.runner import ExperimentRun
from regime.models.registry import create_model, model_configuration
from regime.validation.splitters import (
    ExpandingWindowSplitter,
    PurgedTimeSeriesSplitter,
    RollingWindowSplitter,
)


def _frame(path: Path, features: list[str], timestamp: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    data = (
        pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    )
    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Evaluation dataset missing features: {', '.join(sorted(missing))}")
    if timestamp:
        data[timestamp] = pd.to_datetime(data[timestamp], utc=True, errors="raise")
        data = data.sort_values(timestamp).reset_index(drop=True)
    return data.loc[:, features + [c for c in data.columns if c not in features]]


def _splitter(config: EvaluationWorkflowConfig) -> Any:
    split = config.splitter
    common = {
        "validation_size": split.validation_size,
        "test_size": split.test_size,
        "step": split.step,
    }
    if split.kind == "rolling":
        return RollingWindowSplitter(train_size=split.train_size, **common)
    cls = PurgedTimeSeriesSplitter if split.kind == "purged" else ExpandingWindowSplitter
    extra = {"embargo": split.embargo} if split.kind == "purged" else {}
    return cls(initial_train_size=split.initial_train_size, **common, **extra)


def _model_specs(
    run: ExperimentRun, config: EvaluationWorkflowConfig
) -> list[tuple[str, dict[str, Any]]]:
    source = config.source
    if source.model:
        return [(source.model, source.model_parameters)]
    paths: list[Path] = []
    if source.model_artifact:
        paths = [source.model_artifact]
    else:
        with run.store.connect() as connection:
            group = connection.execute(
                "SELECT group_id FROM experiment_groups WHERE name=? OR group_id=?",
                (source.experiment_group, source.experiment_group),
            ).fetchone()
            if group is None:
                raise FileNotFoundError(f"Experiment group not found: {source.experiment_group}")
            rows = connection.execute(
                "SELECT a.path FROM artifacts a JOIN runs r ON r.run_id=a.run_id "
                "WHERE r.group_id=? AND a.kind='model' ORDER BY a.created_at",
                (group["group_id"],),
            ).fetchall()
            paths = [Path(row["path"]) for row in rows]
    if not paths:
        raise FileNotFoundError("No model artifacts were resolved for evaluation")
    result: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        resolved = path.parent / "resolved_configuration.json"
        if not resolved.exists():
            raise FileNotFoundError(f"Model artifact has no resolved configuration: {resolved}")
        record = json.loads(resolved.read_text(encoding="utf-8"))
        result.append((record["model"], record.get("model_configuration", {})))
    return result


def evaluate_config(run: ExperimentRun, raw_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve configured inputs, execute all selected models, and register outputs."""
    # Historical quick-start placeholders did not execute an evaluation. Keep them
    # resumable while requiring the discriminator for every executable configuration.
    if "evaluation_type" not in raw_config and not {"source", "dataset"} & raw_config.keys():
        return {"evaluations": [], "artifact_paths": [], "metric_summary": {}}
    config = parse_evaluation_config(raw_config)
    data = _frame(config.dataset, config.features, config.timestamp_column)
    outputs: list[dict[str, Any]] = []
    for index, (model_name, parameters) in enumerate(_model_specs(run, config)):
        # Factories intentionally reconstruct fresh instances through the public registry.
        def factory(name: str = model_name, settings: dict[str, Any] = parameters) -> Any:
            return create_model(name, settings)

        fit_config = model_configuration(model_name, parameters)
        selected = config.metrics
        quality = selected or [
            "regime_persistence",
            "switching_frequency",
            "state_entropy",
            "probability_entropy",
        ]
        result = EvaluationRunner().run(
            DatasetConfig(data=data.loc[:, config.features], dataset_id=str(config.dataset)),
            ModelConfig(factory, fit_config),
            ValidationConfig(
                splitter=_splitter(config),
                retraining_schedule=config.retraining_schedule,
                execution_delay=config.execution_delay,
            ),
            EvaluationConfig(
                output_dir=config.output_dir,
                run_id=config.run_id if len(outputs) == 0 else f"{config.run_id}-{index}",
                statistical_metrics=tuple(selected),
                regime_quality_metrics=tuple(quality),
                comparison_contract=config.comparison_contract,
                cost_assumptions=config.cost_assumptions,
                downstream_decision_rules=config.decision_rules,
            ),
        )
        artifacts = {
            "predictions": result.predictions_path,
            "metrics": result.metrics_path,
            "diagnostics": result.diagnostics_path,
            "provenance": result.provenance_path,
            "checkpoint": result.checkpoint_path,
            "comparison": result.comparison_contract_path,
        }
        for kind, path in artifacts.items():
            run.store.add_artifact(run.run_id, kind, path)
        outputs.append(
            {"model": model_name, "artifacts": artifacts, "metrics": dict(result.metrics)}
        )
    return {
        "evaluations": outputs,
        "artifact_paths": [p for o in outputs for p in o["artifacts"].values()],
        "metric_summary": {o["model"]: o["metrics"] for o in outputs},
    }


__all__ = ["evaluate_config"]
