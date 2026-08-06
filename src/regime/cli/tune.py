"""Hyperparameter tuning command backed by production train/evaluate services."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from regime.cli.common import command_errors, config_option, config_workflow, resume_option
from regime.evaluation.service import evaluate_config
from regime.experiments.runner import ExperimentRun
from regime.training.runner import train_model
from regime.tuning.config import TuningConfig
from regime.tuning.runner import StudyConfig, optimize


def _run_tuning(run: ExperimentRun, config: TuningConfig) -> Mapping[str, Any]:
    study_config = StudyConfig(
        name=config.name,
        storage=config.storage,
        algorithm=config.algorithm,
        directions=tuple(item.direction for item in config.objectives),
        seed=config.seed_policy.sampler,
        n_trials=config.trials,
        timeout=config.timeout,
        n_jobs=config.parallelism,
    )

    def objective(parameters: Mapping[str, Any], trial: Any) -> tuple[float, ...] | float:
        resolved = dict(config.base_model)
        resolved.update(parameters)
        if config.seed_policy.model is not None:
            resolved["random_seed"] = config.seed_policy.model
        resolved["output"] = str(run.artifact_path("model", f"trial-{trial.number}"))
        trial.set_user_attr("resolved_configuration", resolved)
        trial.set_user_attr("evaluation_contract", dict(config.validation))
        try:
            trained = train_model(run, resolved)
            evaluation = dict(config.validation)
            evaluation["source"] = {"model": resolved["model"], "model_parameters": resolved}
            evaluated = evaluate_config(run, evaluation)
            summary = evaluated["metric_summary"]
            metrics = next(iter(summary.values()))
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("folds", evaluation.get("splitter", {}))
            model_hash = trained["hashes"].get("model.pkl") or trained["hashes"].get("model.json")
            trial.set_user_attr("model_hash", model_hash)
            violations = tuple(
                float(metrics[name]) - limit for name, limit in config.constraints.items()
            )
            trial.set_user_attr("constraint_violations", violations)
            values = tuple(float(metrics[item.metric]) for item in config.objectives)
            return values[0] if len(values) == 1 else values
        except Exception as error:
            trial.set_user_attr("failure", {"type": type(error).__name__, "message": str(error)})
            raise

    study = optimize(
        study_config, config.search_space, objective, registry=run.store, group_name=config.name
    )
    pareto = [
        {"trial": item.number, "values": list(item.values), "parameters": item.params}
        for item in study.best_trials
    ]
    run.log_json(
        "metrics",
        "pareto-set.json",
        {"objectives": [item.metric for item in config.objectives], "trials": pareto},
    )
    return {"study": config.name, "trials": len(study.trials), "pareto_set": pareto}


@command_errors
def tune(
    config: Path = config_option("Tuning search-space YAML."), resume: bool = resume_option()
) -> None:
    """Validate and run a resumable tuning search."""
    parsed = TuningConfig.from_yaml(config)

    def worker(run: ExperimentRun, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        del raw
        return _run_tuning(run, parsed)

    config_workflow("tune", config, resume=resume, worker=worker)
