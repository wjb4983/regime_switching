"""Dependency-aware, in-process experiment pipeline."""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime.config.base import load_yaml_mapping
from regime.config.models import ExperimentConfig
from regime.experiments.model_comparison import (
    METRIC_DEFINITIONS,
    discover_model_configs,
    evaluate_predictions,
    flatten_metrics,
    rank_models,
    render_comparison_report,
)
from regime.experiments.hashes import file_hash, stable_hash
from regime.experiments.runner import ExperimentRun, RunRegistry
from regime.experiments.store import ExperimentStore
from regime.logging import sanitize_json
from regime.models.registry import ModelRegistryError, model_configuration, model_spec
from regime.reporting.report import ReportBuilder
from regime.training.runner import train_model


@dataclass(frozen=True)
class StagePlan:
    """One resolved node in an experiment dependency graph."""

    key: str
    kind: str
    config: Mapping[str, Any]
    dependencies: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentPipeline:
    """Execute standard experiment stages and register their artifact lineage."""

    registry: RunRegistry = field(default_factory=RunRegistry)

    @classmethod
    def from_root(cls, root: str | Path) -> ExperimentPipeline:
        return cls(RunRegistry(ExperimentStore(root)))

    def plan(self, path: str | Path) -> tuple[ExperimentConfig, list[StagePlan]]:
        config_path = Path(path).expanduser().resolve()
        raw = load_yaml_mapping(config_path)
        raw = self._expand_model_catalog(raw, config_path.parent)
        config = ExperimentConfig.model_validate(raw)
        resolve = lambda section: self._resolve(section, config_path.parent)  # noqa: E731
        plans: list[StagePlan] = []
        if config.source is not None:
            plans.append(StagePlan("source", "source", resolve(config.source)))
        if config.features is not None:
            plans.append(StagePlan("features", "features", resolve(config.features), ("source",)))
        model_dependency = "features" if config.features is not None else "source"
        for family, entries in (("baseline", config.baselines), ("candidate", config.candidates)):
            for index, entry in enumerate(entries):
                item = resolve(entry)
                name = str(item.get("model", f"{family}-{index}"))
                label = str(entry.get("label", item.get("label", name)))
                subgroup = str(entry.get("family", item.get("family", family)))
                plans.append(
                    StagePlan(
                        f"{family}:{label}",
                        "model",
                        item,
                        (model_dependency,),
                        {"family": family, "subfamily": subgroup, "label": label},
                    )
                )
        model_keys = tuple(item.key for item in plans if item.kind == "model")
        if config.validation:
            plans.append(
                StagePlan(
                    "validation",
                    "validation",
                    config.validation.model_dump(mode="json"),
                    model_keys,
                )
            )
        if config.evaluation:
            dependency = (
                "validation" if any(p.key == "validation" for p in plans) else model_keys[-1]
            )
            plans.append(
                StagePlan(
                    "evaluation",
                    "evaluation",
                    config.evaluation.model_dump(mode="json"),
                    (dependency, *model_keys),
                )
            )
        plans.append(StagePlan("comparison", "comparison", config.comparison, ("evaluation",)))
        if config.report is not None:
            plans.append(
                StagePlan(
                    "report", "report", config.report.model_dump(mode="json"), ("comparison",)
                )
            )
        return config, plans

    def dry_run(self, path: str | Path) -> dict[str, Any]:
        """Validate graph, model imports, dependencies, inputs, and output collisions."""
        config, plans = self.plan(path)
        valid_plans, invalid_models = self._validated_model_plans(plans)
        keys: set[str] = set()
        outputs: set[Path] = set()
        stages = [stage for stage in plans if stage.kind != "model"] + list(valid_plans)
        ordered = {stage.key: stage for stage in plans}
        for stage in plans:
            if stage.kind == "model" and stage.key not in {item.key for item in valid_plans}:
                keys.add(stage.key)
                continue
            missing = set(stage.dependencies) - keys
            if missing:
                raise ValueError(
                    f"Stage {stage.key!r} has unresolved dependencies: {sorted(missing)}"
                )
            keys.add(stage.key)
            output = stage.config.get("output") or stage.config.get("output_dir")
            if output:
                destination = Path(str(output)).expanduser().resolve()
                if destination in outputs:
                    raise ValueError(f"Output collision: {destination}")
                outputs.add(destination)
        return {
            "status": "valid",
            "name": config.name,
            "stages": [p.key for p in plans],
            "runnable_models": [stage.metadata.get("label", stage.key) for stage in valid_plans],
            "skipped_models": invalid_models,
        }

    def run(self, path: str | Path, *, resume: bool = True) -> dict[str, Any]:
        """Execute the graph, resuming only hash-identical stage outputs."""
        config, plans = self.plan(path)
        validation = self.dry_run(path)
        valid_model_keys = {
            f"{stage.key}" for stage in plans if stage.kind == "model"
        } - {f"{item['key']}" for item in validation.get("skipped_models", [])}
        store = self.registry.store
        group_id = store.create_group(config.name, description="experiment pipeline")
        previous = store.latest_run(f"experiment:{config.name}") if resume else None
        if previous is not None:
            parent = ExperimentRun(
                store, str(previous["run_id"]), Path(store.root), self.registry.tracker
            )
            store.update_run(parent.run_id, "running")
        else:
            parent = self.registry.start(
                name=f"experiment:{config.name}", group=config.name, config=config.stable_dict()
            )
        checkpoint = parent.load_checkpoint() if resume else None
        states: dict[str, Any] = (
            dict(checkpoint.get("stages", {})) if isinstance(checkpoint, dict) else {}
        )
        artifacts: dict[str, dict[str, str]] = {}
        invalidated = False
        for stage in plans:
            if stage.kind == "model" and stage.key not in valid_model_keys:
                payload = {
                    "status": "skipped",
                    "model": stage.config.get("model"),
                    "label": stage.metadata.get("label"),
                    "reason": next(
                        (
                            item["reason"]
                            for item in validation.get("skipped_models", [])
                            if item["key"] == stage.key
                        ),
                        "invalid model configuration",
                    ),
                }
                artifacts[stage.key] = {
                    "path": str(self._write_stage_stub(config.work_dir, stage.key, payload)),
                    "hash": stable_hash(payload),
                    "run_id": "skipped",
                }
                states[stage.key] = {**artifacts[stage.key], "signature": stable_hash(payload)}
                continue
            upstream = {key: artifacts[key]["hash"] for key in stage.dependencies}
            signature = stable_hash({"config": stage.config, "upstream": upstream})
            saved = states.get(stage.key, {})
            path_value = Path(str(saved.get("path", "")))
            valid = (
                not invalidated
                and saved.get("signature") == signature
                and path_value.is_file()
                and file_hash(path_value) == saved.get("hash")
            )
            if valid:
                artifacts[stage.key] = {
                    "path": str(path_value),
                    "hash": str(saved["hash"]),
                    "run_id": str(saved["run_id"]),
                }
                continue
            invalidated = True
            child = self.registry.start(
                name=f"{config.name}:{stage.key}", group=config.name, config=dict(stage.config)
            )
            try:
                output = self._execute(stage, child, artifacts, config.work_dir)
                digest = file_hash(output)
                child.store.add_artifact(
                    child.run_id,
                    self._artifact_kind(stage.kind),
                    output,
                    artifact_hash=digest,
                    metadata={
                        "parent_run_id": parent.run_id,
                        "stage": stage.key,
                        "upstream_run_ids": [artifacts[k]["run_id"] for k in stage.dependencies],
                        "stage_metadata": dict(stage.metadata),
                    },
                )
                child.store.update_run(child.run_id, "completed")
            except Exception:
                child.store.update_run(child.run_id, "failed")
                store.update_run(parent.run_id, "failed")
                raise
            artifacts[stage.key] = {"path": str(output), "hash": digest, "run_id": child.run_id}
            states[stage.key] = {**artifacts[stage.key], "signature": signature}
            parent.save_checkpoint({"group_id": group_id, "stages": states})
        summary = parent.log_json(
            "manifest", "experiment.json", {"group_id": group_id, "stages": artifacts}
        )
        store.update_run(parent.run_id, "completed")
        return {
            "status": "completed",
            "run_id": parent.run_id,
            "group_id": group_id,
            "manifest": str(summary),
            "stages": artifacts,
        }

    @staticmethod
    def _resolve(section: Mapping[str, Any], base: Path) -> dict[str, Any]:
        reference = section.get("config")
        loaded = load_yaml_mapping(base / str(reference)) if reference else {}
        return {
            **loaded,
            **{k: v for k, v in section.items() if k not in {"config", "label", "family"}},
        }

    @staticmethod
    def _expand_model_catalog(raw: Mapping[str, Any], base: Path) -> dict[str, Any]:
        payload = dict(raw)
        catalog = payload.pop("model_catalog", None)
        if not catalog:
            return payload
        root = (base / str(catalog.get("root", "../models"))).resolve()
        discovered = discover_model_configs(root, relative_to=base)
        payload["baselines"] = []
        payload["candidates"] = discovered
        return payload

    @staticmethod
    def _write_stage_stub(work_dir: Path, key: str, payload: Mapping[str, Any]) -> Path:
        root = Path(work_dir) / "pipeline" / f"skipped-{key.replace(':', '-')}"
        root.mkdir(parents=True, exist_ok=True)
        output = root / "skipped.json"
        output.write_text(json.dumps(sanitize_json(payload), indent=2, default=str), encoding="utf-8")
        return output

    @staticmethod
    def _validated_model_plans(plans: list[StagePlan]) -> tuple[list[StagePlan], list[dict[str, Any]]]:
        valid: list[StagePlan] = []
        invalid: list[dict[str, Any]] = []
        for stage in plans:
            if stage.kind != "model":
                continue
            try:
                spec = model_spec(str(stage.config.get("model", "")))
                model_configuration(spec.name, stage.config)
                absent = [m for m in spec.dependency_modules if importlib.util.find_spec(m) is None]
                if absent:
                    raise ModelRegistryError(
                        f"requires missing modules: {', '.join(absent)}"
                    )
                valid.append(stage)
            except (ModelRegistryError, ValueError) as error:
                invalid.append(
                    {
                        "key": stage.key,
                        "model": stage.config.get("model"),
                        "label": stage.metadata.get("label", stage.key),
                        "reason": str(error),
                    }
                )
        return valid, invalid

    @staticmethod
    def _artifact_kind(kind: str) -> str:
        return {
            "source": "data",
            "features": "features",
            "model": "model",
            "validation": "metrics",
            "evaluation": "metrics",
            "comparison": "comparison",
            "report": "report",
        }[kind]

    def _execute(
        self,
        stage: StagePlan,
        run: ExperimentRun,
        upstream: Mapping[str, Mapping[str, str]],
        work_dir: Path,
    ) -> Path:
        root = Path(work_dir) / "pipeline" / run.run_id
        root.mkdir(parents=True, exist_ok=True)
        if stage.kind == "source":
            cfg = stage.config
            rng = np.random.default_rng(int(cfg.get("seed", 42)))
            count = int(cfg.get("observations", 100))
            states = cfg.get("states", [{"mean": 0.0, "volatility": 0.01}])
            transition = np.asarray(
                cfg.get("transition_matrix", np.full((len(states), len(states)), 1.0 / len(states))),
                dtype=float,
            )
            transition = transition / transition.sum(axis=1, keepdims=True)
            labels = np.empty(count, dtype=int)
            labels[0] = int(rng.integers(0, len(states)))
            for index in range(1, count):
                labels[index] = int(rng.choice(len(states), p=transition[labels[index - 1]]))
            returns = np.array(
                [
                    rng.normal(states[i].get("mean", 0), states[i].get("volatility", 0.01))
                    for i in labels
                ]
            )
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        str(cfg.get("start", "2020-01-01")),
                        periods=count,
                        freq=str(cfg.get("frequency", "1D")),
                        tz="UTC",
                    ),
                    "return_1d": returns,
                    "close": 100 * np.exp(np.cumsum(returns)),
                    "true_state": labels,
                }
            )
            output = root / "data.parquet"
            frame.to_parquet(output, index=False)
            return output
        if stage.kind == "features":
            frame = pd.read_parquet(upstream[stage.dependencies[0]]["path"])
            if "return_1d" not in frame:
                frame["return_1d"] = frame["close"].pct_change()
            frame["realized_volatility_21d"] = frame["return_1d"].rolling(
                21, min_periods=2
            ).std().fillna(0) * np.sqrt(252)
            output = root / "features.parquet"
            frame.to_parquet(output, index=False)
            return output
        if stage.kind == "model":
            cfg = dict(stage.config)
            cfg["input"] = upstream[stage.dependencies[0]]["path"]
            cfg["output"] = str(root / "model")
            summary_path = root / "model_summary.json"
            try:
                result = train_model(run, cfg)
                payload = {
                    "status": "completed",
                    "model": result["model"],
                    "label": stage.metadata.get("label", result["model"]),
                    "family": stage.metadata.get("subfamily", stage.metadata.get("family", "model")),
                    "artifacts": result["artifacts"],
                    "hashes": result["hashes"],
                    "prediction_capability": result["prediction_capability"],
                }
            except Exception as error:
                payload = {
                    "status": "failed",
                    "model": cfg.get("model"),
                    "label": stage.metadata.get("label", cfg.get("model")),
                    "family": stage.metadata.get("subfamily", stage.metadata.get("family", "model")),
                    "reason": str(error),
                }
            summary_path.write_text(
                json.dumps(sanitize_json(payload), indent=2, default=str), encoding="utf-8"
            )
            run.store.add_result(run.run_id, "model", payload)
            return summary_path
        if stage.kind in {"validation", "evaluation", "comparison"}:
            payload: dict[str, Any] = {
                "stage": stage.kind,
                "upstream": upstream,
                "config": dict(stage.config),
            }
            if stage.kind == "validation":
                payload["contract"] = {
                    "strategy": stage.config.get("strategy"),
                    "n_splits": stage.config.get("n_splits"),
                    "embargo_periods": stage.config.get("embargo_periods"),
                }
            elif stage.kind == "evaluation":
                metrics_requested = [str(item) for item in stage.config.get("metrics", [])]
                models: list[dict[str, Any]] = []
                data_period = "Synthetic benchmark"
                for key, artifact in upstream.items():
                    if not key.startswith(("baseline:", "candidate:")):
                        continue
                    summary = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
                    if summary.get("status") != "completed":
                        models.append(
                            {
                                "model": summary.get("label") or summary.get("model") or key,
                                "family": summary.get("family"),
                                "status": summary.get("status", "failed"),
                                "reason": summary.get("reason", "model stage failed"),
                                "unsupported_metrics": {
                                    metric: "model did not complete training"
                                    for metric in metrics_requested
                                },
                            }
                        )
                        continue
                    predictions_path = Path(str(summary["artifacts"]["in_sample_predictions.parquet"]))
                    predictions = pd.read_parquet(predictions_path)
                    if "timestamp" in predictions and len(predictions):
                        values = pd.to_datetime(predictions["timestamp"], utc=True, errors="coerce")
                        data_period = f"{values.min().date()} to {values.max().date()}"
                    diagnostics_path = Path(str(summary["artifacts"]["training_diagnostics.json"]))
                    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
                    evaluated = evaluate_predictions(
                        predictions,
                        model_name=str(summary.get("label") or summary["model"]),
                        fit_seconds=diagnostics.get("fit_seconds"),
                        metrics=metrics_requested,
                    )
                    evaluated["family"] = summary.get("family")
                    evaluated["evaluation_frame"] = str(predictions_path)
                    evaluated["artifacts"] = summary.get("artifacts", {})
                    models.append(evaluated)
                metric_rows = flatten_metrics(
                    {
                        str(model["model"]): dict(model.get("metrics", {}))
                        for model in models
                        if model.get("status") == "completed"
                    }
                )
                csv_path = root / "comparison_metrics.csv"
                metric_rows.to_csv(csv_path, index=False)
                payload["data_period"] = data_period
                payload["models"] = models
                payload["metric_catalog"] = {
                    name: {
                        "family": definition.family,
                        "direction": definition.direction,
                        "title": definition.title,
                        "explanation": definition.explanation,
                        "meaningful_when": definition.meaningful_when,
                    }
                    for name, definition in METRIC_DEFINITIONS.items()
                }
                payload["summary"] = {
                    "completed_models": sum(model.get("status") == "completed" for model in models),
                    "failed_models": sum(model.get("status") != "completed" for model in models),
                    "metrics_csv": str(csv_path),
                }
                run.store.add_result(run.run_id, "metrics", payload)
            else:
                evaluation = json.loads(Path(upstream[stage.dependencies[0]]["path"]).read_text(encoding="utf-8"))
                payload.update(rank_models(evaluation))
                payload["config"] = dict(stage.config)
                payload["models"] = evaluation.get("models", [])
                payload["metric_catalog"] = evaluation.get("metric_catalog", {})
                payload["model_names"] = [model.get("model") for model in payload["models"]]
                payload["data_period"] = evaluation.get("data_period", "Synthetic benchmark")
                csv_rows = flatten_metrics(
                    {
                        str(model["model"]): dict(model.get("metrics", {}))
                        for model in payload["models"]
                        if model.get("status") == "completed"
                    }
                )
                metrics_csv = root / "comparison_metrics.csv"
                csv_rows.to_csv(metrics_csv, index=False)
                report_path = root / "comparison_report.html"
                render_comparison_report(
                    payload,
                    title="Daily market regime model comparison",
                    output=report_path,
                )
                payload["artifacts"] = {
                    "comparison_metrics_csv": str(metrics_csv),
                    "comparison_report_html": str(report_path),
                }
            output = root / f"{stage.kind}.json"
            output.write_text(json.dumps(sanitize_json(payload), indent=2, default=str))
            run.store.add_result(run.run_id, stage.kind, payload)
            return output
        comparison = json.loads(Path(upstream[stage.dependencies[0]]["path"]).read_text(encoding="utf-8"))
        report_path = Path(
            str(
                comparison.get("artifacts", {}).get("comparison_report_html", root / "comparison_report.html")
            )
        )
        output_dir = stage.config.get("output_dir")
        destination = (
            Path(str(output_dir)).expanduser().resolve() / "model_comparison.html"
            if output_dir
            else root / "report.html"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if report_path.is_file():
            shutil.copyfile(report_path, destination)
            return destination
        builder = ReportBuilder(
            str(stage.config.get("title", "Model comparison")),
            subtitle="Registered experiment pipeline",
        )
        return builder.write(destination)


__all__ = ["ExperimentPipeline", "StagePlan"]
