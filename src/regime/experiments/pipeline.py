"""Dependency-aware, in-process experiment pipeline."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime.config.base import load_yaml_mapping
from regime.config.models import ExperimentConfig
from regime.experiments.hashes import file_hash, stable_hash
from regime.experiments.runner import ExperimentRun, RunRegistry
from regime.experiments.store import ExperimentStore
from regime.models.registry import model_configuration, model_spec
from regime.reporting.report import ReportBuilder
from regime.training.runner import train_model


@dataclass(frozen=True)
class StagePlan:
    """One resolved node in an experiment dependency graph."""

    key: str
    kind: str
    config: Mapping[str, Any]
    dependencies: tuple[str, ...] = ()


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
                plans.append(StagePlan(f"{family}:{name}", "model", item, (model_dependency,)))
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
        keys: set[str] = set()
        outputs: set[Path] = set()
        for stage in plans:
            missing = set(stage.dependencies) - keys
            if missing:
                raise ValueError(
                    f"Stage {stage.key!r} has unresolved dependencies: {sorted(missing)}"
                )
            keys.add(stage.key)
            if stage.kind == "model":
                spec = model_spec(str(stage.config.get("model", "")))
                model_configuration(spec.name, stage.config)
                absent = [m for m in spec.dependency_modules if importlib.util.find_spec(m) is None]
                if absent:
                    raise ValueError(
                        f"Model {spec.name!r} requires missing modules: {', '.join(absent)}"
                    )
            output = stage.config.get("output") or stage.config.get("output_dir")
            if output:
                destination = Path(str(output)).expanduser().resolve()
                if destination in outputs:
                    raise ValueError(f"Output collision: {destination}")
                outputs.add(destination)
        return {"status": "valid", "name": config.name, "stages": [p.key for p in plans]}

    def run(self, path: str | Path, *, resume: bool = True) -> dict[str, Any]:
        """Execute the graph, resuming only hash-identical stage outputs."""
        config, plans = self.plan(path)
        self.dry_run(path)
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
        return {**loaded, **{k: v for k, v in section.items() if k != "config"}}

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
            labels = rng.integers(0, len(states), count)
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
            result = train_model(run, cfg)
            artifacts = result["artifacts"]
            return Path(
                str(
                    artifacts.get("in_sample_predictions.parquet")
                    or artifacts.get("model.pkl")
                    or artifacts.get("model.json")
                )
            )
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
                metrics: dict[str, dict[str, float]] = {}
                for key, artifact in upstream.items():
                    candidate = Path(artifact["path"])
                    if candidate.suffix != ".parquet":
                        continue
                    predictions = pd.read_parquet(candidate)
                    if "state" in predictions and len(predictions) > 1:
                        persistence = float(
                            (
                                predictions["state"].to_numpy()[1:]
                                == predictions["state"].to_numpy()[:-1]
                            ).mean()
                        )
                        switching = 1.0 - persistence
                        metrics[key] = {
                            "regime_persistence": persistence,
                            "switching_frequency": switching,
                        }
                payload["metrics"] = metrics
                run.store.add_result(run.run_id, "metrics", metrics)
            else:
                evaluation = json.loads(Path(upstream[stage.dependencies[0]]["path"]).read_text())
                payload["models"] = evaluation.get("metrics", {})
            output = root / f"{stage.kind}.json"
            output.write_text(json.dumps(payload, indent=2, default=str))
            run.store.add_result(run.run_id, stage.kind, payload)
            return output
        builder = ReportBuilder(
            str(stage.config.get("title", "Model comparison")),
            subtitle="Registered experiment pipeline",
        )
        return builder.write(root / "report.html")


__all__ = ["ExperimentPipeline", "StagePlan"]
