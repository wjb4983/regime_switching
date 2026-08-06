"""End-to-end rolling evaluation runner for regime-switching models.

The runner coordinates leakage-aware validation splits, split-local preprocessing,
rolling refits, live-equivalent filtered probabilities, optional offline smoothing,
state alignment, metrics, artifact persistence, provenance capture, and resumable
checkpoints.  The configuration objects are deliberately lightweight dataclasses so
research code can pass concrete callables without introducing a tracking-system or
pipeline dependency.
"""

from __future__ import annotations

import json
import pickle
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
import pandas as pd

from regime.evaluation.alignment import AlignmentDiagnostics, AlignmentMethod, align_states
from regime.evaluation.regime_quality import (
    duration_distribution,
    probability_entropy,
    regime_persistence,
    rolling_refit_stability,
    state_entropy,
    state_occupancy,
    state_recurrence,
    switching_frequency,
    transition_stability,
)
from regime.evaluation.statistical import brier_score, predictive_log_score
from regime.experiments.provenance import RunMetadataRecorder, TimePeriod, stable_hash
from regime.models.base import (
    RegimeInferenceResult,
    RegimeModel,
    RegimeModelConfig,
    UnsupportedModelOperation,
)
from regime.validation.splitters import BaseSplitter, ValidationSplit

T = TypeVar("T")
Json = None | bool | int | float | str | list["Json"] | dict[str, "Json"]


class Transformer(Protocol):
    """Minimal split-specific preprocessing protocol."""

    def fit(self, data: Any, y: Any | None = None) -> Any: ...

    def transform(self, data: Any) -> Any: ...


class MetricsFunction(Protocol):
    """User metric hook receiving the aligned prediction frame and context."""

    def __call__(
        self, predictions: pd.DataFrame, context: Mapping[str, Any]
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset inputs for an evaluation run.

    ``data`` may be supplied directly or lazily via ``loader``.  ``preprocessor_factory``
    is called once per split and fitted only on the training window to prevent leakage.
    """

    data: Any | None = None
    loader: Callable[[], Any] | None = None
    target_column: str | None = None
    timestamp_column: str | None = None
    preprocessor_factory: Callable[[], Transformer] | None = None
    dataset_id: str | None = None
    feature_hash: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Model construction and fitting configuration."""

    model_factory: Callable[[], RegimeModel]
    fit_config: RegimeModelConfig | Mapping[str, Any] = field(default_factory=RegimeModelConfig)
    refit: bool = True
    save_models: bool = True


@dataclass(frozen=True)
class ValidationConfig:
    """Validation splits and retraining semantics."""

    splitter: BaseSplitter | None = None
    splits: Sequence[ValidationSplit] | None = None
    evaluation_role: str = "test"
    retraining_schedule: str = "each_window"
    execution_delay: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation options, persistence locations, and comparison guard assumptions."""

    output_dir: str | Path
    run_id: str = "evaluation"
    produce_smoothed: bool = False
    alignment_method: AlignmentMethod | str | None = None
    statistical_metrics: Sequence[str | MetricsFunction] = ("predictive_log_score",)
    regime_quality_metrics: Sequence[str | MetricsFunction] = (
        "regime_persistence",
        "switching_frequency",
        "state_entropy",
        "probability_entropy",
    )
    comparison_contract: Mapping[str, Any] = field(default_factory=dict)
    cost_assumptions: Mapping[str, Any] = field(default_factory=dict)
    downstream_decision_rules: Mapping[str, Any] = field(default_factory=dict)
    package_names: Sequence[str] | None = ("regime-switching", "numpy", "pandas", "scikit-learn")
    checkpoint_every_window: bool = True
    resume: bool = True


@dataclass(frozen=True)
class WindowEvaluationResult:
    """Artifacts and metrics for one validation window."""

    window_id: int
    predictions_path: str
    diagnostics_path: str
    model_path: str | None
    metrics: Mapping[str, float]
    alignment: AlignmentDiagnostics | None = None


@dataclass(frozen=True)
class EvaluationRunResult:
    """Summary returned by :class:`EvaluationRunner.run`."""

    run_id: str
    output_dir: str
    predictions_path: str
    metrics_path: str
    diagnostics_path: str
    provenance_path: str
    checkpoint_path: str
    comparison_contract_path: str
    windows: tuple[WindowEvaluationResult, ...]
    metrics: Mapping[str, float]


class ComparisonContractError(ValueError):
    """Raised when evaluation runs cannot be compared under a common contract."""


class EvaluationRunner:
    """Run leak-safe rolling/refit evaluation for regime-switching models."""

    _REQUIRED_COMPARISON_FIELDS = (
        "information_set",
        "validation_period",
        "retraining_schedule",
        "execution_delay",
        "cost_assumptions",
        "downstream_decision_rules",
    )

    def run(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        validation_config: ValidationConfig,
        evaluation_config: EvaluationConfig,
    ) -> EvaluationRunResult:
        """Execute or resume a complete evaluation run."""
        self._validate_comparison_contract(validation_config, evaluation_config)
        output_dir = Path(evaluation_config.output_dir) / evaluation_config.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        for child in ("predictions", "diagnostics", "models", "checkpoints"):
            (output_dir / child).mkdir(exist_ok=True)
        checkpoint_path = output_dir / "checkpoints" / "runner_state.json"
        data = self._load_data(dataset_config)
        splits = self._splits(data, validation_config)
        checkpoint_signature = stable_hash(
            {
                "dataset": dataset_config.dataset_id,
                "feature_hash": dataset_config.feature_hash,
                "model": self._jsonable(model_config.fit_config),
                "model_factory": getattr(
                    model_config.model_factory, "__qualname__", repr(model_config.model_factory)
                ),
                "validation": self._jsonable(validation_config),
                "splits": [self._jsonable(split) for split in splits],
                "metrics": {
                    "statistical": self._jsonable(evaluation_config.statistical_metrics),
                    "regime_quality": self._jsonable(evaluation_config.regime_quality_metrics),
                },
            }
        )
        completed = (
            self._load_checkpoint(checkpoint_path, checkpoint_signature)
            if evaluation_config.resume
            else set()
        )
        recorder = RunMetadataRecorder(
            repo=Path.cwd(), package_names=evaluation_config.package_names
        )
        window_results: list[WindowEvaluationResult] = []
        prediction_frames: list[pd.DataFrame] = []
        reference_summary: Mapping[str, Any] | None = None
        previous_model: RegimeModel | None = None
        transition_matrices: list[Sequence[Sequence[float]]] = []
        label_runs: list[np.ndarray] = []

        for window_id, split in enumerate(splits):
            window_started = time.perf_counter()
            pred_path = output_dir / "predictions" / f"window_{window_id:04d}.parquet"
            diag_path = output_dir / "diagnostics" / f"window_{window_id:04d}.json"
            model_path = output_dir / "models" / f"window_{window_id:04d}.pkl"
            if window_id in completed and pred_path.exists() and diag_path.exists():
                frame = pd.read_parquet(pred_path)
                prediction_frames.append(frame)
                window_results.append(
                    self._rehydrate_window(window_id, pred_path, diag_path, model_path)
                )
                continue

            train_data, eval_data = self._prepare_split(
                data, split, dataset_config, validation_config
            )
            model = (
                model_config.model_factory()
                if model_config.refit or previous_model is None
                else previous_model
            )
            model.fit(train_data, self._fit_config(model_config.fit_config))
            previous_model = model
            predictions, diagnostics = self._predict_window(
                model, eval_data, split, window_id, evaluation_config.produce_smoothed
            )
            alignment_diag = None
            summary = self._state_summary(model, train_data)
            if reference_summary is None:
                reference_summary = summary
            else:
                alignment = align_states(
                    reference_summary, summary, evaluation_config.alignment_method
                )
                predictions = self._apply_alignment(predictions, alignment.candidate_to_reference)
                alignment_diag = alignment.diagnostics
                diagnostics["alignment"] = self._jsonable(alignment.diagnostics)
            metrics = self._compute_metrics(
                predictions, eval_data, evaluation_config, transition_matrices, label_runs
            )
            diagnostics["metrics"] = metrics
            requested_metrics = [
                metric
                for metric in (
                    *evaluation_config.statistical_metrics,
                    *evaluation_config.regime_quality_metrics,
                )
                if isinstance(metric, str)
            ]
            diagnostics["unsupported_metrics"] = [
                {"metric": metric, "status": "unsupported", "reason": "required inputs unavailable"}
                for metric in requested_metrics
                if metric not in metrics
                and not any(key.startswith(f"{metric}_") for key in metrics)
            ]
            diagnostics["state_occupancy"] = state_occupancy(predictions["state"]).copy()
            diagnostics["state_durations"] = duration_distribution(predictions["state"])
            diagnostics["runtime_seconds"] = time.perf_counter() - window_started
            diagnostics["max_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            predictions.to_parquet(pred_path, index=False)
            diag_path.write_text(
                json.dumps(self._jsonable(diagnostics), indent=2, sort_keys=True), encoding="utf-8"
            )
            actual_model_path = None
            if model_config.save_models:
                with model_path.open("wb") as file_obj:
                    pickle.dump(model, file_obj)
                actual_model_path = str(model_path)
                recorder.add_artifact(model_path)
            recorder.add_artifact(pred_path)
            recorder.add_artifact(diag_path)
            prediction_frames.append(predictions)
            window_results.append(
                WindowEvaluationResult(
                    window_id,
                    str(pred_path),
                    str(diag_path),
                    actual_model_path,
                    metrics,
                    alignment_diag,
                )
            )
            if evaluation_config.checkpoint_every_window:
                completed.add(window_id)
                checkpoint_path.write_text(
                    json.dumps(
                        {"completed_windows": sorted(completed), "signature": checkpoint_signature}
                    ),
                    encoding="utf-8",
                )

        all_predictions = (
            pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        )
        aggregate_metrics = self._aggregate_metrics(window_results, all_predictions)
        predictions_path = output_dir / "predictions.parquet"
        metrics_path = output_dir / "metrics.json"
        diagnostics_path = output_dir / "diagnostics.json"
        provenance_path = output_dir / "provenance.json"
        contract_path = output_dir / "comparison_contract.json"
        resolved_contract = self._resolved_contract(validation_config, evaluation_config)
        contract_path.write_text(
            json.dumps(self._jsonable(resolved_contract), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        all_predictions.to_parquet(predictions_path, index=False)
        metrics_path.write_text(
            json.dumps(self._jsonable(aggregate_metrics), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        diagnostics_path.write_text(
            json.dumps(
                self._jsonable([asdict(w) for w in window_results]), indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        provenance = recorder.capture(
            config_hash=stable_hash(
                self._config_record(
                    dataset_config, model_config, validation_config, evaluation_config
                )
            ),
            dataset_hash=dataset_config.dataset_id or stable_hash(self._jsonable(data)),
            feature_hash=dataset_config.feature_hash,
            model_hash=stable_hash(self._jsonable(model_config.fit_config)),
            training_period=self._period(splits, "train"),
            validation_period=self._period(splits, validation_config.evaluation_role),
            execution_assumptions={"execution_delay": validation_config.execution_delay},
            cost_assumptions=evaluation_config.cost_assumptions,
        )
        provenance_path.write_text(
            json.dumps(provenance.to_record(), indent=2, sort_keys=True), encoding="utf-8"
        )
        checkpoint_path.write_text(
            json.dumps(
                {
                    "completed_windows": list(range(len(splits))),
                    "complete": True,
                    "signature": checkpoint_signature,
                }
            ),
            encoding="utf-8",
        )
        return EvaluationRunResult(
            evaluation_config.run_id,
            str(output_dir),
            str(predictions_path),
            str(metrics_path),
            str(diagnostics_path),
            str(provenance_path),
            str(checkpoint_path),
            str(contract_path),
            tuple(window_results),
            aggregate_metrics,
        )

    @classmethod
    def assert_comparable(cls, left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
        """Reject model comparisons unless all causal/economic assumptions match."""
        missing = [
            field
            for field in cls._REQUIRED_COMPARISON_FIELDS
            if field not in left or field not in right
        ]
        if missing:
            raise ComparisonContractError(
                f"comparison contract missing fields: {', '.join(missing)}"
            )
        mismatched = [
            field for field in cls._REQUIRED_COMPARISON_FIELDS if left[field] != right[field]
        ]
        if mismatched:
            raise ComparisonContractError(
                f"runs are not comparable; mismatched fields: {', '.join(mismatched)}"
            )

    def _validate_comparison_contract(
        self, validation: ValidationConfig, evaluation: EvaluationConfig
    ) -> None:
        contract = dict(evaluation.comparison_contract)
        if not contract:
            return
        expected = {
            "retraining_schedule": validation.retraining_schedule,
            "execution_delay": validation.execution_delay,
            "cost_assumptions": dict(evaluation.cost_assumptions),
            "downstream_decision_rules": dict(evaluation.downstream_decision_rules),
        }
        for key, value in expected.items():
            if key in contract and contract[key] != value:
                raise ComparisonContractError(
                    f"comparison contract field {key!r} does not match this run"
                )

    def _resolved_contract(
        self, validation: ValidationConfig, evaluation: EvaluationConfig
    ) -> dict[str, Any]:
        contract = dict(evaluation.comparison_contract)
        contract.update(
            {
                "retraining_schedule": validation.retraining_schedule,
                "execution_delay": validation.execution_delay,
                "cost_assumptions": dict(evaluation.cost_assumptions),
                "downstream_decision_rules": dict(evaluation.downstream_decision_rules),
            }
        )
        return contract

    def _load_data(self, config: DatasetConfig) -> Any:
        if config.loader is not None:
            return config.loader()
        if config.data is None:
            raise ValueError("DatasetConfig requires data or loader")
        return config.data

    def _splits(self, data: Any, config: ValidationConfig) -> tuple[ValidationSplit, ...]:
        if config.splits is not None:
            return tuple(config.splits)
        if config.splitter is None:
            raise ValueError("ValidationConfig requires splitter or splits")
        return tuple(config.splitter.split(data))

    def _prepare_split(
        self,
        data: Any,
        split: ValidationSplit,
        dataset: DatasetConfig,
        validation: ValidationConfig,
    ) -> tuple[Any, Any]:
        train = self._take(data, split.train)
        eval_indices = getattr(split, validation.evaluation_role)
        eval_data = self._take(data, eval_indices)
        if dataset.preprocessor_factory is None:
            return train, eval_data
        transformer = dataset.preprocessor_factory()
        y = (
            train[dataset.target_column]
            if dataset.target_column
            and isinstance(train, pd.DataFrame)
            and dataset.target_column in train
            else None
        )
        fitted = transformer.fit(train, y)
        return fitted.transform(train), fitted.transform(eval_data)

    def _predict_window(
        self,
        model: RegimeModel,
        eval_data: Any,
        split: ValidationSplit,
        window_id: int,
        smooth: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        try:
            probs = np.asarray(model.predict_proba(eval_data), dtype=float)
            states = probs.argmax(axis=1).astype(int)
        except (AttributeError, NotImplementedError, UnsupportedModelOperation):
            states = np.asarray(model.predict(eval_data), dtype=int)
            width = max(int(states.max(initial=0)) + 1, 1)
            probs = np.eye(width, dtype=float)[states]
            probability_status = "unsupported"
        else:
            probability_status = "supported"
        records: dict[str, Any] = {
            "window_id": window_id,
            "row": list(range(len(states))),
            "state": states,
        }
        for state in range(probs.shape[1]):
            records[f"filtered_state_{state}"] = probs[:, state]
        diagnostics: dict[str, Any] = {
            "split_metadata": split.metadata,
            "windows": [asdict(window) for window in split.windows],
            "probabilities": {"status": probability_status, "kind": "filtered"},
        }
        if smooth:
            smoothed = self._smoothed_probabilities(model, eval_data)
            if smoothed is not None:
                for state in range(smoothed.shape[1]):
                    records[f"smoothed_state_{state}"] = smoothed[:, state]
        return pd.DataFrame(records), diagnostics

    def _smoothed_probabilities(self, model: RegimeModel, eval_data: Any) -> np.ndarray | None:
        try:
            results = model.smooth(eval_data)
        except Exception:  # optional diagnostics must not change live-equivalent evaluation
            return None
        rows = [
            r.smoothed_probabilities or r.filtered_probabilities
            for r in results
            if isinstance(r, RegimeInferenceResult)
        ]
        return np.asarray(rows, dtype=float) if rows else None

    def _apply_alignment(
        self, predictions: pd.DataFrame, mapping: Mapping[int, int]
    ) -> pd.DataFrame:
        out = predictions.copy()
        out["unaligned_state"] = out["state"]
        out["state"] = out["state"].map(lambda value: mapping.get(int(value), int(value)))
        prob_cols = [col for col in out if col.startswith("filtered_state_")]
        for col in prob_cols:
            old = int(col.rsplit("_", 1)[1])
            new = mapping.get(old, old)
            out[f"aligned_filtered_state_{new}"] = out[col]
        return out

    def _state_summary(self, model: RegimeModel, train_data: Any) -> Mapping[str, Any]:
        try:
            stats = model.state_statistics()
            labels = tuple(int(k) for k in stats)
            return {"state_means": stats, "state_statistics": stats, "labels": labels}
        except Exception:
            labels = np.asarray(model.predict(train_data), dtype=int)
            return {
                "labels": tuple(int(x) for x in np.unique(labels)),
                "durations": {int(k): v for k, v in state_occupancy(labels).items()},
            }

    def _compute_metrics(
        self,
        predictions: pd.DataFrame,
        eval_data: Any,
        config: EvaluationConfig,
        transition_matrices: list[Any],
        label_runs: list[np.ndarray],
    ) -> dict[str, float]:
        labels = predictions["state"].to_numpy(dtype=int)
        label_runs.append(labels)
        probs = predictions[[c for c in predictions if c.startswith("filtered_state_")]].to_numpy(
            dtype=float
        )
        metrics: dict[str, float] = {}
        for metric in config.regime_quality_metrics:
            if callable(metric):
                metrics.update(metric(predictions, {"eval_data": eval_data}))
            elif metric == "regime_persistence":
                metrics[metric] = regime_persistence(labels)
            elif metric == "switching_frequency":
                metrics[metric] = switching_frequency(labels)
            elif metric == "state_entropy":
                metrics[metric] = state_entropy(labels)
            elif metric == "probability_entropy":
                metrics[metric] = probability_entropy(probs)
            elif metric == "rolling_refit_stability":
                metrics[metric] = rolling_refit_stability(label_runs)
            elif metric == "transition_stability":
                metrics[metric] = transition_stability(transition_matrices)
            elif metric == "state_recurrence":
                metrics.update(
                    {f"state_recurrence_{k}": float(v) for k, v in state_recurrence(labels).items()}
                )
        if isinstance(eval_data, pd.DataFrame) and "state" in eval_data:
            y_true = eval_data["state"].to_numpy(dtype=int)
            metrics["brier_score"] = brier_score(y_true, probs)
        if "predictive_density" in predictions:
            metrics["predictive_log_score"] = predictive_log_score(
                np.ones(len(predictions)), predictions["predictive_density"]
            )
        return metrics

    def _aggregate_metrics(
        self, windows: Sequence[WindowEvaluationResult], predictions: pd.DataFrame
    ) -> dict[str, float]:
        out: dict[str, float] = {
            "n_windows": float(len(windows)),
            "n_predictions": float(len(predictions)),
        }
        keys = sorted({key for window in windows for key in window.metrics})
        for key in keys:
            vals = [
                window.metrics[key]
                for window in windows
                if key in window.metrics and np.isfinite(window.metrics[key])
            ]
            if vals:
                out[key] = float(np.mean(vals))
        return out

    def _take(self, data: Any, indices: Sequence[int]) -> Any:
        if isinstance(data, pd.DataFrame | pd.Series):
            return data.iloc[list(indices)].copy()
        arr = np.asarray(data)
        return arr[list(indices)]

    def _fit_config(self, config: RegimeModelConfig | Mapping[str, Any]) -> RegimeModelConfig:
        return (
            config
            if isinstance(config, RegimeModelConfig)
            else RegimeModelConfig.model_validate(config)
        )

    def _load_checkpoint(self, path: Path, signature: str) -> set[int]:
        if not path.exists():
            return set()
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            return set()
        return set(record.get("completed_windows", []))

    def _rehydrate_window(
        self, window_id: int, pred_path: Path, diag_path: Path, model_path: Path
    ) -> WindowEvaluationResult:
        diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
        return WindowEvaluationResult(
            window_id,
            str(pred_path),
            str(diag_path),
            str(model_path) if model_path.exists() else None,
            diagnostics.get("metrics", {}),
            None,
        )

    def _period(self, splits: Sequence[ValidationSplit], role: str) -> TimePeriod | None:
        if not splits:
            return None
        indices = [idx for split in splits for idx in getattr(split, role)]
        if not indices:
            return None
        return TimePeriod(str(min(indices)), str(max(indices) + 1))

    def _config_record(self, *configs: Any) -> dict[str, Any]:
        return {type(config).__name__: self._jsonable(config) for config in configs}

    def _jsonable(self, value: Any) -> Json:
        if is_dataclass(value):
            return self._jsonable(asdict(value))
        if isinstance(value, Mapping):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, list | tuple | set):
            return [self._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return self._jsonable(value.tolist())
        if isinstance(value, np.integer | np.floating):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return repr(value)


__all__ = [
    "ComparisonContractError",
    "DatasetConfig",
    "EvaluationConfig",
    "EvaluationRunResult",
    "EvaluationRunner",
    "ModelConfig",
    "ValidationConfig",
    "WindowEvaluationResult",
]
