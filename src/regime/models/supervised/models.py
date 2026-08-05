"""Supervised regime classifiers and transition-hazard models.

The classes in this module consume regime labels from economic rules, forward
outcomes, unsupervised pseudo-labels, or synthetic known states.  Pseudo-label
use is deliberately represented in configuration, reports, and model cards so
that downstream users do not confuse replicated unsupervised assignments with
independently observed regimes.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from regime.models.base import ModelMetadata, RegimeModel, RegimeModelConfig

ArrayLike = pd.DataFrame | pd.Series | Sequence[Sequence[float]] | Sequence[float] | np.ndarray
LabelSource = Literal[
    "economic_definition",
    "future_outcome_definition",
    "unsupervised_pseudo_label",
    "synthetic_known_state",
]
ClassifierKind = Literal["logistic_regression", "random_forest", "gradient_boosted_trees"]
CalibrationMethod = Literal["sigmoid", "isotonic"]


class SupervisedRegimeConfig(RegimeModelConfig):
    """Configuration for supervised regime classifiers.

    Set ``label_source='unsupervised_pseudo_label'`` only when training labels are
    inherited from an unsupervised model.  In that case, ``pseudo_label_replication_risk``
    must describe the risk that this supervised model merely reproduces the source
    model's errors, cluster instability, or arbitrary label mapping.
    """

    model_name: str = "supervised_regime_classifier"
    classifier: ClassifierKind = "logistic_regression"
    label_source: LabelSource = "economic_definition"
    label_definition: str = Field(default="unspecified", min_length=1)
    pseudo_label_source_model: str | None = None
    pseudo_label_replication_risk: str | None = None
    require_pseudo_label_risk_acknowledgement: bool = True
    calibrate: bool = True
    calibration_method: CalibrationMethod = "sigmoid"
    cv: int = Field(default=3, ge=2)
    scale: bool = True
    feature_names: tuple[str, ...] | None = None
    class_weight: str | None = None
    max_iter: int = Field(default=500, ge=1)
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int | None = Field(default=3, ge=1)
    learning_rate: float = Field(default=0.1, gt=0)

    @model_validator(mode="after")
    def _validate_pseudo_label_risk(self) -> Self:
        is_pseudo = self.label_source == "unsupervised_pseudo_label"
        if is_pseudo and self.require_pseudo_label_risk_acknowledgement:
            if not self.pseudo_label_source_model or not self.pseudo_label_replication_risk:
                raise ValueError(
                    "pseudo-label training requires pseudo_label_source_model and "
                    "pseudo_label_replication_risk to make replication risk explicit"
                )
        if not is_pseudo and self.pseudo_label_replication_risk:
            raise ValueError("pseudo_label_replication_risk is only valid for pseudo-labels")
        return self


class TransitionHazardConfig(SupervisedRegimeConfig):
    """Configuration for transition/event probability models."""

    model_name: str = "supervised_transition_hazard"
    event_definition: str = "next label differs from current label"
    event_horizon: int = Field(default=1, ge=1)


@dataclass(frozen=True)
class SupervisedPredictionResult:
    """Prediction payload emitted by supervised classifiers."""

    predicted_class: int
    calibrated_probabilities: tuple[float, ...]
    transition_or_event_probability: float | None


@dataclass(frozen=True)
class SupervisedReport:
    """Portable report/model-card fields for supervised regime models."""

    label_source: LabelSource
    label_definition: str
    pseudo_label_replication_risk: str | None
    feature_importance: Mapping[str, float]
    calibration_diagnostics: Mapping[str, Any]
    model_card: Mapping[str, Any]


def _as_frame(data: ArrayLike, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "value")
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = list(feature_names or [f"feature_{i}" for i in range(arr.shape[1])])
        frame = pd.DataFrame(arr, columns=names)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("data must contain at least one numeric feature")
    return numeric.ffill().bfill().fillna(0.0)


def _labels(labels: Sequence[int] | pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=int)
    if arr.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(arr) == 0:
        raise ValueError("labels must not be empty")
    return arr


def transition_events(
    labels: Sequence[int] | pd.Series | np.ndarray, horizon: int = 1
) -> np.ndarray:
    """Return binary events indicating whether the label changes within ``horizon``."""
    arr = _labels(labels)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    events = np.zeros(len(arr), dtype=int)
    for i in range(max(0, len(arr) - horizon)):
        events[i] = int(np.any(arr[i + 1 : i + horizon + 1] != arr[i]))
    return events


class SupervisedRegimeClassifier(RegimeModel):
    """Logistic-regression, random-forest, or gradient-boosted regime classifier."""

    def __init__(self, config: SupervisedRegimeConfig | None = None) -> None:
        self.config = config or SupervisedRegimeConfig()
        self.scaler: StandardScaler | None = None
        self.estimator: Any | None = None
        self.classes_: tuple[int, ...] = ()
        self.feature_names: tuple[str, ...] = ()
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=self.config.n_states,
            config_hash=self.configuration_hash(self.config),
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def _base_estimator(self) -> Any:
        if self.config.classifier == "logistic_regression":
            return LogisticRegression(
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_seed,
            )
        if self.config.classifier == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                class_weight=self.config.class_weight,
                random_state=self.config.random_seed,
            )
        return GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_seed,
        )

    def _prepare_fit(self, data: ArrayLike) -> tuple[pd.DataFrame, np.ndarray]:
        frame = _as_frame(data, self.config.feature_names)
        self.feature_names = tuple(str(c) for c in frame.columns)
        values = frame.to_numpy(dtype=float)
        if self.config.scale and self.config.classifier == "logistic_regression":
            self.scaler = StandardScaler().fit(values)
            values = self.scaler.transform(values)
        return frame, values

    def _prepare_predict(self, data: ArrayLike) -> np.ndarray:
        values = _as_frame(data, self.feature_names or self.config.feature_names).to_numpy(
            dtype=float
        )
        return self.scaler.transform(values) if self.scaler is not None else values

    def fit(
        self,
        dataset: ArrayLike,
        config: RegimeModelConfig | None = None,
        labels: Sequence[int] | None = None,
    ) -> Self:  # type: ignore[override]
        if config is not None:
            self.config = SupervisedRegimeConfig.model_validate(config.model_dump())
        if labels is None:
            raise ValueError("supervised classifiers require labels")
        frame, x = self._prepare_fit(dataset)
        y = _labels(labels)
        if len(frame) != len(y):
            raise ValueError("dataset and labels must have the same length")
        base = self._base_estimator()
        if self.config.calibrate:
            base = CalibratedClassifierCV(
                base, method=self.config.calibration_method, cv=self.config.cv
            )
        self.estimator = base.fit(x, y)
        self.classes_ = tuple(int(c) for c in self.estimator.classes_)
        self._metadata = ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            n_states=max(self.config.n_states, len(self.classes_)),
            fitted_at=datetime.now(UTC),
            training_observations=len(frame),
            feature_names=self.feature_names,
            config_hash=self.configuration_hash(self.config),
            attributes={"label_source": self.config.label_source},
        )
        return self

    def predict(self, dataset: ArrayLike) -> tuple[int, ...]:
        if self.estimator is None:
            raise ValueError("model is not fitted")
        return tuple(int(x) for x in self.estimator.predict(self._prepare_predict(dataset)))

    def predict_proba(self, dataset: ArrayLike) -> tuple[tuple[float, ...], ...]:
        if self.estimator is None:
            raise ValueError("model is not fitted")
        probs = self.estimator.predict_proba(self._prepare_predict(dataset))
        return tuple(tuple(float(x) for x in row) for row in probs)

    def predict_full(self, dataset: ArrayLike) -> tuple[SupervisedPredictionResult, ...]:
        preds = self.predict(dataset)
        probs = self.predict_proba(dataset)
        return tuple(
            SupervisedPredictionResult(p, pr, None) for p, pr in zip(preds, probs, strict=True)
        )

    def feature_importance(self) -> dict[str, float]:
        if self.estimator is None:
            raise ValueError("model is not fitted")
        estimator = getattr(self.estimator, "estimator", self.estimator)
        if hasattr(estimator, "feature_importances_"):
            vals = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            vals = np.mean(np.abs(estimator.coef_), axis=0)
        elif hasattr(self.estimator, "calibrated_classifiers_"):
            inner = self.estimator.calibrated_classifiers_[0].estimator
            vals = getattr(
                inner,
                "feature_importances_",
                np.mean(
                    np.abs(getattr(inner, "coef_", np.zeros((1, len(self.feature_names))))), axis=0
                ),
            )
        else:
            return {}
        total = float(np.sum(np.abs(vals))) or 1.0
        return {
            name: float(abs(v) / total) for name, v in zip(self.feature_names, vals, strict=True)
        }

    def calibration_diagnostics(
        self, dataset: ArrayLike, labels: Sequence[int], bins: int = 10
    ) -> dict[str, Any]:
        y = _labels(labels)
        probs = np.asarray(self.predict_proba(dataset), dtype=float)
        diagnostics: dict[str, Any] = {"brier_by_class": {}, "curve_by_class": {}}
        diagnostics["log_loss"] = float(log_loss(y, probs, labels=list(self.classes_)))
        for i, cls in enumerate(self.classes_):
            binary = (y == cls).astype(int)
            diagnostics["brier_by_class"][str(cls)] = float(brier_score_loss(binary, probs[:, i]))
            frac, mean = calibration_curve(binary, probs[:, i], n_bins=bins, strategy="uniform")
            diagnostics["curve_by_class"][str(cls)] = {
                "fraction_of_positives": frac.tolist(),
                "mean_predicted_probability": mean.tolist(),
            }
        return diagnostics

    def report(
        self, dataset: ArrayLike | None = None, labels: Sequence[int] | None = None
    ) -> SupervisedReport:
        diagnostics = (
            {}
            if dataset is None or labels is None
            else self.calibration_diagnostics(dataset, labels)
        )
        card = {
            "model_name": self.config.model_name,
            "classifier": self.config.classifier,
            "label_source": self.config.label_source,
            "label_definition": self.config.label_definition,
            "pseudo_label_source_model": self.config.pseudo_label_source_model,
            "pseudo_label_replication_risk": self.config.pseudo_label_replication_risk,
            "intended_outputs": [
                "predicted_class",
                "calibrated_probabilities",
                "feature_importance",
                "calibration_diagnostics",
            ],
        }
        return SupervisedReport(
            self.config.label_source,
            self.config.label_definition,
            self.config.pseudo_label_replication_risk,
            self.feature_importance() if self.estimator is not None else {},
            diagnostics,
            card,
        )

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"saved object is not a {cls.__name__}")
        return obj


class TransitionHazardModel(SupervisedRegimeClassifier):
    """Binary supervised hazard model for regime transition/event probabilities."""

    config: TransitionHazardConfig

    def __init__(self, config: TransitionHazardConfig | None = None) -> None:
        super().__init__(
            config or TransitionHazardConfig(classifier="logistic_regression", n_states=2)
        )

    def fit(
        self,
        dataset: ArrayLike,
        config: RegimeModelConfig | None = None,
        labels: Sequence[int] | None = None,
    ) -> Self:  # type: ignore[override]
        if config is not None:
            self.config = TransitionHazardConfig.model_validate(config.model_dump())
        if labels is None:
            raise ValueError("transition hazard models require state labels")
        events = transition_events(labels, self.config.event_horizon)
        return super().fit(dataset, None, events)

    def predict_event_proba(self, dataset: ArrayLike) -> tuple[float, ...]:
        probs = np.asarray(self.predict_proba(dataset), dtype=float)
        if 1 in self.classes_:
            idx = self.classes_.index(1)
            return tuple(float(x) for x in probs[:, idx])
        return tuple(0.0 for _ in range(len(probs)))

    def predict_full(self, dataset: ArrayLike) -> tuple[SupervisedPredictionResult, ...]:
        preds = self.predict(dataset)
        probs = self.predict_proba(dataset)
        events = self.predict_event_proba(dataset)
        return tuple(
            SupervisedPredictionResult(p, pr, e)
            for p, pr, e in zip(preds, probs, events, strict=True)
        )
