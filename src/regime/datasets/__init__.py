"""Dataset assembly primitives for regime-switching workflows.

The module provides leakage-aware helpers for turning point-in-time features,
targets, and split metadata into immutable :class:`RegimeDataset` objects.  The
builder fits preprocessing transforms only on training rows and defaults to
live-equivalent probability handling: smoothed probabilities are rejected unless
explicitly enabled for offline diagnostics.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from regime.errors import RegimeDataError
from regime.experiments.provenance import stable_hash


class MissingValuePolicy(StrEnum):
    """Missing-value policies applied during dataset assembly."""

    ERROR = "error"
    DROP = "drop"
    FORWARD_FILL = "forward_fill"
    FILL_ZERO = "fill_zero"
    LEAVE = "leave"


class ProbabilityMode(StrEnum):
    """Controls whether smoothed probabilities are allowed downstream."""

    LIVE_EQUIVALENT = "live_equivalent"
    OFFLINE_DIAGNOSTICS = "offline_diagnostics"


class SplitName(StrEnum):
    """Canonical dataset split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Inclusive time window used for train, validation, or test rows."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("time window end must be greater than or equal to start")

    def contains(self, values: pd.Series) -> pd.Series:
        """Return a boolean mask for values inside this window."""
        timestamps = pd.to_datetime(values)
        return (timestamps >= self.start) & (timestamps <= self.end)

    def to_record(self) -> dict[str, str]:
        """Return JSON-compatible metadata."""
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True, slots=True)
class SplitMetadata:
    """Train/validation/test period metadata."""

    train: TimeWindow
    validation: TimeWindow | None = None
    test: TimeWindow | None = None

    def window_for(self, split: SplitName) -> TimeWindow | None:
        """Return the configured window for a split."""
        if split is SplitName.TRAIN:
            return self.train
        if split is SplitName.VALIDATION:
            return self.validation
        return self.test

    def to_record(self) -> dict[str, dict[str, str] | None]:
        """Return JSON-compatible split metadata."""
        return {
            "train": self.train.to_record(),
            "validation": self.validation.to_record() if self.validation else None,
            "test": self.test.to_record() if self.test else None,
        }


@dataclass(frozen=True, slots=True)
class ExecutionDelay:
    """Feature/decision/target timing assumptions for live-equivalent evaluation."""

    feature_lag: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    decision_delay: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    target_horizon: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))

    def available_at(self, timestamp: pd.Series) -> pd.Series:
        """Return when feature rows are available for decisions."""
        return pd.to_datetime(timestamp) + self.feature_lag + self.decision_delay

    def label_at(self, timestamp: pd.Series) -> pd.Series:
        """Return when labels can be observed."""
        return pd.to_datetime(timestamp) + self.target_horizon

    def to_record(self) -> dict[str, float]:
        """Return delays in seconds."""
        return {
            "feature_lag_seconds": self.feature_lag.total_seconds(),
            "decision_delay_seconds": self.decision_delay.total_seconds(),
            "target_horizon_seconds": self.target_horizon.total_seconds(),
        }


@dataclass(frozen=True, slots=True)
class EmbargoMetadata:
    """Embargo settings used to avoid leakage between split boundaries."""

    duration: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    purged_rows: int = 0

    def to_record(self) -> dict[str, float | int]:
        """Return JSON-compatible embargo metadata."""
        return {"duration_seconds": self.duration.total_seconds(), "purged_rows": self.purged_rows}


@dataclass(frozen=True, slots=True)
class AssetUniverseSnapshot:
    """Point-in-time asset universe membership for a dataset."""

    as_of: pd.Timestamp
    assets: tuple[str, ...]
    source: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return deterministic universe metadata."""
        return {"as_of": self.as_of.isoformat(), "assets": self.assets, "source": self.source}


@dataclass(frozen=True, slots=True)
class LazyFrame:
    """Lazy pandas frame wrapper for large on-disk datasets."""

    path: Path
    loader: Callable[[Path], pd.DataFrame] | None = None
    _frame: pd.DataFrame | None = field(default=None, init=False, compare=False, repr=False)

    def load(self) -> pd.DataFrame:
        """Load and cache the frame on first access."""
        cached = object.__getattribute__(self, "_frame")
        if cached is not None:
            return cached.copy()
        loader = self.loader or _default_loader
        frame = loader(self.path)
        object.__setattr__(self, "_frame", frame.copy())
        return frame


def _default_loader(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as file_obj:
            loaded = pickle.load(file_obj)
        if not isinstance(loaded, pd.DataFrame):
            raise RegimeDataError("lazy pickle did not contain a pandas DataFrame")
        return loaded
    raise RegimeDataError(f"unsupported lazy dataset format: {path.suffix}")


FrameLike: TypeAlias = pd.DataFrame | LazyFrame


@dataclass(frozen=True, slots=True)
class RegimeDataset:
    """Assembled dataset plus reproducibility and leakage-control metadata."""

    features: FrameLike
    targets: FrameLike
    labels: FrameLike | None
    split_metadata: SplitMetadata
    asset_universe: AssetUniverseSnapshot
    execution_delay: ExecutionDelay
    embargo: EmbargoMetadata
    missing_value_policy: MissingValuePolicy
    dataset_hash: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    split_column: str = "split"
    timestamp_column: str = "timestamp"
    asset_column: str | None = "asset"
    probability_mode: ProbabilityMode = ProbabilityMode.LIVE_EQUIVALENT
    allow_smoothed_probabilities: bool = False
    preprocessing_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.probability_mode is ProbabilityMode.LIVE_EQUIVALENT
            and self.allow_smoothed_probabilities
        ):
            raise RegimeDataError(
                "smoothed probabilities are disabled for live-equivalent datasets"
            )

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802
        """Return the feature matrix, loading lazily if required."""
        return (
            self.features.load() if isinstance(self.features, LazyFrame) else self.features.copy()
        )

    @property
    def y(self) -> pd.DataFrame:
        """Return target values, loading lazily if required."""
        return self.targets.load() if isinstance(self.targets, LazyFrame) else self.targets.copy()

    def labels_frame(self) -> pd.DataFrame | None:
        """Return optional labels, loading lazily if required."""
        if self.labels is None:
            return None
        return self.labels.load() if isinstance(self.labels, LazyFrame) else self.labels.copy()

    def split(self, name: SplitName | str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return feature and target matrices for one split."""
        split_name = SplitName(name)
        features = self.X
        targets = self.y
        if self.split_column not in features:
            raise RegimeDataError(f"features are missing split column {self.split_column!r}")
        mask = features[self.split_column] == split_name.value
        return features.loc[mask, list(self.feature_columns)].copy(), targets.loc[mask].copy()

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible dataset metadata."""
        return {
            "dataset_hash": self.dataset_hash,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "split_metadata": self.split_metadata.to_record(),
            "asset_universe": self.asset_universe.to_record(),
            "execution_delay": self.execution_delay.to_record(),
            "embargo": self.embargo.to_record(),
            "missing_value_policy": self.missing_value_policy.value,
            "probability_mode": self.probability_mode.value,
            "allow_smoothed_probabilities": self.allow_smoothed_probabilities,
            "preprocessing": dict(self.preprocessing_metadata),
        }


@dataclass(slots=True)
class DatasetBuilder:
    """Build :class:`RegimeDataset` instances with split-aware preprocessing."""

    timestamp_column: str = "timestamp"
    asset_column: str | None = "asset"
    split_column: str = "split"
    missing_value_policy: MissingValuePolicy = MissingValuePolicy.ERROR
    execution_delay: ExecutionDelay = field(default_factory=ExecutionDelay)
    embargo_duration: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    probability_mode: ProbabilityMode = ProbabilityMode.LIVE_EQUIVALENT
    allow_smoothed_probabilities: bool = False
    scaler: str | BaseEstimator | None = None
    pca_components: int | float | None = None
    categorical_columns: tuple[str, ...] = ()
    feature_selector: BaseEstimator | None = None

    def build(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame | pd.Series,
        *,
        splits: SplitMetadata,
        asset_universe: AssetUniverseSnapshot | Sequence[str],
        labels: pd.DataFrame | pd.Series | None = None,
        feature_columns: Sequence[str] | None = None,
        target_columns: Sequence[str] | None = None,
        smoothed_probability_columns: Sequence[str] = (),
        lazy_output_dir: str | Path | None = None,
    ) -> RegimeDataset:
        """Assemble feature matrix, targets, labels, metadata, and hash."""
        if self.probability_mode is ProbabilityMode.LIVE_EQUIVALENT and (
            self.allow_smoothed_probabilities or smoothed_probability_columns
        ):
            raise RegimeDataError(
                "smoothed probabilities require probability_mode='offline_diagnostics' and "
                "allow_smoothed_probabilities=True"
            )
        feature_frame = features.copy()
        target_frame = targets.to_frame() if isinstance(targets, pd.Series) else targets.copy()
        label_frame = (
            labels.to_frame()
            if isinstance(labels, pd.Series)
            else labels.copy()
            if labels is not None
            else None
        )
        feature_frame = self._stamp_splits(feature_frame, splits)
        feature_frame = self._apply_execution_delay(feature_frame)
        feature_frame, target_frame, label_frame, purged = self._apply_embargo(
            feature_frame, target_frame, label_frame, splits
        )
        feature_frame = self._apply_missing_values(feature_frame)
        target_frame = self._apply_missing_values(target_frame)
        common_index = feature_frame.index.intersection(target_frame.index)
        feature_frame = feature_frame.loc[common_index].copy()
        target_frame = target_frame.loc[common_index].copy()
        if label_frame is not None:
            label_frame = label_frame.loc[common_index.intersection(label_frame.index)].copy()
            feature_frame = feature_frame.loc[label_frame.index].copy()
            target_frame = target_frame.loc[label_frame.index].copy()
        selected_features = tuple(feature_columns or self._infer_feature_columns(feature_frame))
        selected_targets = tuple(target_columns or target_frame.columns.astype(str))
        feature_frame, preprocessing = self._preprocess(feature_frame, selected_features)
        universe = self._coerce_universe(asset_universe, feature_frame)
        embargo = EmbargoMetadata(self.embargo_duration, purged)
        dataset_hash = self._dataset_hash(
            feature_frame, target_frame, label_frame, splits, universe, embargo, preprocessing
        )
        feature_like, target_like, label_like = self._maybe_lazy(
            feature_frame, target_frame, label_frame, lazy_output_dir
        )
        return RegimeDataset(
            features=feature_like,
            targets=target_like,
            labels=label_like,
            split_metadata=splits,
            asset_universe=universe,
            execution_delay=self.execution_delay,
            embargo=embargo,
            missing_value_policy=self.missing_value_policy,
            dataset_hash=dataset_hash,
            feature_columns=tuple(c for c in feature_frame.columns if c.startswith("feature__"))
            or selected_features,
            target_columns=selected_targets,
            split_column=self.split_column,
            timestamp_column=self.timestamp_column,
            asset_column=self.asset_column,
            probability_mode=self.probability_mode,
            allow_smoothed_probabilities=self.allow_smoothed_probabilities,
            preprocessing_metadata=preprocessing,
        )

    def _stamp_splits(self, frame: pd.DataFrame, splits: SplitMetadata) -> pd.DataFrame:
        if self.timestamp_column not in frame:
            raise RegimeDataError(
                f"features are missing timestamp column {self.timestamp_column!r}"
            )
        stamped = frame.copy()
        stamped[self.split_column] = pd.NA
        for split in SplitName:
            window = splits.window_for(split)
            if window is not None:
                stamped.loc[window.contains(stamped[self.timestamp_column]), self.split_column] = (
                    split.value
                )
        return stamped.dropna(subset=[self.split_column]).copy()

    def _apply_execution_delay(self, frame: pd.DataFrame) -> pd.DataFrame:
        delayed = frame.copy()
        delayed["available_at"] = self.execution_delay.available_at(delayed[self.timestamp_column])
        delayed["label_available_at"] = self.execution_delay.label_at(
            delayed[self.timestamp_column]
        )
        return delayed

    def _apply_embargo(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        labels: pd.DataFrame | None,
        splits: SplitMetadata,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, int]:
        if self.embargo_duration <= pd.Timedelta(0):
            return (
                features,
                targets.loc[features.index].copy(),
                labels.loc[features.index].copy() if labels is not None else None,
                0,
            )
        keep = pd.Series(True, index=features.index)
        ts = pd.to_datetime(features[self.timestamp_column])
        for window in (splits.train, splits.validation):
            if window is None:
                continue
            boundary = window.end
            keep &= ~((ts > boundary) & (ts <= boundary + self.embargo_duration))
        purged = int((~keep).sum())
        kept_features = features.loc[keep].copy()
        return (
            kept_features,
            targets.loc[kept_features.index].copy(),
            labels.loc[kept_features.index].copy() if labels is not None else None,
            purged,
        )

    def _apply_missing_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.missing_value_policy is MissingValuePolicy.ERROR and frame.isna().any().any():
            raise RegimeDataError("dataset contains missing values")
        if self.missing_value_policy is MissingValuePolicy.DROP:
            return frame.dropna().copy()
        if self.missing_value_policy is MissingValuePolicy.FORWARD_FILL:
            return frame.ffill()
        if self.missing_value_policy is MissingValuePolicy.FILL_ZERO:
            return frame.fillna(0)
        return frame

    def _infer_feature_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        excluded = {self.timestamp_column, self.split_column, "available_at", "label_available_at"}
        if self.asset_column is not None:
            excluded.add(self.asset_column)
        return tuple(column for column in frame.columns.astype(str) if column not in excluded)

    def _preprocess(
        self, frame: pd.DataFrame, feature_columns: tuple[str, ...]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        train_mask = frame[self.split_column] == SplitName.TRAIN.value
        if not train_mask.any():
            raise RegimeDataError("training split contains no rows for preprocessing fit")
        numeric_columns = [c for c in feature_columns if c not in self.categorical_columns]
        transformed = frame.copy()
        output_parts: list[pd.DataFrame] = []
        steps: list[str] = []
        if numeric_columns:
            numeric = transformed[numeric_columns]
            estimator = self._make_numeric_pipeline()
            if estimator is not None:
                estimator.fit(numeric.loc[train_mask])
                values = estimator.transform(numeric)
                names = [f"feature__{i}" for i in range(values.shape[1])]
                output_parts.append(pd.DataFrame(values, index=frame.index, columns=names))
                steps.append(repr(estimator))
            else:
                output_parts.append(numeric.add_prefix("feature__"))
        if self.categorical_columns:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.fit(frame.loc[train_mask, list(self.categorical_columns)])
            values = encoder.transform(frame[list(self.categorical_columns)])
            names = [
                f"feature__{name}"
                for name in encoder.get_feature_names_out(self.categorical_columns)
            ]
            output_parts.append(pd.DataFrame(values, index=frame.index, columns=names))
            steps.append("OneHotEncoder(handle_unknown='ignore')")
        meta = frame[
            [self.timestamp_column, self.split_column, "available_at", "label_available_at"]
        ].copy()
        if self.asset_column and self.asset_column in frame:
            meta[self.asset_column] = frame[self.asset_column]
        return pd.concat([meta, *output_parts], axis=1), {"fit_split": "train", "steps": steps}

    def _make_numeric_pipeline(self) -> Pipeline | None:
        steps: list[tuple[str, BaseEstimator]] = []
        if isinstance(self.scaler, str):
            if self.scaler == "standard":
                steps.append(("scaler", StandardScaler()))
            elif self.scaler == "minmax":
                steps.append(("scaler", MinMaxScaler()))
            elif self.scaler not in {"none", ""}:
                raise RegimeDataError(f"unknown scaler {self.scaler!r}")
        elif self.scaler is not None:
            steps.append(("scaler", clone(self.scaler)))
        if self.pca_components is not None:
            steps.append(("pca", PCA(n_components=self.pca_components)))
        if self.feature_selector is not None:
            steps.append(("feature_selector", clone(self.feature_selector)))
        return Pipeline(steps) if steps else None

    def _coerce_universe(
        self, universe: AssetUniverseSnapshot | Sequence[str], frame: pd.DataFrame
    ) -> AssetUniverseSnapshot:
        if isinstance(universe, AssetUniverseSnapshot):
            return universe
        as_of = pd.to_datetime(frame[self.timestamp_column]).max()
        return AssetUniverseSnapshot(as_of=as_of, assets=tuple(sorted(map(str, universe))))

    def _dataset_hash(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        labels: pd.DataFrame | None,
        splits: SplitMetadata,
        universe: AssetUniverseSnapshot,
        embargo: EmbargoMetadata,
        preprocessing: Mapping[str, Any],
    ) -> str:
        return stable_hash(
            {
                "features": pd.util.hash_pandas_object(features, index=True)
                .to_numpy()
                .tobytes()
                .hex(),
                "targets": pd.util.hash_pandas_object(targets, index=True)
                .to_numpy()
                .tobytes()
                .hex(),
                "labels": pd.util.hash_pandas_object(labels, index=True).to_numpy().tobytes().hex()
                if labels is not None
                else None,
                "splits": splits.to_record(),
                "universe": universe.to_record(),
                "execution_delay": self.execution_delay.to_record(),
                "embargo": embargo.to_record(),
                "preprocessing": preprocessing,
                "probability_mode": self.probability_mode.value,
            }
        )

    def _maybe_lazy(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        labels: pd.DataFrame | None,
        directory: str | Path | None,
    ) -> tuple[FrameLike, FrameLike, FrameLike | None]:
        if directory is None:
            return features, targets, labels
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        feature_path = out / "features.parquet"
        target_path = out / "targets.parquet"
        features.to_parquet(feature_path)
        targets.to_parquet(target_path)
        label_path = out / "labels.parquet"
        if labels is not None:
            labels.to_parquet(label_path)
        return (
            LazyFrame(feature_path),
            LazyFrame(target_path),
            LazyFrame(label_path) if labels is not None else None,
        )


__all__ = [
    "AssetUniverseSnapshot",
    "DatasetBuilder",
    "EmbargoMetadata",
    "ExecutionDelay",
    "LazyFrame",
    "MissingValuePolicy",
    "ProbabilityMode",
    "RegimeDataset",
    "SplitMetadata",
    "SplitName",
    "TimeWindow",
]
