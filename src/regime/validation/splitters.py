"""Leakage-aware validation splitters for financial time-series workflows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence, Sized
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

WindowRole = Literal["train", "validation", "test", "embargo", "execution"]


@dataclass(frozen=True)
class TimeWindow:
    """Explicit inclusive/exclusive positional window emitted by every splitter."""

    role: WindowRole
    start: int
    stop: int
    dates: tuple[pd.Timestamp, ...] = ()

    @property
    def indices(self) -> tuple[int, ...]:
        """Return integer sample positions contained in the window."""
        return tuple(range(self.start, self.stop))


@dataclass(frozen=True)
class ValidationSplit:
    """A single train/validation/test split with leakage-control windows."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    embargo: tuple[int, ...]
    execution: tuple[int, ...]
    windows: tuple[TimeWindow, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def nested(self, splitter: BaseSplitter) -> tuple[ValidationSplit, ...]:
        """Run nested model-selection splits inside this split's training sample."""
        nested_splits = []
        for split in splitter.split(range(len(self.train))):
            nested_splits.append(
                ValidationSplit(
                    train=tuple(self.train[index] for index in split.train),
                    validation=tuple(self.train[index] for index in split.validation),
                    test=tuple(self.train[index] for index in split.test),
                    embargo=tuple(self.train[index] for index in split.embargo),
                    execution=tuple(self.train[index] for index in split.execution),
                    windows=split.windows,
                    metadata={**split.metadata, "outer_train_size": len(self.train)},
                )
            )
        return tuple(nested_splits)


@dataclass(frozen=True)
class BaseSplitter:
    """Base class shared by all splitters."""

    embargo: int = 0
    execution_delay: int = 0
    label_start_col: str = "label_start"
    label_end_col: str = "label_end"
    purge_overlapping_labels: bool = False

    def split(self, data: object) -> Iterator[ValidationSplit]:
        raise NotImplementedError

    def _frame(self, data: object) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.reset_index(drop=True).copy()
        elif isinstance(data, Sized):
            frame = pd.DataFrame(index=range(len(data)))
        else:
            rows = tuple(data)  # type: ignore[arg-type]
            frame = pd.DataFrame(index=range(len(rows)))
        if "date" not in frame:
            frame["date"] = pd.RangeIndex(len(frame))
        return frame

    def _make_split(
        self,
        frame: pd.DataFrame,
        train: Iterable[int],
        validation: Iterable[int],
        test: Iterable[int],
        *,
        metadata: dict[str, object] | None = None,
    ) -> ValidationSplit:
        train_set = set(train)
        validation_tuple = tuple(validation)
        test_tuple = tuple(test)
        eval_tuple = validation_tuple + test_tuple
        blocked = set(self._embargo_indices(eval_tuple, len(frame)))
        if self.purge_overlapping_labels:
            blocked.update(self._label_overlap_indices(frame, eval_tuple))
        train_tuple = tuple(index for index in sorted(train_set) if index not in blocked)
        embargo_tuple = tuple(sorted(blocked - set(validation_tuple) - set(test_tuple)))
        execution_tuple = tuple(self._execution_indices(eval_tuple, len(frame)))
        windows = self._windows(
            frame, train_tuple, validation_tuple, test_tuple, embargo_tuple, execution_tuple
        )
        return ValidationSplit(
            train=train_tuple,
            validation=validation_tuple,
            test=test_tuple,
            embargo=embargo_tuple,
            execution=execution_tuple,
            windows=windows,
            metadata=metadata or {},
        )

    def _embargo_indices(self, eval_indices: Sequence[int], size: int) -> tuple[int, ...]:
        if self.embargo <= 0 or not eval_indices:
            return ()
        blocked: set[int] = set()
        for index in eval_indices:
            blocked.update(range(max(0, index - self.embargo), min(size, index + self.embargo + 1)))
        return tuple(sorted(blocked))

    def _execution_indices(self, eval_indices: Sequence[int], size: int) -> tuple[int, ...]:
        if self.execution_delay <= 0:
            return tuple(eval_indices)
        return tuple(
            index for index in (i + self.execution_delay for i in eval_indices) if index < size
        )

    def _label_overlap_indices(
        self, frame: pd.DataFrame, eval_indices: Sequence[int]
    ) -> tuple[int, ...]:
        if self.label_start_col not in frame or self.label_end_col not in frame or not eval_indices:
            return ()
        eval_spans = frame.loc[list(eval_indices), [self.label_start_col, self.label_end_col]]
        blocked: set[int] = set(eval_indices)
        for index, row in frame[[self.label_start_col, self.label_end_col]].iterrows():
            start = row[self.label_start_col]
            end = row[self.label_end_col]
            overlaps = (
                (start < eval_spans[self.label_end_col]) & (end > eval_spans[self.label_start_col])
            ).any()
            if bool(overlaps):
                blocked.add(int(index))
        return tuple(sorted(blocked))

    def _windows(
        self,
        frame: pd.DataFrame,
        *groups: tuple[int, ...],
    ) -> tuple[TimeWindow, ...]:
        roles: tuple[WindowRole, ...] = ("train", "validation", "test", "embargo", "execution")
        windows = []
        for role, group in zip(roles, groups, strict=True):
            if group:
                dates = tuple(
                    pd.Timestamp(value)
                    for value in frame.loc[list(group), "date"]
                    if not isinstance(value, int)
                )
                windows.append(TimeWindow(role, min(group), max(group) + 1, dates))
            else:
                windows.append(TimeWindow(role, 0, 0, ()))
        return tuple(windows)


@dataclass(frozen=True)
class ExpandingWindowSplitter(BaseSplitter):
    initial_train_size: int = 20
    validation_size: int = 5
    test_size: int = 5
    step: int = 5

    def split(self, data: object) -> Iterator[ValidationSplit]:
        frame = self._frame(data)
        size = len(frame)
        train_stop = self.initial_train_size
        while train_stop + self.validation_size + self.test_size <= size:
            val_stop = train_stop + self.validation_size
            test_stop = val_stop + self.test_size
            yield self._make_split(
                frame, range(train_stop), range(train_stop, val_stop), range(val_stop, test_stop)
            )
            train_stop += self.step


@dataclass(frozen=True)
class RollingWindowSplitter(BaseSplitter):
    train_size: int = 20
    validation_size: int = 5
    test_size: int = 5
    step: int = 5

    def split(self, data: object) -> Iterator[ValidationSplit]:
        frame = self._frame(data)
        start = 0
        while start + self.train_size + self.validation_size + self.test_size <= len(frame):
            train_stop = start + self.train_size
            val_stop = train_stop + self.validation_size
            test_stop = val_stop + self.test_size
            yield self._make_split(
                frame,
                range(start, train_stop),
                range(train_stop, val_stop),
                range(val_stop, test_stop),
            )
            start += self.step


@dataclass(frozen=True)
class AnchoredWalkForwardSplitter(ExpandingWindowSplitter):
    """Expanding walk-forward splitter with the training start anchored at zero."""


@dataclass(frozen=True)
class PurgedTimeSeriesSplitter(ExpandingWindowSplitter):
    """Expanding splitter that purges label overlaps and applies embargo periods."""

    embargo: int = 1
    purge_overlapping_labels: bool = True


@dataclass(frozen=True)
class CrossSectionalSplitter(BaseSplitter):
    asset_col: str = "asset"
    validation_assets: tuple[str, ...] = ()
    test_assets: tuple[str, ...] = ()

    def split(self, data: object) -> Iterator[ValidationSplit]:
        frame = self._frame(data)
        if self.asset_col not in frame:
            raise ValueError(f"missing required column: {self.asset_col}")
        validation = tuple(frame.index[frame[self.asset_col].isin(self.validation_assets)])
        test = tuple(frame.index[frame[self.asset_col].isin(self.test_assets)])
        train = tuple(
            frame.index[~frame[self.asset_col].isin(self.validation_assets + self.test_assets)]
        )
        yield self._make_split(
            frame, train, validation, test, metadata={"holdout": "cross_sectional"}
        )


@dataclass(frozen=True)
class AssetUniverseHoldoutSplitter(CrossSectionalSplitter):
    """Hold out named assets from the model-selection and final-test universe."""


@dataclass(frozen=True)
class MarketPeriodHoldoutSplitter(BaseSplitter):
    period_col: str = "period"
    validation_periods: tuple[str, ...] = ()
    test_periods: tuple[str, ...] = ()

    def split(self, data: object) -> Iterator[ValidationSplit]:
        frame = self._frame(data)
        if self.period_col not in frame:
            raise ValueError(f"missing required column: {self.period_col}")
        validation = tuple(frame.index[frame[self.period_col].isin(self.validation_periods)])
        test = tuple(frame.index[frame[self.period_col].isin(self.test_periods)])
        train = tuple(
            frame.index[~frame[self.period_col].isin(self.validation_periods + self.test_periods)]
        )
        yield self._make_split(
            frame, train, validation, test, metadata={"holdout": "market_period"}
        )


@dataclass(frozen=True)
class CrisisPeriodStressTestSplitter(MarketPeriodHoldoutSplitter):
    """Specialized market-period holdout for named crisis/stress regimes."""


@dataclass(frozen=True)
class GeographicMarketHoldoutSplitter(BaseSplitter):
    geography_col: str = "geography"
    validation_geographies: tuple[str, ...] = ()
    test_geographies: tuple[str, ...] = ()

    def split(self, data: object) -> Iterator[ValidationSplit]:
        frame = self._frame(data)
        if self.geography_col not in frame:
            raise ValueError(f"missing required column: {self.geography_col}")
        validation = tuple(frame.index[frame[self.geography_col].isin(self.validation_geographies)])
        test = tuple(frame.index[frame[self.geography_col].isin(self.test_geographies)])
        train = tuple(
            frame.index[
                ~frame[self.geography_col].isin(self.validation_geographies + self.test_geographies)
            ]
        )
        yield self._make_split(
            frame, train, validation, test, metadata={"holdout": "geographic_market"}
        )


@dataclass(frozen=True)
class RefitFrequencySplitter(ExpandingWindowSplitter):
    refit_frequency: int = 5

    def split(self, data: object) -> Iterator[ValidationSplit]:
        splitter = ExpandingWindowSplitter(
            embargo=self.embargo,
            execution_delay=self.execution_delay,
            label_start_col=self.label_start_col,
            label_end_col=self.label_end_col,
            purge_overlapping_labels=self.purge_overlapping_labels,
            initial_train_size=self.initial_train_size,
            validation_size=self.validation_size,
            test_size=self.test_size,
            step=self.refit_frequency,
        )
        yield from splitter.split(data)


@dataclass(frozen=True)
class ExecutionDelaySensitivitySplitter(ExpandingWindowSplitter):
    execution_delays: tuple[int, ...] = (0, 1, 2)

    def split(self, data: object) -> Iterator[ValidationSplit]:
        for delay in self.execution_delays:
            delayed = ExpandingWindowSplitter(
                embargo=self.embargo,
                execution_delay=delay,
                label_start_col=self.label_start_col,
                label_end_col=self.label_end_col,
                purge_overlapping_labels=self.purge_overlapping_labels,
                initial_train_size=self.initial_train_size,
                validation_size=self.validation_size,
                test_size=self.test_size,
                step=self.step,
            )
            for split in delayed.split(data):
                yield ValidationSplit(
                    train=split.train,
                    validation=split.validation,
                    test=split.test,
                    embargo=split.embargo,
                    execution=split.execution,
                    windows=split.windows,
                    metadata={**split.metadata, "execution_delay": delay},
                )
