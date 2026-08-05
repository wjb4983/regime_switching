from __future__ import annotations

import numpy as np
import pytest

from regime.models.base import UnsupportedModelOperation
from regime.models.changepoint import (
    BayesianOnlineChangePointConfig,
    BayesianOnlineChangePointDetector,
    BinarySegmentationConfig,
    BinarySegmentationDetector,
    CUSUMConfig,
    CUSUMDetector,
    DistributionalChangePointConfig,
    DistributionalChangePointDetector,
    PageHinkleyConfig,
    PageHinkleyDetector,
    PELTConfig,
    PELTDetector,
    RupturesAdapter,
)


def shifted_series() -> np.ndarray:
    return np.concatenate([np.zeros(30), np.ones(35) * 5.0, np.ones(30) * -2.0])


@pytest.mark.parametrize(
    "detector",
    [
        CUSUMDetector(CUSUMConfig(threshold=4.0, min_size=5, tolerance=8)),
        PageHinkleyDetector(PageHinkleyConfig(threshold=8.0, min_size=5, tolerance=8)),
        BinarySegmentationDetector(
            BinarySegmentationConfig(max_breakpoints=2, threshold=10.0, min_size=5, tolerance=3)
        ),
        PELTDetector(PELTConfig(penalty=20.0, min_size=5, tolerance=3)),
        DistributionalChangePointDetector(
            DistributionalChangePointConfig(window=8, threshold=0.5, tolerance=8)
        ),
        BayesianOnlineChangePointDetector(
            BayesianOnlineChangePointConfig(hazard=0.08, threshold=0.07, min_size=5, tolerance=10)
        ),
    ],
)
def test_changepoint_detectors_emit_boundary_events(detector) -> None:
    result = detector.detect(shifted_series(), ground_truth=[30, 65])

    assert len(result.scores) == 95
    assert len(result.change_probabilities) == 95
    assert all(0.0 <= probability <= 1.0 for probability in result.change_probabilities)
    assert result.boundary_indices
    assert result.segment_ids[0] == 0
    assert max(result.segment_ids) >= 1
    assert result.detection_delays is not None
    assert result.false_alarm_rate is not None
    assert result.metadata is not None
    assert result.metadata["semantics"] == "boundary_event"


def test_ruptures_adapter_makes_missing_optional_dependency_explicit() -> None:
    adapter = RupturesAdapter("Unknown")
    with pytest.raises(UnsupportedModelOperation):
        adapter.detect(np.arange(10, dtype=float).reshape(-1, 1), np.arange(10))
