"""Boundary/event change-point detectors.

These models detect one-off boundaries between time-series segments.  They do not
produce recurring latent regime classes; ``segment_ids`` are chronological segment
identifiers induced by detected boundaries.
"""

from regime.models.changepoint.detectors import (
    BayesianOnlineChangePointConfig,
    BayesianOnlineChangePointDetector,
    BinarySegmentationConfig,
    BinarySegmentationDetector,
    ChangePointDetectionResult,
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

__all__ = [
    "BayesianOnlineChangePointConfig",
    "BayesianOnlineChangePointDetector",
    "BinarySegmentationConfig",
    "BinarySegmentationDetector",
    "CUSUMConfig",
    "CUSUMDetector",
    "ChangePointDetectionResult",
    "DistributionalChangePointConfig",
    "DistributionalChangePointDetector",
    "PELTConfig",
    "PELTDetector",
    "PageHinkleyConfig",
    "PageHinkleyDetector",
    "RupturesAdapter",
]
