"""Recurring-state jump segmentation models.

Jump models reuse a fixed set of state labels across non-contiguous time spans;
they are distinct from change-point models that produce one-off chronological
segments.
"""

from regime.models.jump.models import (
    JumpSegmentationConfig,
    JumpSegmentationModel,
    JumpSegmentationResult,
    align_jump_labels,
)

__all__ = [
    "JumpSegmentationConfig",
    "JumpSegmentationModel",
    "JumpSegmentationResult",
    "align_jump_labels",
]
