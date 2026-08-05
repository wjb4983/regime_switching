"""Feature engineering interfaces for regime switching workflows."""

from regime.features import cross_sectional, liquidity, macro, options, returns, risk
from regime.features.registry import (
    FeatureBuildConfig,
    FeatureBuilder,
    FeatureBuildResult,
    FeatureDefinition,
    FeatureRegistry,
    FeatureSemantics,
    FittingRequirement,
    MissingValuePolicy,
    OutputField,
    ScalingMethod,
)

__all__ = [
    "FeatureBuildConfig",
    "FeatureBuildResult",
    "FeatureBuilder",
    "FeatureDefinition",
    "FeatureRegistry",
    "FeatureSemantics",
    "FittingRequirement",
    "MissingValuePolicy",
    "OutputField",
    "ScalingMethod",
    "cross_sectional",
    "liquidity",
    "macro",
    "options",
    "returns",
    "risk",
]
