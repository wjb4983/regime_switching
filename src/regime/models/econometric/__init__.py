"""Econometric regime-switching model adapters and lightweight research baselines."""

from regime.models.econometric.config import EconometricModelConfig
from regime.models.econometric.custom import (
    RegimeSwitchingCopula,
    RegimeSwitchingCorrelation,
    RegimeSwitchingJumpDiffusion,
    SmoothTransitionAutoregression,
    SwitchingStochasticVolatility,
    ThresholdAutoregression,
)
from regime.models.econometric.markov import (
    MarkovSwitchingAR,
    MarkovSwitchingGARCH,
    MarkovSwitchingHAR,
    MarkovSwitchingRegression,
    MarkovSwitchingVAR,
)

__all__ = [
    "EconometricModelConfig",
    "MarkovSwitchingAR",
    "MarkovSwitchingGARCH",
    "MarkovSwitchingHAR",
    "MarkovSwitchingRegression",
    "MarkovSwitchingVAR",
    "RegimeSwitchingCopula",
    "RegimeSwitchingCorrelation",
    "RegimeSwitchingJumpDiffusion",
    "SmoothTransitionAutoregression",
    "SwitchingStochasticVolatility",
    "ThresholdAutoregression",
]
