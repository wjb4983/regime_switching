"""Experimental switching state-space models and backend adapters."""

from regime.models.state_space.backends import ArrayBackend, CuPyBackend, NumPyBackend, get_backend
from regime.models.state_space.models import (
    ExplicitDurationSwitchingLinearDynamicalSystem,
    RecurrentSwitchingLinearDynamicalSystem,
    StateSpaceConfig,
    StateSpaceParameters,
    StateSpaceResult,
    SwitchingDynamicFactorModel,
    SwitchingKalmanFilter,
    SwitchingLinearDynamicalSystem,
)

__all__ = [
    "ArrayBackend",
    "CuPyBackend",
    "ExplicitDurationSwitchingLinearDynamicalSystem",
    "NumPyBackend",
    "RecurrentSwitchingLinearDynamicalSystem",
    "StateSpaceConfig",
    "StateSpaceParameters",
    "StateSpaceResult",
    "SwitchingDynamicFactorModel",
    "SwitchingKalmanFilter",
    "SwitchingLinearDynamicalSystem",
    "get_backend",
]
