"""Shared configuration for econometric regime-switching models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, NonNegativeFloat, PositiveInt

from regime.models.base import RegimeModelConfig


class EconometricModelConfig(RegimeModelConfig):
    """Configuration knobs common to econometric regime-switching estimators.

    The fields are intentionally broad enough to cover statsmodels-backed Markov
    switching models and fragile/custom estimators that may live behind optional
    adapter modules.
    """

    ar_order: PositiveInt = 1
    transition_dynamics: Literal["fixed", "time_varying", "duration", "exogenous"] = "fixed"
    variance_model: Literal["constant", "switching", "garch", "har", "stochastic"] = "constant"
    innovation_distribution: Literal["gaussian", "student_t", "skew_t", "empirical"] = "gaussian"
    regularization: NonNegativeFloat = 0.0
    optimizer: str = "lbfgs"
    max_iter: PositiveInt = 100
    tol: float = Field(default=1e-6, gt=0.0)
    threshold: float | None = None
    transition_variable: str | int | None = None
    smoothness: float = Field(default=1.0, gt=0.0)
    exog_switching: bool = True
    switching_variance: bool = True
    search_reps: int = Field(default=0, ge=0)
    covariance_type: str = "approx"
    optimizer_options: dict[str, Any] = Field(default_factory=dict)
