"""Option surface, risk-premium, implied-correlation, and hedged-return features."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import mean

from regime.data.options.pricing import Greeks


@dataclass(frozen=True, slots=True)
class OptionFactors:
    smile: float
    skew: float
    curvature: float
    term_structure: float


def build_delta_moneyness_tenor_grid(
    *, deltas: Iterable[float], moneyness: Iterable[float], tenors: Iterable[float]
) -> tuple[tuple[float, float, float], ...]:
    """Construct a stable delta/moneyness/tenor grid for surface features."""
    return tuple((d, m, t) for t in tenors for m in moneyness for d in deltas)


def compute_option_factors(rows: Sequence[tuple[float, float, float]]) -> OptionFactors:
    """Compute smile, skew, curvature, and term-structure factors from (tenor, delta, iv)."""
    if not rows:
        raise ValueError("rows must not be empty")
    by_delta = {delta: iv for _, delta, iv in rows}
    atm = min(rows, key=lambda row: abs(abs(row[1]) - 0.5))[2]
    wing_ivs = [iv for _, delta, iv in rows if abs(delta) <= 0.25 or abs(delta) >= 0.75]
    put_25 = by_delta.get(-0.25)
    call_25 = by_delta.get(0.25)
    front = min(rows, key=lambda row: row[0])[2]
    back = max(rows, key=lambda row: row[0])[2]
    smile = mean(wing_ivs) - atm if wing_ivs else 0.0
    skew = (put_25 - call_25) if put_25 is not None and call_25 is not None else 0.0
    curvature = smile - skew / 2.0
    return OptionFactors(smile=smile, skew=skew, curvature=curvature, term_structure=back - front)


def implied_correlation(
    index_variance: float, component_variances: Sequence[float], weights: Sequence[float]
) -> float:
    """Estimate average implied correlation from index and component implied variances."""
    if len(component_variances) != len(weights) or len(weights) < 2:
        raise ValueError("component_variances and weights must have equal length >= 2")
    weighted_component = sum((w**2) * v for w, v in zip(weights, component_variances, strict=True))
    cross_weight = sum(
        weights[i] * weights[j] * sqrt(component_variances[i] * component_variances[j])
        for i in range(len(weights))
        for j in range(i + 1, len(weights))
    )
    return (index_variance - weighted_component) / (2.0 * cross_weight)


def variance_risk_premium_inputs(
    *, implied_volatility: float, realized_volatility: float
) -> dict[str, float]:
    """Return annualized variance inputs used for variance-risk-premium features."""
    return {
        "implied_variance": implied_volatility**2,
        "realized_variance": realized_volatility**2,
        "vrp": implied_volatility**2 - realized_volatility**2,
    }


def delta_hedged_return_inputs(
    *, option_return: float, underlying_return: float, greeks: Greeks
) -> dict[str, float]:
    """Return inputs for a one-period delta-hedged option return calculation."""
    hedge_return = option_return - greeks.delta * underlying_return
    return {
        "option_return": option_return,
        "underlying_return": underlying_return,
        "delta": greeks.delta,
        "delta_hedged_return": hedge_return,
    }
