"""Black-Scholes pricing, implied volatility, and Greeks without mandatory option libraries."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from statistics import NormalDist

from regime.data.options.types import OptionType

_N = NormalDist()


@dataclass(frozen=True, slots=True)
class BlackScholesInputs:
    spot: float
    strike: float
    tenor: float
    rate: float = 0.0
    dividend_yield: float = 0.0


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes_price(
    inputs: BlackScholesInputs, volatility: float, option_type: OptionType
) -> float:
    """Return the continuous-dividend Black-Scholes premium."""
    if inputs.tenor <= 0 or volatility <= 0:
        intrinsic = (
            max(inputs.spot - inputs.strike, 0.0)
            if option_type == "call"
            else max(inputs.strike - inputs.spot, 0.0)
        )
        return intrinsic
    d1, d2 = _d1_d2(inputs, volatility)
    df_r = exp(-inputs.rate * inputs.tenor)
    df_q = exp(-inputs.dividend_yield * inputs.tenor)
    if option_type == "call":
        return inputs.spot * df_q * _N.cdf(d1) - inputs.strike * df_r * _N.cdf(d2)
    return inputs.strike * df_r * _N.cdf(-d2) - inputs.spot * df_q * _N.cdf(-d1)


def implied_volatility(
    price: float,
    inputs: BlackScholesInputs,
    option_type: OptionType,
    *,
    lower: float = 1e-6,
    upper: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float | None:
    """Solve implied volatility by bounded bisection; return ``None`` outside no-arb bounds."""
    low_price = black_scholes_price(inputs, lower, option_type)
    high_price = black_scholes_price(inputs, upper, option_type)
    if price < low_price - tolerance or price > high_price + tolerance:
        return None
    lo, hi = lower, upper
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        mid_price = black_scholes_price(inputs, mid, option_type)
        if abs(mid_price - price) <= tolerance:
            return mid
        if mid_price < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def option_greeks(inputs: BlackScholesInputs, volatility: float, option_type: OptionType) -> Greeks:
    """Calculate Black-Scholes Greeks with continuous rates and dividends."""
    if inputs.tenor <= 0 or volatility <= 0:
        return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    d1, d2 = _d1_d2(inputs, volatility)
    pdf = _N.pdf(d1)
    df_r = exp(-inputs.rate * inputs.tenor)
    df_q = exp(-inputs.dividend_yield * inputs.tenor)
    gamma = df_q * pdf / (inputs.spot * volatility * sqrt(inputs.tenor))
    vega = inputs.spot * df_q * pdf * sqrt(inputs.tenor)
    if option_type == "call":
        delta = df_q * _N.cdf(d1)
        theta = (
            -(inputs.spot * df_q * pdf * volatility) / (2 * sqrt(inputs.tenor))
            - inputs.rate * inputs.strike * df_r * _N.cdf(d2)
            + inputs.dividend_yield * inputs.spot * df_q * _N.cdf(d1)
        )
        rho = inputs.strike * inputs.tenor * df_r * _N.cdf(d2)
    else:
        delta = -df_q * _N.cdf(-d1)
        theta = (
            -(inputs.spot * df_q * pdf * volatility) / (2 * sqrt(inputs.tenor))
            + inputs.rate * inputs.strike * df_r * _N.cdf(-d2)
            - inputs.dividend_yield * inputs.spot * df_q * _N.cdf(-d1)
        )
        rho = -inputs.strike * inputs.tenor * df_r * _N.cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta / 365.0, rho=rho / 100.0)


def _d1_d2(inputs: BlackScholesInputs, volatility: float) -> tuple[float, float]:
    vol_sqrt_t = volatility * sqrt(inputs.tenor)
    d1 = (
        log(inputs.spot / inputs.strike)
        + (inputs.rate - inputs.dividend_yield + 0.5 * volatility**2) * inputs.tenor
    ) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t
