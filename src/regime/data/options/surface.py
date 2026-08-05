"""Volatility surface construction and bilinear interpolation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SurfacePoint:
    tenor: float
    moneyness: float
    volatility: float


@dataclass(frozen=True, slots=True)
class VolSurface:
    points: tuple[SurfacePoint, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[float, float, float]]) -> VolSurface:
        return cls(tuple(SurfacePoint(*row) for row in rows))


def interpolate_surface(surface: VolSurface, *, tenor: float, moneyness: float) -> float:
    """Interpolate implied volatility by inverse-distance weighting over surface points."""
    if not surface.points:
        raise ValueError("surface must contain at least one point")
    weighted_sum = 0.0
    weight_total = 0.0
    for point in surface.points:
        distance = abs(point.tenor - tenor) + abs(point.moneyness - moneyness)
        if distance == 0:
            return point.volatility
        weight = 1.0 / distance
        weighted_sum += weight * point.volatility
        weight_total += weight
    return weighted_sum / weight_total
