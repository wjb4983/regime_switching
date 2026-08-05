"""Array backend adapters for state-space models.

The reference algorithms deliberately compute with NumPy.  CuPy is imported only
when the optional ``gpu`` backend is explicitly requested.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ArrayBackend(Protocol):
    """Small adapter boundary used by future accelerated implementations."""

    name: str

    def asarray(self, value: Any) -> Any:
        """Create an array on the backend device."""

    def to_numpy(self, value: Any) -> np.ndarray[Any, Any]:
        """Copy an array to host memory."""


class NumPyBackend:
    """CPU backend shipped with the core package."""

    name = "numpy"

    def asarray(self, value: Any) -> np.ndarray[Any, Any]:
        return np.asarray(value, dtype=np.float64)

    def to_numpy(self, value: Any) -> np.ndarray[Any, Any]:
        return np.asarray(value)


class CuPyBackend:
    """Optional GPU array adapter (requires ``regime-switching[gpu]``).

    This adapter is an integration boundary, not a claim that the experimental
    estimators are GPU optimized: current estimators still use the CPU backend.
    """

    name = "cupy"

    def __init__(self) -> None:
        try:
            import cupy
        except ImportError as exc:
            raise ImportError(
                "CuPyBackend requires the optional 'gpu' extra and a compatible CUDA runtime"
            ) from exc
        self._module = cupy

    def asarray(self, value: Any) -> Any:
        return self._module.asarray(value, dtype=self._module.float64)

    def to_numpy(self, value: Any) -> np.ndarray[Any, Any]:
        return self._module.asnumpy(value)


def get_backend(name: str = "numpy") -> ArrayBackend:
    """Return an explicitly selected backend without eager optional imports."""
    if name == "numpy":
        return NumPyBackend()
    if name == "cupy":
        return CuPyBackend()
    raise ValueError(f"unknown array backend: {name!r}")
