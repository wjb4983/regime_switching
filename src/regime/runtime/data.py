"""Lazy and memory-mapped access for large local arrays."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MemmapArray:
    """Serializable array descriptor that opens storage only when requested."""

    path: str | Path
    dtype: str | npt.DTypeLike
    shape: tuple[int, ...]
    mode: Literal["r", "c", "r+", "w+"] = "r"
    offset: int = 0
    order: Literal["C", "F"] = "C"

    def open(self) -> np.memmap[Any, Any]:
        """Open the mapping in the current worker process."""
        return np.memmap(
            str(self.path),
            dtype=self.dtype,
            mode=self.mode,
            offset=self.offset,
            shape=self.shape,
            order=self.order,
        )

    def chunks(self, size: int, *, axis: int = 0) -> Iterator[npt.NDArray[Any]]:
        """Yield views over bounded regions, reopening safely in this process."""
        if size < 1:
            raise ValueError("size must be at least one")
        if not 0 <= axis < len(self.shape):
            raise ValueError("axis is out of bounds")
        array = self.open()
        for start in range(0, self.shape[axis], size):
            selection = [slice(None)] * len(self.shape)
            selection[axis] = slice(start, min(start + size, self.shape[axis]))
            yield array[tuple(selection)]


__all__ = ["MemmapArray"]
