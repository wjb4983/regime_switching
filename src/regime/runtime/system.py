"""Runtime initialization, hardware discovery, and interruption control."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GPUInfo:
    available: bool
    backend: str | None = None
    count: int = 0


def detect_gpu() -> GPUInfo:
    """Detect optional Torch or CuPy GPUs without making either a dependency."""
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")

        if torch.cuda.is_available():
            return GPUInfo(True, "torch", torch.cuda.device_count())
    if importlib.util.find_spec("cupy") is not None:
        cupy = importlib.import_module("cupy")

        try:
            count = cupy.cuda.runtime.getDeviceCount()
        except cupy.cuda.runtime.CUDARuntimeError:
            count = 0
        if count:
            return GPUInfo(True, "cupy", count)
    return GPUInfo(False)


def gpu_available() -> bool:
    """Return whether a supported GPU backend can access a device."""
    return detect_gpu().available


def initialize_seed(seed: int, *, deterministic: bool = True) -> np.random.Generator:
    """Seed core and installed optional numerical frameworks consistently."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    generator = np.random.default_rng(seed)
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
    return generator


class InterruptionError(RuntimeError):
    """Raised at a cooperative interruption checkpoint."""


class InterruptionHandler:
    """Record SIGINT/SIGTERM and allow work to stop at safe checkpoints."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handle(self, signal_number: int, _frame: FrameType | None) -> None:
        self.requested = True
        self.signal_number = signal_number

    def checkpoint(self) -> None:
        if self.requested:
            raise InterruptionError(f"received signal {self.signal_number}")


@contextmanager
def graceful_interrupts() -> Iterator[InterruptionHandler]:
    """Temporarily install cooperative signal handlers and restore predecessors."""
    handler = InterruptionHandler()
    supported = [
        candidate for candidate in (signal.SIGINT, signal.SIGTERM) if candidate is not None
    ]
    previous: dict[signal.Signals, Any] = {}
    try:
        for candidate in supported:
            previous[candidate] = signal.getsignal(candidate)
            signal.signal(candidate, handler.handle)
        yield handler
    finally:
        for candidate, old_handler in previous.items():
            signal.signal(candidate, old_handler)


__all__ = [
    "GPUInfo",
    "InterruptionError",
    "InterruptionHandler",
    "detect_gpu",
    "gpu_available",
    "graceful_interrupts",
    "initialize_seed",
]
