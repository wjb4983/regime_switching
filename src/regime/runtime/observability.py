"""Progress and process resource observability with no mandatory extras."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import platform
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import TypeVar

from regime.logging import log_event

T = TypeVar("T")


@dataclass(frozen=True)
class Progress:
    """Immutable progress update passed to user interfaces."""

    completed: int
    total: int | None
    elapsed_seconds: float

    @property
    def fraction(self) -> float | None:
        return None if self.total is None else self.completed / self.total if self.total else 1.0


def report_progress(
    items: Iterable[T],
    *,
    total: int | None = None,
    callback: Callable[[Progress], None] | None = None,
    every: int = 1,
) -> Iterator[T]:
    """Yield items and emit dependency-free progress updates."""
    if every < 1:
        raise ValueError("every must be at least one")
    started = time.monotonic()
    for completed, item in enumerate(items, 1):
        yield item
        if callback is not None and (completed % every == 0 or completed == total):
            callback(Progress(completed, total, time.monotonic() - started))


@dataclass(frozen=True)
class ResourceUsage:
    """Portable subset of process resource metrics."""

    pid: int
    user_cpu_seconds: float
    system_cpu_seconds: float
    max_rss_bytes: int


def resource_usage() -> ResourceUsage:
    """Capture current-process CPU and peak resident memory usage."""
    if importlib.util.find_spec("resource") is None:
        return ResourceUsage(os.getpid(), time.process_time(), 0.0, 0)
    resource = importlib.import_module("resource")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB, macOS bytes. Windows is not supported by resource, but
    # Python's resource module is only present where this metric is available.
    rss = int(usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024))
    return ResourceUsage(os.getpid(), usage.ru_utime, usage.ru_stime, rss)


def log_resource_usage(logger: logging.Logger, *, event: str = "resource_usage") -> ResourceUsage:
    """Capture and emit resource metrics through the project's JSON logger."""
    usage = resource_usage()
    log_event(logger, logging.INFO, event, **asdict(usage))
    return usage


__all__ = ["Progress", "ResourceUsage", "log_resource_usage", "report_progress", "resource_usage"]
